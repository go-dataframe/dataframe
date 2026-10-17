package dataframe

import (
	"fmt"
	"math"
	"slices"

	"github.com/go-ndarray/ndarray"
)

// DataFrame is a table of named columns of equal length sharing one Index of
// row labels. Each column has its own dtype; numeric columns are go-ndarray
// arrays. A DataFrame is immutable: operations return new frames, sharing
// the columns they do not change.
type DataFrame struct {
	index *Index
	names []string
	cols  []column
}

// Column names one column of data for New.
type Column struct {
	Name string
	Data any
}

// Col returns a Column for New. data is anything NewSeries accepts, or a
// *Series (whose index then must match the frame's).
func Col(name string, data any) Column { return Column{name, data} }

// New returns a frame of the given columns, in order, with a range index.
//
//	df, err := dataframe.New(
//		dataframe.Col("city", []string{"Lyon", "Paris"}),
//		dataframe.Col("pop", []int{522_000, 2_103_000}),
//	)
func New(cols ...Column) (*DataFrame, error) {
	df := &DataFrame{}
	n := -1
	for _, c := range cols {
		cc, err := fromAny(c.Data)
		if err != nil {
			return nil, fmt.Errorf("column %q: %w", c.Name, err)
		}
		if n >= 0 && cc.len() != n {
			return nil, fmt.Errorf("%w: column %q has %d rows, not %d", ErrLength, c.Name, cc.len(), n)
		}
		n = cc.len()
		if s, ok := c.Data.(*Series); ok && df.index == nil && !s.index.IsRange() {
			df.index = s.index
		}
		df.names = append(df.names, c.Name)
		df.cols = append(df.cols, cc)
	}
	if df.index == nil {
		df.index = NewRangeIndex(max(n, 0))
	}
	return df, nil
}

// FromSeries returns a frame of the series as columns, aligned on their
// indexes: when they differ, the rows are the union of the labels (in
// sorted order, as pandas builds a frame from a dict of series), and a series
// lacking a label gets a missing value there.
func FromSeries(series ...*Series) (*DataFrame, error) {
	if len(series) == 0 {
		return &DataFrame{index: NewRangeIndex(0)}, nil
	}
	ix := series[0].index
	same := true
	for _, s := range series[1:] {
		same = same && s.index.equal(ix)
	}
	df := &DataFrame{index: ix}
	if !same {
		var err error
		ix, err = unionIndex(series)
		if err != nil {
			return nil, err
		}
		df.index = ix
	}
	for _, s := range series {
		c := s.c
		if !same {
			idx, err := alignTo(s.index, ix)
			if err != nil {
				return nil, err
			}
			c = c.take(idx)
		}
		df.names = append(df.names, s.name)
		df.cols = append(df.cols, c)
	}
	return df, nil
}

// FromArray returns a frame of the columns of a two-dimensional array, named
// by names (or "0", "1", ... when names is empty).
func FromArray(a *ndarray.Array, names ...string) (*DataFrame, error) {
	if a.Ndim() != 2 {
		return nil, fmt.Errorf("%w: FromArray needs a 2-D array, got shape %v", ErrArgument, a.Shape())
	}
	rows, ncols := a.Shape()[0], a.Shape()[1]
	if len(names) == 0 {
		for j := range ncols {
			names = append(names, fmt.Sprint(j))
		}
	}
	if len(names) != ncols {
		return nil, fmt.Errorf("%w: %d names for %d columns", ErrLength, len(names), ncols)
	}
	t := a.Transpose().Copy()
	df := &DataFrame{index: NewRangeIndex(rows), names: append([]string(nil), names...)}
	for j := range ncols {
		row, _ := t.Slice(ndarray.A(j))
		c, _ := fromArray(row.Copy())
		df.cols = append(df.cols, c)
	}
	return df, nil
}

func (df *DataFrame) derive(cols []column) *DataFrame {
	return &DataFrame{index: df.index, names: df.names, cols: cols}
}

// Shape returns the number of rows and of columns.
func (df *DataFrame) Shape() (rows, cols int) { return df.index.Len(), len(df.cols) }

// Len returns the number of rows.
func (df *DataFrame) Len() int { return df.index.Len() }

// Columns returns the column names, in order.
func (df *DataFrame) Columns() []string { return append([]string(nil), df.names...) }

// Index returns the row labels.
func (df *DataFrame) Index() *Index { return df.index }

// DTypes returns the dtype of each column, in column order.
func (df *DataFrame) DTypes() []DType {
	out := make([]DType, len(df.cols))
	for i, c := range df.cols {
		out[i] = c.dtype()
	}
	return out
}

// DTypesSeries returns the dtypes as a string series indexed by column name,
// the way pandas shows df.dtypes.
func (df *DataFrame) DTypesSeries() *Series {
	names := make([]string, len(df.cols))
	for i, c := range df.cols {
		names[i] = c.dtype().String()
	}
	ix, _ := NewIndex("", df.Columns())
	return &Series{index: ix, c: newCol(String, names, nil)}
}

// pos returns the position of a column.
func (df *DataFrame) pos(name string) (int, error) {
	if i := slices.Index(df.names, name); i >= 0 {
		return i, nil
	}
	return 0, fmt.Errorf("%w: %q", ErrColumn, name)
}

// Column returns one column as a series sharing the frame's index.
func (df *DataFrame) Column(name string) (*Series, error) {
	i, err := df.pos(name)
	if err != nil {
		return nil, err
	}
	return &Series{name: name, index: df.index, c: df.cols[i]}, nil
}

// ColumnAt returns the column at position i.
func (df *DataFrame) ColumnAt(i int) (*Series, error) {
	p, err := position(i, len(df.cols))
	if err != nil {
		return nil, err
	}
	return &Series{name: df.names[p], index: df.index, c: df.cols[p]}, nil
}

// Select returns the frame restricted to the named columns, in that order.
func (df *DataFrame) Select(names ...string) (*DataFrame, error) {
	out := &DataFrame{index: df.index}
	for _, n := range names {
		i, err := df.pos(n)
		if err != nil {
			return nil, err
		}
		out.names = append(out.names, n)
		out.cols = append(out.cols, df.cols[i])
	}
	return out, nil
}

// Drop returns the frame without the named columns.
func (df *DataFrame) Drop(names ...string) (*DataFrame, error) {
	for _, n := range names {
		if _, err := df.pos(n); err != nil {
			return nil, err
		}
	}
	out := &DataFrame{index: df.index}
	for i, n := range df.names {
		if !slices.Contains(names, n) {
			out.names = append(out.names, n)
			out.cols = append(out.cols, df.cols[i])
		}
	}
	return out, nil
}

// WithColumn returns the frame with a column set: replaced in place if the
// name exists, appended otherwise. data is a *Series (aligned on the frame's
// index by label, as pandas assigns one), a slice or array of the frame's
// length, or a scalar repeated on every row.
func (df *DataFrame) WithColumn(name string, data any) (*DataFrame, error) {
	c, err := df.columnFor(data)
	if err != nil {
		return nil, err
	}
	out := &DataFrame{index: df.index, names: slices.Clone(df.names), cols: slices.Clone(df.cols)}
	if i := slices.Index(df.names, name); i >= 0 {
		out.cols[i] = c
	} else {
		out.names = append(out.names, name)
		out.cols = append(out.cols, c)
	}
	return out, nil
}

// columnFor turns assignable data into a column of the frame's length.
func (df *DataFrame) columnFor(data any) (column, error) {
	n := df.Len()
	if s, ok := data.(*Series); ok {
		if s.index.equal(df.index) {
			return s.c, nil
		}
		idx, err := alignTo(s.index, df.index)
		if err != nil {
			return nil, err
		}
		return s.c.take(idx), nil
	}
	c, err := fromAny(data)
	if err != nil {
		c, err = fullCol(data, n)
		if err != nil {
			return nil, err
		}
	}
	if c.len() != n {
		return nil, fmt.Errorf("%w: %d values for %d rows", ErrLength, c.len(), n)
	}
	return c, nil
}

// Head returns the first n rows (with a negative n, all but the last -n).
func (df *DataFrame) Head(n int) *DataFrame {
	i, j := headRange(n, df.Len())
	return df.slice(i, j)
}

// Tail returns the last n rows (with a negative n, all but the first -n).
func (df *DataFrame) Tail(n int) *DataFrame {
	i, j := tailRange(n, df.Len())
	return df.slice(i, j)
}

func (df *DataFrame) slice(i, j int) *DataFrame {
	cols := make([]column, len(df.cols))
	for k, c := range df.cols {
		cols[k] = c.slice(i, j)
	}
	return &DataFrame{index: df.index.slice(i, j), names: df.names, cols: cols}
}

func (df *DataFrame) take(idx []int) *DataFrame {
	cols := make([]column, len(df.cols))
	eachColumn(len(idx), len(df.cols), func(k int) { cols[k] = df.cols[k].take(idx) })
	return &DataFrame{index: df.index.take(idx), names: df.names, cols: cols}
}

// takeOwned is take for positions, none negative, that the caller hands
// over: a default range index then keeps them as its labels.
func (df *DataFrame) takeOwned(idx []int) *DataFrame {
	cols := make([]column, len(df.cols))
	eachColumn(len(idx), len(df.cols), func(k int) { cols[k] = df.cols[k].take(idx) })
	return &DataFrame{index: df.index.takeOwned(idx), names: df.names, cols: cols}
}

// Copy returns a deep copy of the frame.
func (df *DataFrame) Copy() *DataFrame {
	idx := make([]int, df.Len())
	for i := range idx {
		idx[i] = i
	}
	cols := make([]column, len(df.cols))
	for k, c := range df.cols {
		cols[k] = c.take(idx)
	}
	return &DataFrame{index: df.index, names: slices.Clone(df.names), cols: cols}
}

// Rename returns the frame with columns renamed by the map; names it does
// not mention are kept, and an unknown key is ignored, as in pandas.
func (df *DataFrame) Rename(names map[string]string) *DataFrame {
	out := &DataFrame{index: df.index, names: slices.Clone(df.names), cols: df.cols}
	for i, n := range out.names {
		if m, ok := names[n]; ok {
			out.names[i] = m
		}
	}
	return out
}

// SetColumns returns the frame with all column names replaced.
func (df *DataFrame) SetColumns(names ...string) (*DataFrame, error) {
	if len(names) != len(df.cols) {
		return nil, fmt.Errorf("%w: %d names for %d columns", ErrLength, len(names), len(df.cols))
	}
	return &DataFrame{index: df.index, names: slices.Clone(names), cols: df.cols}, nil
}

// Values returns the frame as a two-dimensional array, rows by columns. All
// columns must be numeric or bool; the dtype is the one NumPy would pick for
// them all (an int column with missing values counts as float64), and
// missing values are NaN.
func (df *DataFrame) Values() (*ndarray.Array, error) {
	rows, ncols := df.Shape()
	if ncols == 0 {
		return ndarray.New(rows, 0)
	}
	var dt ndarray.DType
	for k, c := range df.cols {
		if !c.dtype().IsNumeric() {
			return nil, fmt.Errorf("%w: column %q is %v", ErrDType, df.names[k], c.dtype())
		}
		d := ndarray.DType(c.dtype())
		if c.naCount() > 0 && !c.dtype().isFloat() && c.dtype() != Complex64 && c.dtype() != Complex128 {
			d = ndarray.Float64
		}
		if k == 0 {
			dt = d
		} else {
			dt = ndarray.ResultType(dt, d)
		}
	}
	parts := make([]*ndarray.Array, ncols)
	for k, c := range df.cols {
		a, _ := c.ndarray()
		a = a.AsType(dt)
		if v := c.validity(); v != nil {
			for i := range rows {
				if !v.get(i) {
					a.Set(math.NaN(), i)
				}
			}
		}
		parts[k] = a
	}
	m, _ := ndarray.Stack(parts, 1)
	return m, nil
}

// Series returns every column as a series, in order.
func (df *DataFrame) Series() []*Series {
	out := make([]*Series, len(df.cols))
	for i := range df.cols {
		out[i] = &Series{name: df.names[i], index: df.index, c: df.cols[i]}
	}
	return out
}
