package dataframe

import (
	"fmt"

	"github.com/go-ndarray/ndarray"
)

// Series is one named, labelled column: values of one dtype, with missing
// values marked in a validity bitmap, and an Index of row labels. A Series is
// immutable: every operation returns a new one, sharing storage where it can.
type Series struct {
	name  string
	index *Index
	c     column
}

// NewSeries returns a series over a copy of data, with a range index. data
// is a slice of any Go number type, []bool, []string, []time.Time, []any
// (nil is missing; the dtype is inferred), or a one-dimensional
// *ndarray.Array, which is shared rather than copied when it is contiguous.
func NewSeries(name string, data any) (*Series, error) {
	c, err := fromAny(data)
	if err != nil {
		return nil, err
	}
	return &Series{name: name, index: NewRangeIndex(c.len()), c: c}, nil
}

// NewSeriesIndexed is NewSeries with the given row labels.
func NewSeriesIndexed(name string, data any, index *Index) (*Series, error) {
	s, err := NewSeries(name, data)
	if err != nil {
		return nil, err
	}
	if index.Len() != s.Len() {
		return nil, fmt.Errorf("%w: %d values for an index of %d", ErrLength, s.Len(), index.Len())
	}
	s.index = index
	return s, nil
}

func (s *Series) with(c column) *Series { return &Series{name: s.name, index: s.index, c: c} }

// Name returns the series name.
func (s *Series) Name() string { return s.name }

// Rename returns the series under another name.
func (s *Series) Rename(name string) *Series {
	return &Series{name: name, index: s.index, c: s.c}
}

// Len returns the number of values.
func (s *Series) Len() int { return s.c.len() }

// DType returns the dtype of the values.
func (s *Series) DType() DType { return s.c.dtype() }

// Index returns the row labels.
func (s *Series) Index() *Index { return s.index }

// SetIndex returns the series with other row labels of the same length.
func (s *Series) SetIndex(ix *Index) (*Series, error) {
	if ix.Len() != s.Len() {
		return nil, fmt.Errorf("%w: index of %d for %d values", ErrLength, ix.Len(), s.Len())
	}
	return &Series{name: s.name, index: ix, c: s.c}, nil
}

// ResetIndex returns the series with a range index.
func (s *Series) ResetIndex() *Series {
	return &Series{name: s.name, index: NewRangeIndex(s.Len()), c: s.c}
}

// Values returns the values as a go-ndarray array of the series' dtype; a
// datetime or period series gives its int64 storage. A series that owns its
// storage hands it out without copying. Missing values are left as they are
// stored (NaN for floats, zero otherwise); IsNA tells them apart. A string or
// category series has no array form.
func (s *Series) Values() (*ndarray.Array, error) {
	if a, ok := s.c.ndarray(); ok {
		return a, nil
	}
	return nil, fmt.Errorf("%w: a %v series has no array form", ErrDType, s.DType())
}

// Floats returns the values as float64s, missing values as NaN. It accepts
// every real number dtype and bool.
func (s *Series) Floats() ([]float64, error) {
	f, ok := floatsOf(s.c)
	if !ok {
		return nil, fmt.Errorf("%w: %v is not a real number dtype", ErrDType, s.DType())
	}
	return f, nil
}

// Strings returns every value formatted as text; missing values give "".
func (s *Series) Strings() []string {
	out := make([]string, s.Len())
	for i := range out {
		if !s.c.isNA(i) {
			out[i] = s.c.str(i)
		}
	}
	return out
}

// Items returns every value as a Go value, nil when missing.
func (s *Series) Items() []any {
	out := make([]any, s.Len())
	for i := range out {
		out[i] = s.c.value(i)
	}
	return out
}

// At returns the value at position i (negative counts from the end), nil
// when missing.
func (s *Series) At(i int) (any, error) {
	p, err := position(i, s.Len())
	if err != nil {
		return nil, err
	}
	return s.c.value(p), nil
}

// Get returns the value with row label label. A label held by several rows
// gives the first.
func (s *Series) Get(label any) (any, error) {
	p, err := s.index.Get(label)
	if err != nil {
		return nil, err
	}
	return s.c.value(p[0]), nil
}

func position(i, n int) (int, error) {
	if i < 0 {
		i += n
	}
	if i < 0 || i >= n {
		return 0, fmt.Errorf("%w: %d of %d", ErrIndex, i, n)
	}
	return i, nil
}

// Head returns the first n values (all of them if there are fewer; with a
// negative n, all but the last -n).
func (s *Series) Head(n int) *Series {
	i, j := headRange(n, s.Len())
	return s.slice(i, j)
}

// Tail returns the last n values (with a negative n, all but the first -n).
func (s *Series) Tail(n int) *Series {
	i, j := tailRange(n, s.Len())
	return s.slice(i, j)
}

func headRange(n, l int) (int, int) {
	if n < 0 {
		n = max(l+n, 0)
	}
	return 0, min(n, l)
}

func tailRange(n, l int) (int, int) {
	if n < 0 {
		n = max(l+n, 0)
	}
	return l - min(n, l), l
}

func (s *Series) slice(i, j int) *Series {
	return &Series{name: s.name, index: s.index.slice(i, j), c: s.c.slice(i, j)}
}

func (s *Series) take(idx []int) *Series {
	return &Series{name: s.name, index: s.index.take(idx), c: s.c.take(idx)}
}

// Copy returns a deep copy.
func (s *Series) Copy() *Series {
	idx := make([]int, s.Len())
	for i := range idx {
		idx[i] = i
	}
	return &Series{name: s.name, index: s.index, c: s.c.take(idx)}
}

// IsNA returns a bool series, true where a value is missing (NaN counts as
// missing in a float series, as in pandas).
func (s *Series) IsNA() *Series {
	c := makeCol[bool](Bool, s.Len())
	for i := range c.data {
		c.data[i] = s.c.isNA(i)
	}
	return s.with(c)
}

// NotNA is the negation of IsNA.
func (s *Series) NotNA() *Series {
	c := makeCol[bool](Bool, s.Len())
	for i := range c.data {
		c.data[i] = !s.c.isNA(i)
	}
	return s.with(c)
}

// ToFrame returns a one-column frame of the series.
func (s *Series) ToFrame() *DataFrame {
	return &DataFrame{index: s.index, names: []string{s.name}, cols: []column{s.c}}
}

// bools returns a bool series' values, missing as false.
func (s *Series) bools() ([]bool, error) {
	c, ok := s.c.(*col[bool])
	if !ok {
		return nil, fmt.Errorf("%w: a mask must be bool, got %v", ErrDType, s.DType())
	}
	if c.valid == nil {
		return c.data, nil // read only
	}
	out := make([]bool, len(c.data))
	for i, b := range c.data {
		out[i] = b && c.valid.get(i)
	}
	return out, nil
}

// NewSeriesMasked is NewSeries with a validity mask: valid[i] false marks
// value i missing (its slot in data is kept but ignored). valid must be as
// long as data.
func NewSeriesMasked(name string, data any, valid []bool) (*Series, error) {
	s, err := NewSeries(name, data)
	if err != nil {
		return nil, err
	}
	if len(valid) != s.Len() {
		return nil, fmt.Errorf("%w: %d validity flags for %d values", ErrLength, len(valid), s.Len())
	}
	b := allValid(len(valid))
	missing := false
	for i, v := range valid {
		if !v {
			b.clear(i)
			missing = true
		}
	}
	if missing {
		s.c = s.c.withValidity(b)
	}
	return s, nil
}

// Valid returns, for each value, whether it is present (the negation of
// IsNA, as a plain slice).
func (s *Series) Valid() []bool {
	out := make([]bool, s.Len())
	for i := range out {
		out[i] = !s.c.isNA(i)
	}
	return out
}
