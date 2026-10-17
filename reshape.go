package dataframe

import (
	"fmt"
	"slices"
)

// commonOf returns the dtype a set of columns share: their common type
// when they have one, string otherwise (pandas would make an object column).
func commonOf(cols []column) DType {
	dt := cols[0].dtype()
	for _, c := range cols[1:] {
		var err error
		if dt, err = commonType(dt, c.dtype()); err != nil {
			return String
		}
	}
	if dt == Category {
		return String
	}
	return dt
}

// stackAll returns the columns converted to their common dtype and laid end
// to end: element j*n+i is row i of column j.
func stackAll(cols []column) column {
	dt := commonOf(cols)
	out, _ := cast(cols[0], dt)
	for _, c := range cols[1:] {
		x, _ := cast(c, dt)
		out = out.appendCol(x)
	}
	return out
}

// T returns the transpose: one column per row, named after the row's label,
// and one row per column, labelled by its name. The values take the
// columns' common dtype, or become strings when they have none (pandas
// makes an object column there).
func (df *DataFrame) T() *DataFrame {
	nrows, ncols := df.Shape()
	ix, _ := NewIndex("", df.Columns())
	out := &DataFrame{index: ix}
	if ncols == 0 {
		out.index = NewRangeIndex(0)
		return out
	}
	all := stackAll(df.cols)
	for i := range nrows {
		idx := make([]int, ncols)
		for j := range idx {
			idx[j] = j*nrows + i
		}
		out.names = append(out.names, df.index.labelString(i))
		out.cols = append(out.cols, all.take(idx))
	}
	return out
}

// Stack returns the frame as a series with one value per (row, column)
// pair, the column name becoming the last index level; missing values are
// dropped (pandas' stack).
func (df *DataFrame) Stack() (*Series, error) {
	nrows, ncols := df.Shape()
	if ncols == 0 {
		return nil, fmt.Errorf("%w: nothing to stack", ErrArgument)
	}
	all := stackAll(df.cols)
	var rows, colIdx, vals []int
	for i := range nrows {
		for j := range ncols {
			if k := j*nrows + i; !all.isNA(k) {
				rows, colIdx, vals = append(rows, i), append(colIdx, j), append(vals, k)
			}
		}
	}
	levels := make([]column, 0, df.index.NLevels()+1)
	for k := range df.index.NLevels() {
		levels = append(levels, df.index.level(k).take(rows))
	}
	levels = append(levels, newStrCol(df.names, nil).take(colIdx))
	names := append(slices.Clone(df.index.names), "")
	return &Series{index: labelIndex(names, levels), c: all.take(vals)}, nil
}

// Unstack pivots the last index level of a multi-level series into columns,
// one per distinct label (sorted), rows being the distinct remaining labels
// (sorted); a missing combination is a missing value (pandas' unstack).
func (s *Series) Unstack() (*DataFrame, error) {
	nl := s.index.NLevels()
	if nl < 2 {
		return nil, fmt.Errorf("%w: Unstack needs a multi-level index", ErrArgument)
	}
	levels := indexColumns(s.index)
	outer, inner := levels[:nl-1], levels[nl-1:]
	rc, rn, rrep := sortedCodes(outer)
	cc, cn, crep := sortedCodes(inner)
	cell := make([]int, rn*cn)
	for i := range cell {
		cell[i] = -1
	}
	for i := range s.Len() {
		if rc[i] < 0 || cc[i] < 0 {
			continue
		}
		k := rc[i]*cn + cc[i]
		if cell[k] >= 0 {
			return nil, fmt.Errorf("%w: duplicate entry %v", ErrArgument, s.index.Label(i))
		}
		cell[k] = i
	}
	rowLevels := make([]column, nl-1)
	for k := range rowLevels {
		rowLevels[k] = outer[k].take(rrep)
	}
	out := &DataFrame{index: labelIndex(s.index.names[:nl-1], rowLevels)}
	for j := range cn {
		idx := make([]int, rn)
		for r := range rn {
			idx[r] = cell[r*cn+j]
		}
		out.names = append(out.names, cellString(inner[0], crep[j]))
		out.cols = append(out.cols, s.c.take(idx))
	}
	return out, nil
}

// sortedCodes codes rows by key tuple with codes in sorted key order; it
// returns the codes, their count, and a representative row of each.
func sortedCodes(keys []column) ([]int, int, []int) {
	if len(keys) == 1 {
		if codes, n, sorted, ok := sortedDense(keys[0]); ok {
			return codes, n, sorted
		}
	}
	codes, n := groupCodes(keys)
	reps := make([]int, n)
	for i := range reps {
		reps[i] = -1
	}
	for i, c := range codes {
		if c >= 0 && reps[c] < 0 {
			reps[c] = i
		}
	}
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	slices.SortStableFunc(order, func(a, b int) int { return compareRows(keys, reps[a], reps[b]) })
	rank := make([]int, n)
	sorted := make([]int, n)
	for r, c := range order {
		rank[c], sorted[r] = r, reps[c]
	}
	for i, c := range codes {
		if c >= 0 {
			codes[i] = rank[c]
		}
	}
	return codes, n, sorted
}

// Unstack pivots the last index level into columns. With one value column
// the new columns are named by the labels; with several, column_label.
func (df *DataFrame) Unstack() (*DataFrame, error) {
	var out *DataFrame
	for k, s := range df.Series() {
		u, err := s.Unstack()
		if err != nil {
			return nil, err
		}
		if len(df.cols) > 1 {
			for j := range u.names {
				u.names[j] = df.names[k] + "_" + u.names[j]
			}
		}
		if out == nil {
			out = u
			continue
		}
		// Every part has the same rows: the frame's sorted outer keys.
		out, _ = ConcatWith(ConcatOptions{Axis: 1}, out, u)
	}
	if out == nil {
		return nil, fmt.Errorf("%w: nothing to unstack", ErrArgument)
	}
	return out, nil
}

// Pivot reshapes long data to wide: rows from the index column, columns from
// the columns column, cells from the values column (pandas' pivot). A
// repeated (index, columns) pair is an error; use PivotTable to aggregate.
func (df *DataFrame) Pivot(index, columns, values string) (*DataFrame, error) {
	s, err := df.SetIndex(index, columns)
	if err != nil {
		return nil, err
	}
	v, err := s.Column(values)
	if err != nil {
		return nil, err
	}
	return v.Unstack()
}

// PivotTable is Pivot with repeated pairs aggregated by aggfunc ("mean",
// pandas' default, "sum", "count", ...).
func (df *DataFrame) PivotTable(index, columns, values, aggfunc string) (*DataFrame, error) {
	if aggfunc == "" {
		aggfunc = "mean"
	}
	g, err := df.GroupBy(index, columns)
	if err != nil {
		return nil, err
	}
	a, err := g.Agg(Agg(values, aggfunc))
	if err != nil {
		return nil, err
	}
	s, _ := a.Column(values)
	return s.Unstack()
}

// Melt reshapes wide data to long (pandas' melt): the id columns repeat, and
// each value column becomes rows of ("variable", "value") pairs. With no
// value columns named, every non-id column is one.
func (df *DataFrame) Melt(idVars, valueVars []string) (*DataFrame, error) {
	if len(valueVars) == 0 {
		for _, n := range df.names {
			if !slices.Contains(idVars, n) {
				valueVars = append(valueVars, n)
			}
		}
	}
	vals, err := keyColumns(df, valueVars)
	if err != nil {
		return nil, err
	}
	ids, err := keyColumns(df, idVars)
	if err != nil {
		return nil, err
	}
	n := df.Len()
	rep := make([]int, 0, n*len(vals))
	varIdx := make([]int, 0, n*len(vals))
	for j := range vals {
		for i := range n {
			rep = append(rep, i)
			varIdx = append(varIdx, j)
		}
	}
	out := &DataFrame{index: NewRangeIndex(len(rep))}
	for k, c := range ids {
		out.names = append(out.names, idVars[k])
		out.cols = append(out.cols, c.take(rep))
	}
	out.names = append(out.names, "variable", "value")
	out.cols = append(out.cols, newStrCol(valueVars, nil).take(varIdx))
	if len(vals) == 0 {
		out.cols = append(out.cols, makeCol[float64](Float64, 0))
	} else {
		out.cols = append(out.cols, stackAll(vals))
	}
	return out, nil
}

// sortedDense is sortedCodes for one int64 key whose range of values is no
// wider than factorizeDense takes: a table over the range marks the values
// present, and walking it in order ranks them, with no hashing and no sort.
func sortedDense(key column) ([]int, int, []int, bool) {
	p, ok := key.(*col[int64])
	if !ok || p.dt != Int64 || len(p.data) == 0 {
		return nil, 0, nil, false
	}
	lo, hi := p.data[0], p.data[0]
	for _, v := range p.data {
		lo, hi = min(lo, v), max(hi, v)
	}
	if hi-lo < 0 || uint64(hi-lo) > uint64(4*len(p.data)+1024) {
		return nil, 0, nil, false
	}
	table := make([]int32, hi-lo+1)
	for i, v := range p.data {
		if p.valid.get(i) {
			table[v-lo] = 1
		}
	}
	n := 0
	for j, t := range table {
		if t == 0 {
			table[j] = -1
			continue
		}
		table[j] = int32(n)
		n++
	}
	codes := make([]int, len(p.data))
	sorted := make([]int, n)
	for i := range sorted {
		sorted[i] = -1
	}
	for i, v := range p.data {
		if !p.valid.get(i) {
			codes[i] = -1
			continue
		}
		c := int(table[v-lo])
		codes[i] = c
		if sorted[c] < 0 {
			sorted[c] = i
		}
	}
	return codes, n, sorted, true
}
