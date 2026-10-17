package dataframe

import (
	"fmt"
	"runtime"
	"slices"
	"sync"
	"unsafe"
)

// Selector picks rows (or, for Iloc, columns): all of them, a range or list
// of positions, a list or inclusive range of labels, or a boolean mask.
// Build one with All, Slice, Pos, Labels, LabelRange or Mask.
type Selector struct {
	kind        selKind
	start, stop int
	open        bool // Slice to the end
	pos         []int
	labels      []any
	from, to    any
	mask        any
}

type selKind uint8

const (
	selAll selKind = iota
	selSlice
	selPos
	selLabels
	selLabelRange
	selMask
)

// All selects everything (pandas' ":").
func All() Selector { return Selector{kind: selAll} }

// Slice selects positions [start, stop), as a Go slice expression would,
// except that negative values count from the end and out-of-range bounds are
// clipped, as in pandas' iloc[start:stop].
func Slice(start, stop int) Selector { return Selector{kind: selSlice, start: start, stop: stop} }

// SliceFrom selects positions from start to the end (iloc[start:]).
func SliceFrom(start int) Selector { return Selector{kind: selSlice, start: start, open: true} }

// Pos selects the given positions, in that order; negative ones count from
// the end.
func Pos(i ...int) Selector { return Selector{kind: selPos, pos: i} }

// Labels selects the rows with the given labels, in that order (a []any
// tuple, or a prefix of one, for a multi-level index).
func Labels(l ...any) Selector { return Selector{kind: selLabels, labels: l} }

// LabelRange selects from label from to label to, both included, as pandas'
// loc[from:to]. A nil bound is open. A label that is absent is placed by
// binary search, which needs a sorted index.
func LabelRange(from, to any) Selector { return Selector{kind: selLabelRange, from: from, to: to} }

// Mask selects the rows where a mask is true: a []bool of the right length,
// or a bool *Series, which is aligned on the index by label when its index
// differs (it must then hold every label; missing values count as false).
func Mask(m any) Selector { return Selector{kind: selMask, mask: m} }

// resolve returns the positions a selector picks out of n, with an index for
// label selectors (nil when selecting columns). It returns ok=false with
// i, j set when the selection is the contiguous range [i, j).
func (sel Selector) resolve(n int, ix *Index) (idx []int, i, j int, err error) {
	switch sel.kind {
	case selAll:
		return nil, 0, n, nil
	case selSlice:
		i, j = clip(sel.start, n), clip(sel.stop, n)
		if sel.open {
			j = n
		}
		return nil, i, max(i, j), nil
	case selPos:
		idx = make([]int, len(sel.pos))
		for k, p := range sel.pos {
			if idx[k], err = position(p, n); err != nil {
				return nil, 0, 0, err
			}
		}
		return idx, 0, 0, nil
	case selLabels:
		if ix == nil {
			return nil, 0, 0, fmt.Errorf("%w: label selection of columns; name them", ErrArgument)
		}
		for _, l := range sel.labels {
			p, err := ix.Get(l)
			if err != nil {
				return nil, 0, 0, err
			}
			idx = append(idx, p...)
		}
		return idx, 0, 0, nil
	case selLabelRange:
		if ix == nil {
			return nil, 0, 0, fmt.Errorf("%w: label selection of columns; name them", ErrArgument)
		}
		i, j, err = ix.labelRange(sel.from, sel.to)
		return nil, i, j, err
	}
	m, err := maskOf(sel.mask, n, ix)
	if err != nil {
		return nil, 0, 0, err
	}
	return maskPositions(m), 0, 0, nil
}

// clip resolves a slice bound as Python does.
func clip(i, n int) int {
	if i < 0 {
		i += n
	}
	return min(max(i, 0), n)
}

// maskOf returns a mask as a []bool of length n.
func maskOf(m any, n int, ix *Index) ([]bool, error) {
	switch m := m.(type) {
	case []bool:
		if len(m) != n {
			return nil, fmt.Errorf("%w: mask of %d for %d rows", ErrLength, len(m), n)
		}
		return m, nil
	case *Series:
		b, err := m.bools()
		if err != nil {
			return nil, err
		}
		if ix == nil || m.index.equal(ix) {
			return maskOf(b, n, nil)
		}
		idx, err := alignTo(m.index, ix)
		if err != nil {
			return nil, err
		}
		out := make([]bool, n)
		for k, p := range idx {
			if p < 0 {
				return nil, fmt.Errorf("%w: the mask has no label %v", ErrLabel, ix.Label(k))
			}
			out[k] = b[p]
		}
		return out, nil
	}
	return nil, fmt.Errorf("%w: a mask is a []bool or a bool *Series, not %T", ErrArgument, m)
}

// labelRange returns the positions [i, j) from label from through label to.
func (ix *Index) labelRange(from, to any) (int, int, error) {
	i, j := 0, ix.n
	if from != nil {
		p, err := ix.Get(from)
		switch {
		case err == nil:
			i = p[0]
		default:
			if i, err = ix.searchSorted(from, false); err != nil {
				return 0, 0, err
			}
		}
	}
	if to != nil {
		p, err := ix.Get(to)
		switch {
		case err == nil:
			j = p[len(p)-1] + 1
		default:
			if j, err = ix.searchSorted(to, true); err != nil {
				return 0, 0, err
			}
		}
	}
	return i, max(i, j), nil
}

// searchSorted places a label in a sorted one-level index: the first
// position whose label is >= v (> v when right is set).
func (ix *Index) searchSorted(v any, right bool) (int, error) {
	c := ix.level(0)
	if !ix.monotonic() || ix.NLevels() > 1 {
		return 0, fmt.Errorf("%w: %v (and the index is not sorted)", ErrLabel, v)
	}
	one, ok := c.convert(v)
	if !ok {
		return 0, fmt.Errorf("%w: %v", ErrLabel, v)
	}
	both := c.appendCol(one)
	n := c.len()
	lo, hi := 0, n
	for lo < hi {
		mid := (lo + hi) / 2
		r := both.compare(mid, n)
		if r < 0 || (right && r == 0) {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	return lo, nil
}

// monotonic reports an index sorted ascending, with no missing labels.
func (ix *Index) monotonic() bool {
	if ix.levels == nil {
		return true
	}
	c := ix.levels[0]
	for i := range ix.n {
		if c.isNA(i) || (i > 0 && c.compare(i-1, i) > 0) {
			return false
		}
	}
	return true
}

// rows returns the rows a selector picks.
func (df *DataFrame) rows(sel Selector) (*DataFrame, error) {
	idx, i, j, err := sel.resolve(df.Len(), df.index)
	switch {
	case err != nil:
		return nil, err
	case idx == nil:
		return df.slice(i, j), nil
	}
	return df.takeOwned(idx), nil
}

// Iloc selects rows by position, and optionally columns by position:
//
//	df.Iloc(dataframe.Slice(0, 5))                        // df.iloc[0:5]
//	df.Iloc(dataframe.Pos(0, 2), dataframe.Slice(1, 3))   // df.iloc[[0, 2], 1:3]
//	df.Iloc(dataframe.Mask(m))                            // df.iloc[m]
func (df *DataFrame) Iloc(rows Selector, cols ...Selector) (*DataFrame, error) {
	out := df
	if len(cols) > 0 {
		idx, i, j, err := cols[0].resolve(len(df.cols), nil)
		if err != nil {
			return nil, err
		}
		if idx == nil {
			for k := i; k < j; k++ {
				idx = append(idx, k)
			}
		}
		out = &DataFrame{index: df.index}
		for _, k := range idx {
			out.names = append(out.names, df.names[k])
			out.cols = append(out.cols, df.cols[k])
		}
	}
	if rows.kind == selLabels || rows.kind == selLabelRange {
		return nil, fmt.Errorf("%w: Iloc takes positions; use Loc for labels", ErrArgument)
	}
	return out.rows(rows)
}

// Loc selects rows by label (or mask), and optionally columns by name:
//
//	df.Loc(dataframe.Labels("a", "c"))                    // df.loc[["a", "c"]]
//	df.Loc(dataframe.LabelRange("2020-01", "2020-03"))    // df.loc["2020-01":"2020-03"]
//	df.Loc(dataframe.Mask(m), "x", "y")                   // df.loc[m, ["x", "y"]]
func (df *DataFrame) Loc(rows Selector, cols ...string) (*DataFrame, error) {
	out := df
	if len(cols) > 0 {
		var err error
		if out, err = df.Select(cols...); err != nil {
			return nil, err
		}
	}
	if rows.kind == selPos || rows.kind == selSlice {
		return nil, fmt.Errorf("%w: Loc takes labels; use Iloc for positions", ErrArgument)
	}
	return out.rows(rows)
}

// Filter returns the rows where mask is true (pandas' df[mask]). The mask is
// a bool series (aligned by label) or a []bool.
func (df *DataFrame) Filter(mask any) (*DataFrame, error) { return df.rows(Mask(mask)) }

// Iloc selects values by position, as DataFrame.Iloc selects rows.
func (s *Series) Iloc(sel Selector) (*Series, error) {
	if sel.kind == selLabels || sel.kind == selLabelRange {
		return nil, fmt.Errorf("%w: Iloc takes positions; use Loc for labels", ErrArgument)
	}
	return s.sel(sel)
}

// Loc selects values by label or mask, as DataFrame.Loc selects rows.
func (s *Series) Loc(sel Selector) (*Series, error) {
	if sel.kind == selPos || sel.kind == selSlice {
		return nil, fmt.Errorf("%w: Loc takes labels; use Iloc for positions", ErrArgument)
	}
	return s.sel(sel)
}

// Filter returns the values where mask is true.
func (s *Series) Filter(mask any) (*Series, error) { return s.sel(Mask(mask)) }

func (s *Series) sel(sel Selector) (*Series, error) {
	idx, i, j, err := sel.resolve(s.Len(), s.index)
	switch {
	case err != nil:
		return nil, err
	case idx == nil:
		return s.slice(i, j), nil
	}
	return s.take(idx), nil
}

// At returns the value at a row label and column name (pandas' df.at).
func (df *DataFrame) At(label any, column string) (any, error) {
	s, err := df.Column(column)
	if err != nil {
		return nil, err
	}
	return s.Get(label)
}

// IAt returns the value at row position i and column position j (pandas'
// df.iat); negative positions count from the end.
func (df *DataFrame) IAt(i, j int) (any, error) {
	s, err := df.ColumnAt(j)
	if err != nil {
		return nil, err
	}
	return s.At(i)
}

// SetIndex returns the frame indexed by the named columns (several make a
// multi-level index), which leave the columns, as pandas' set_index does by
// default.
func (df *DataFrame) SetIndex(columns ...string) (*DataFrame, error) {
	if len(columns) == 0 {
		return nil, fmt.Errorf("%w: SetIndex needs a column", ErrArgument)
	}
	levels := make([]column, len(columns))
	for k, name := range columns {
		i, err := df.pos(name)
		if err != nil {
			return nil, err
		}
		levels[k] = df.cols[i]
	}
	out, _ := df.Drop(columns...)
	out.index = labelIndex(slices.Clone(columns), levels)
	return out, nil
}

// WithIndex returns the frame with the given index, of the same length.
func (df *DataFrame) WithIndex(ix *Index) (*DataFrame, error) {
	if ix.Len() != df.Len() {
		return nil, fmt.Errorf("%w: index of %d for %d rows", ErrLength, ix.Len(), df.Len())
	}
	return &DataFrame{index: ix, names: df.names, cols: df.cols}, nil
}

// ResetIndex returns the frame with a range index. Unless drop is set, the
// old index levels become the first columns, named after the levels ("index"
// for an unnamed one-level index, "level_0", "level_1", ... for unnamed
// levels of a multi-level one).
func (df *DataFrame) ResetIndex(drop bool) *DataFrame {
	out := &DataFrame{index: NewRangeIndex(df.Len())}
	if !drop {
		for k, name := range df.index.names {
			if name == "" {
				name = "index"
				if df.index.NLevels() > 1 {
					name = fmt.Sprintf("level_%d", k)
				}
			}
			out.names = append(out.names, name)
			out.cols = append(out.cols, df.index.level(k))
		}
	}
	out.names = append(out.names, df.names...)
	out.cols = append(out.cols, df.cols...)
	return out
}

// Reindex returns the frame with rows for the given labels (an *Index or
// anything NewIndex accepts), in that order; a label the frame lacks gives a
// row of missing values.
func (df *DataFrame) Reindex(labels any) (*DataFrame, error) {
	ix, ok := labels.(*Index)
	if !ok {
		var err error
		if ix, err = NewIndex(df.index.Name(), labels); err != nil {
			return nil, err
		}
	}
	idx, err := alignTo(df.index, ix)
	if err != nil {
		return nil, err
	}
	cols := make([]column, len(df.cols))
	for k, c := range df.cols {
		cols[k] = c.take(idx)
	}
	return &DataFrame{index: ix, names: df.names, cols: cols}, nil
}

// ReindexColumns returns the frame with the given columns in that order; a
// name the frame lacks gives an all-missing float64 column.
func (df *DataFrame) ReindexColumns(names ...string) *DataFrame {
	out := &DataFrame{index: df.index, names: slices.Clone(names)}
	for _, n := range names {
		if i, err := df.pos(n); err == nil {
			out.cols = append(out.cols, df.cols[i])
		} else {
			out.cols = append(out.cols, makeCol[float64](Float64, df.Len()).nulls(df.Len()))
		}
	}
	return out
}

// Reindex returns the series with values for the given labels, as
// DataFrame.Reindex.
func (s *Series) Reindex(labels any) (*Series, error) {
	df, err := s.ToFrame().Reindex(labels)
	if err != nil {
		return nil, err
	}
	return &Series{name: s.name, index: df.index, c: df.cols[0]}, nil
}

// maskPositions returns the positions where m is true, counting and filling
// in parallel chunks for a long mask.
func maskPositions(m []bool) []int {
	if len(m) < parallelMin || runtime.GOMAXPROCS(0) < 2 {
		// One pass to count, one to fill, both without branches: a bool
		// is a 0 or 1 byte.
		total := 0
		for _, b := range m {
			total += int(*(*uint8)(unsafe.Pointer(&b)))
		}
		idx := make([]int, total+1)
		j := 0
		for i, b := range m {
			idx[j] = i
			j += int(*(*uint8)(unsafe.Pointer(&b)))
		}
		return idx[:total:total]
	}
	var mu sync.Mutex
	counts := map[int]int{}
	parallel(len(m), func(lo, hi int) {
		c := 0
		for _, b := range m[lo:hi] {
			if b {
				c++
			}
		}
		mu.Lock()
		counts[lo] = c
		mu.Unlock()
	})
	los := make([]int, 0, len(counts))
	for lo := range counts {
		los = append(los, lo)
	}
	slices.Sort(los)
	start := map[int]int{}
	total := 0
	for _, lo := range los {
		start[lo] = total
		total += counts[lo]
	}
	idx := make([]int, total)
	parallel(len(m), func(lo, hi int) {
		mu.Lock()
		j := start[lo]
		mu.Unlock()
		for k, b := range m[lo:hi] {
			if b {
				idx[j] = lo + k
				j++
			}
		}
	})
	return idx
}
