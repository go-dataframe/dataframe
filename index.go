package dataframe

import (
	"fmt"
	"strings"
	"sync"

	"github.com/go-ndarray/ndarray"
)

// Index holds the row labels of a Series or DataFrame. It is a range
// (0, 1, 2, ... as a frame gets by default; nothing is stored), one column of
// labels of any dtype (numbers, strings, instants: a datetime index), or
// several such columns, one per level, for a multi-level index such as a
// group-by over several keys produces. An Index is immutable.
type Index struct {
	n           int
	start, step int      // a range index: label i is start + i*step
	names       []string // one per level
	levels      []column // nil for a range index

	once sync.Once
	find func(any) []int
}

// NewRangeIndex returns the index 0, 1, ..., n-1.
func NewRangeIndex(n int) *Index { return &Index{n: n, step: 1, names: []string{""}} }

// NewIndex returns an index of the given labels: any data NewSeries accepts.
func NewIndex(name string, data any) (*Index, error) {
	c, err := fromAny(data)
	if err != nil {
		return nil, err
	}
	return labelIndex([]string{name}, []column{c}), nil
}

func labelIndex(names []string, levels []column) *Index {
	return &Index{n: levels[0].len(), names: names, levels: levels}
}

// Len returns the number of labels.
func (ix *Index) Len() int { return ix.n }

// IsRange reports a default range index.
func (ix *Index) IsRange() bool { return ix.levels == nil }

// NLevels returns the number of levels: 1 unless the index is multi-level.
func (ix *Index) NLevels() int { return len(ix.names) }

// Name returns the name of the first level ("" when unnamed).
func (ix *Index) Name() string { return ix.names[0] }

// Names returns the level names.
func (ix *Index) Names() []string { return append([]string(nil), ix.names...) }

// DType returns the dtype of the first level.
func (ix *Index) DType() DType { return ix.level(0).dtype() }

// Rename returns the index with new level names.
func (ix *Index) Rename(names ...string) (*Index, error) {
	if len(names) != len(ix.names) {
		return nil, fmt.Errorf("%w: %d names for %d levels", ErrLength, len(names), len(ix.names))
	}
	return &Index{n: ix.n, start: ix.start, step: ix.step, levels: ix.levels,
		names: append([]string(nil), names...)}, nil
}

// level returns level k as a column, materialising a range.
func (ix *Index) level(k int) column {
	if ix.levels != nil {
		return ix.levels[k]
	}
	c := makeCol[int64](Int64, ix.n)
	for i := range c.data {
		c.data[i] = int64(ix.start + i*ix.step)
	}
	return c
}

// Level returns level k as a Series named after the level.
func (ix *Index) Level(k int) (*Series, error) {
	if k < 0 || k >= len(ix.names) {
		return nil, fmt.Errorf("%w: level %d of %d", ErrIndex, k, len(ix.names))
	}
	return &Series{name: ix.names[k], index: NewRangeIndex(ix.n), c: ix.level(k)}, nil
}

// Label returns the label at position i: a Go value for a one-level index
// (nil when missing), a []any tuple for a multi-level one.
func (ix *Index) Label(i int) any {
	if ix.levels == nil {
		return int64(ix.start + i*ix.step)
	}
	if len(ix.levels) == 1 {
		return ix.levels[0].value(i)
	}
	t := make([]any, len(ix.levels))
	for k, c := range ix.levels {
		t[k] = c.value(i)
	}
	return t
}

// Labels returns every label, as Label does.
func (ix *Index) Labels() []any {
	out := make([]any, ix.n)
	for i := range out {
		out[i] = ix.Label(i)
	}
	return out
}

// Values returns the labels of a one-level numeric (or datetime, as int64
// nanoseconds) index as an array.
func (ix *Index) Values() (*ndarray.Array, error) {
	c := ix.level(0)
	if a, ok := c.ndarray(); ok && len(ix.names) == 1 {
		return a, nil
	}
	return nil, fmt.Errorf("%w: a %v index has no array form", ErrDType, c.dtype())
}

// Get returns the positions of a label (a []any tuple, or a prefix of one,
// for a multi-level index). A missing label is an ErrLabel.
func (ix *Index) Get(label any) ([]int, error) {
	if ix.levels == nil {
		i, ok := toInt(label)
		k := (int(i) - ix.start) / ix.step
		if ok && k >= 0 && k < ix.n && ix.start+k*ix.step == int(i) {
			return []int{k}, nil
		}
		return nil, fmt.Errorf("%w: %v", ErrLabel, label)
	}
	if len(ix.levels) == 1 {
		ix.once.Do(func() { ix.find = ix.levels[0].lookup() })
		if p := ix.find(label); len(p) > 0 {
			return p, nil
		}
		return nil, fmt.Errorf("%w: %v", ErrLabel, label)
	}
	t, ok := label.([]any)
	if !ok {
		t = []any{label}
	}
	if len(t) > len(ix.levels) {
		return nil, fmt.Errorf("%w: a %d-tuple for %d levels", ErrLabel, len(t), len(ix.levels))
	}
	var out []int
	for i := range ix.n {
		if ix.matches(i, t) {
			out = append(out, i)
		}
	}
	if out == nil {
		return nil, fmt.Errorf("%w: %v", ErrLabel, label)
	}
	return out, nil
}

// matches reports whether row i of a multi-level index starts with tuple t.
func (ix *Index) matches(i int, t []any) bool {
	for k, want := range t {
		got := ix.levels[k].value(i)
		if got == nil || !sameValue(ix.levels[k], got, want) {
			return false
		}
	}
	return true
}

// sameValue reports whether a value of column c equals want, converted to
// c's dtype.
func sameValue(c column, got, want any) bool {
	w, ok := c.convert(want)
	return ok && w.value(0) == got
}

// take returns the index of the rows at idx (-1: missing label).
func (ix *Index) take(idx []int) *Index {
	if ix.levels == nil {
		if r, ok := asRange(idx); ok {
			return &Index{n: len(idx), start: ix.start + r.start*ix.step, step: ix.step * r.step, names: ix.names}
		}
	}
	if ix.levels == nil {
		c := makeCol[int64](Int64, len(idx))
		var valid bitmap
		for k, i := range idx {
			if i < 0 {
				if valid == nil {
					valid = allValid(len(idx))
				}
				valid.clear(k)
				continue
			}
			c.data[k] = int64(ix.start + i*ix.step)
		}
		c.valid = valid
		return labelIndex(ix.names, []column{c})
	}
	levels := make([]column, len(ix.names))
	for k := range levels {
		levels[k] = ix.levels[k].take(idx)
	}
	return labelIndex(ix.names, levels)
}

// slice returns the index of rows [i, j).
func (ix *Index) slice(i, j int) *Index {
	if ix.levels == nil {
		return &Index{n: j - i, start: ix.start + i*ix.step, step: ix.step, names: ix.names}
	}
	levels := make([]column, len(ix.levels))
	for k, c := range ix.levels {
		levels[k] = c.slice(i, j)
	}
	return labelIndex(ix.names, levels)
}

type rng struct{ start, step int }

// asRange reports whether positions form an arithmetic progression with a
// positive step (or are a single position), and which.
func asRange(idx []int) (rng, bool) {
	if len(idx) == 0 {
		return rng{0, 1}, true
	}
	if idx[0] < 0 {
		return rng{}, false
	}
	step := 1
	if len(idx) > 1 {
		step = idx[1] - idx[0]
	}
	if step <= 0 {
		return rng{}, false
	}
	for k := 1; k < len(idx); k++ {
		if idx[k]-idx[k-1] != step {
			return rng{}, false
		}
	}
	return rng{idx[0], step}, true
}

// equal reports whether two indexes hold the same labels in the same order.
func (ix *Index) equal(o *Index) bool {
	if ix == o {
		return true
	}
	if ix.n != o.n || len(ix.names) != len(o.names) {
		return false
	}
	if ix.levels == nil && o.levels == nil {
		return ix.n == 0 || (ix.start == o.start && ix.step == o.step)
	}
	for k := range ix.names {
		a, b := ix.level(k), o.level(k)
		for i := range ix.n {
			na, nb := a.isNA(i), b.isNA(i)
			if na != nb || (!na && !a.equal(i, b, i)) {
				return false
			}
		}
	}
	return true
}

// labelString formats label i for display; a multi-level label is its
// levels joined by commas, in parentheses.
func (ix *Index) labelString(i int) string {
	if len(ix.names) == 1 {
		return cellString(ix.level(0), i)
	}
	parts := make([]string, len(ix.levels))
	for k, c := range ix.levels {
		parts[k] = cellString(c, i)
	}
	return "(" + strings.Join(parts, ", ") + ")"
}

// cellString formats element i of a column, a missing one as pandas shows
// it: NaN for numbers, NaT for instants, <NA> otherwise.
func cellString(c column, i int) string {
	if c.isNA(i) {
		return naString(c.dtype())
	}
	return c.str(i)
}

func naString(dt DType) string {
	switch {
	case dt == Datetime || dt == Period:
		return "NaT"
	case dt.isFloat() || dt == Complex64 || dt == Complex128 || dt == String || dt == Category:
		return "NaN"
	}
	return "<NA>"
}

// String formats the index the way pandas prints one.
func (ix *Index) String() string {
	var b strings.Builder
	kind := "Index"
	switch {
	case ix.levels == nil:
		return fmt.Sprintf("RangeIndex(start=%d, stop=%d, step=%d)", ix.start, ix.start+ix.n*ix.step, ix.step)
	case len(ix.levels) > 1:
		kind = "MultiIndex"
	case ix.levels[0].dtype() == Datetime:
		kind = "DatetimeIndex"
	}
	b.WriteString(kind + "([")
	for i := 0; i < ix.n; i++ {
		if i > 0 {
			b.WriteString(", ")
		}
		if i == 10 && ix.n > 20 {
			b.WriteString("..., ")
			i = ix.n - 10
		}
		s := ix.labelString(i)
		if len(ix.levels) == 1 && (ix.levels[0].dtype() == String || ix.levels[0].dtype() == Datetime) && !ix.levels[0].isNA(i) {
			s = "'" + s + "'"
		}
		b.WriteString(s)
	}
	fmt.Fprintf(&b, "], dtype='%v'", ix.levels[0].dtype())
	if ix.names[0] != "" {
		fmt.Fprintf(&b, ", name='%s'", ix.names[0])
	}
	b.WriteString(")")
	return b.String()
}
