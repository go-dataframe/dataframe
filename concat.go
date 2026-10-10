package dataframe

import (
	"fmt"
	"slices"
)

// ConcatOptions tunes Concat.
type ConcatOptions struct {
	// Axis 0 stacks frames vertically (rows after rows); 1 places them side
	// by side, aligned on the index.
	Axis int
	// IgnoreIndex gives the result a range index (axis 0).
	IgnoreIndex bool
	// Inner keeps only the columns (axis 0) or labels (axis 1) every frame
	// has, instead of their union.
	Inner bool
}

// Concat stacks frames vertically: the columns are the union of the frames'
// columns in order of appearance (a frame lacking one contributes missing
// values), and the indexes are concatenated (pandas' concat).
func Concat(frames ...*DataFrame) (*DataFrame, error) {
	return ConcatWith(ConcatOptions{}, frames...)
}

// ConcatWith is Concat with options.
func ConcatWith(opts ConcatOptions, frames ...*DataFrame) (*DataFrame, error) {
	if len(frames) == 0 {
		return nil, fmt.Errorf("%w: nothing to concatenate", ErrArgument)
	}
	if opts.Axis == 1 {
		return concatColumns(opts, frames)
	}
	var names []string
	for _, f := range frames {
		for _, n := range f.names {
			if !slices.Contains(names, n) {
				names = append(names, n)
			}
		}
	}
	if opts.Inner {
		names = slices.DeleteFunc(names, func(n string) bool {
			for _, f := range frames {
				if !slices.Contains(f.names, n) {
					return true
				}
			}
			return false
		})
	}
	out := &DataFrame{names: names}
	for _, n := range names {
		var parts []column
		var proto column
		for _, f := range frames {
			if i, err := f.pos(n); err == nil {
				parts = append(parts, f.cols[i])
				if proto == nil {
					proto = f.cols[i]
				}
			} else {
				parts = append(parts, nil)
			}
		}
		for k, f := range frames {
			if parts[k] == nil {
				parts[k] = proto.nulls(f.Len())
			}
		}
		c, err := appendAll(parts)
		if err != nil {
			return nil, fmt.Errorf("column %q: %w", n, err)
		}
		out.cols = append(out.cols, c)
	}
	total := 0
	for _, f := range frames {
		total += f.Len()
	}
	if opts.IgnoreIndex {
		out.index = NewRangeIndex(total)
		return out, nil
	}
	ix, err := concatIndexes(frames)
	if err != nil {
		return nil, err
	}
	out.index = ix
	return out, nil
}

// appendAll concatenates columns, converting them to a common dtype.
func appendAll(parts []column) (column, error) {
	dt := parts[0].dtype()
	for _, p := range parts[1:] {
		var err error
		if dt, err = commonType(dt, p.dtype()); err != nil {
			return nil, err
		}
	}
	out, _ := cast(parts[0], dt)
	for _, p := range parts[1:] {
		q, _ := cast(p, dt)
		out = out.appendCol(q)
	}
	return out, nil
}

// concatIndexes concatenates the frames' indexes, level by level.
func concatIndexes(frames []*DataFrame) (*Index, error) {
	first := frames[0].index
	levels := make([]column, first.NLevels())
	for k := range levels {
		parts := make([]column, len(frames))
		for i, f := range frames {
			if f.index.NLevels() != len(levels) {
				return nil, fmt.Errorf("%w: indexes of %d and %d levels", ErrArgument, len(levels), f.index.NLevels())
			}
			parts[i] = f.index.level(k)
		}
		c, err := appendAll(parts)
		if err != nil {
			return nil, err
		}
		levels[k] = c
	}
	return labelIndex(first.names, levels), nil
}

// concatColumns places frames side by side, aligned on the union (or the
// intersection) of their labels, in order of appearance.
func concatColumns(opts ConcatOptions, frames []*DataFrame) (*DataFrame, error) {
	ix := frames[0].index
	same := true
	for _, f := range frames[1:] {
		same = same && f.index.equal(ix)
	}
	if !same {
		var err error
		if ix, err = combineIndexes(frames, opts.Inner); err != nil {
			return nil, err
		}
	}
	out := &DataFrame{index: ix}
	for _, f := range frames {
		g := f
		if !same {
			var err error
			if g, err = f.Reindex(ix); err != nil {
				return nil, err
			}
		}
		out.names = append(out.names, g.names...)
		out.cols = append(out.cols, g.cols...)
	}
	return out, nil
}

// combineIndexes returns the union (in order of appearance) or the
// intersection (in the first index's order) of the frames' labels.
func combineIndexes(frames []*DataFrame, inner bool) (*Index, error) {
	all := make([]*DataFrame, len(frames))
	for i, f := range frames {
		all[i] = &DataFrame{index: NewRangeIndex(f.Len()), names: []string{"_"},
			cols: []column{makeCol[bool](Bool, f.Len())}}
		all[i], _ = all[i].WithIndex(f.index)
	}
	stacked, err := concatIndexes(all)
	if err != nil {
		return nil, err
	}
	levels := indexColumns(stacked)
	codes, _, n, _ := jointCodes(levels, nil)
	seen := make([]int, n) // how many frames have each label
	first := make([]int, n)
	for i := range first {
		first[i] = -1
	}
	start := 0
	for _, f := range frames {
		mark := make([]bool, n)
		for i := start; i < start+f.Len(); i++ {
			c := codes[i]
			if c < 0 {
				continue
			}
			if first[c] < 0 {
				first[c] = i
			}
			if !mark[c] {
				mark[c] = true
				seen[c]++
			}
		}
		start += f.Len()
	}
	var keep []int
	for c, i := range first {
		if i >= 0 && (!inner || seen[c] == len(frames)) {
			keep = append(keep, i)
		}
	}
	slices.Sort(keep)
	return stacked.take(keep), nil
}

// ConcatSeries stacks series end to end.
func ConcatSeries(series ...*Series) (*Series, error) {
	frames := make([]*DataFrame, len(series))
	for i, s := range series {
		frames[i] = &DataFrame{index: s.index, names: []string{series[0].name}, cols: []column{s.c}}
	}
	df, err := Concat(frames...)
	if err != nil {
		return nil, err
	}
	return &Series{name: series[0].name, index: df.index, c: df.cols[0]}, nil
}
