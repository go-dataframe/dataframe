package dataframe

import (
	"fmt"
	"math"
	"slices"
)

// commonType returns the dtype two key or operand columns are compared in:
// NumPy's result type for two numbers, string for text and categories, and
// the dtype itself when they agree. Converting a column to it never fails.
func commonType(a, b DType) (DType, error) {
	switch {
	case a == b && a != Category:
		return a, nil
	case a.IsNumeric() && b.IsNumeric():
		return resultType(a, b), nil
	case (a == String || a == Category) && (b == String || b == Category):
		return String, nil
	}
	return 0, fmt.Errorf("%w: cannot match %v with %v", ErrDType, a, b)
}

// jointCodes codes the rows of two tables, each given as its key columns,
// into one code space: equal key tuples get equal codes, a tuple with a
// missing key gets -1. b may be nil to code a single table. It returns the
// codes and the number of distinct tuples.
func jointCodes(a, b []column) ([]int, []int, int, error) {
	na := a[0].len()
	nb := 0
	if b != nil {
		nb = b[0].len()
	}
	ca := make([]int, na)
	cb := make([]int, nb)
	card := 1
	for k := range a {
		ka := a[k]
		var kb column
		if b != nil {
			dt, err := commonType(ka.dtype(), b[k].dtype())
			if err != nil {
				return nil, nil, 0, err
			}
			ka, _ = cast(ka, dt)
			kb, _ = cast(b[k], dt)
		} else {
			kb = ka.slice(0, 0)
		}
		xa, xb, n, ok := factorizeDense(ka, kb)
		if !ok {
			var u column
			xa, xb, u = ka.factorizeWith(kb)
			n = u.len()
		}
		if k == 0 {
			for i, c := range xa {
				ca[i] = int(c)
			}
			for i, c := range xb {
				cb[i] = int(c)
			}
		} else {
			combine(ca, xa, n)
			combine(cb, xb, n)
		}
		card *= n
		if card > math.MaxInt32 && k < len(a)-1 {
			card = compact(ca, cb)
		}
	}
	if len(a) > 1 {
		card = compact(ca, cb)
	}
	return ca, cb, card, nil
}

// combine folds one key's codes into running tuple codes.
func combine(acc []int, codes []int32, n int) {
	for i, c := range codes {
		switch {
		case acc[i] < 0:
		case c < 0:
			acc[i] = -1
		default:
			acc[i] = acc[i]*n + int(c)
		}
	}
}

// compact renumbers tuple codes densely, in order of first appearance over
// a then b, and returns their count.
func compact(a, b []int) int {
	m := make(map[int]int)
	for _, s := range [][]int{a, b} {
		for i, c := range s {
			if c < 0 {
				continue
			}
			k, ok := m[c]
			if !ok {
				k = len(m)
				m[c] = k
			}
			s[i] = k
		}
	}
	return len(m)
}

// indexColumns returns the levels of an index as columns.
func indexColumns(ix *Index) []column {
	out := make([]column, ix.NLevels())
	for k := range out {
		out[k] = ix.level(k)
	}
	return out
}

// alignTo returns, for each label of to, the position of that label in
// from, or -1 when from does not have it. Labels of from must be unique.
func alignTo(from, to *Index) ([]int, error) {
	if from.equal(to) {
		idx := make([]int, from.Len())
		for i := range idx {
			idx[i] = i
		}
		return idx, nil
	}
	cf, ct, n, err := jointCodes(indexColumns(from), indexColumns(to))
	if err != nil {
		return nil, err
	}
	where := make([]int, n)
	for i := range where {
		where[i] = -1
	}
	for i, c := range cf {
		if c < 0 {
			continue
		}
		if where[c] >= 0 {
			return nil, fmt.Errorf("%w: cannot align on an index with duplicate label %v", ErrLabel, from.Label(i))
		}
		where[c] = i
	}
	out := make([]int, len(ct))
	for i, c := range ct {
		out[i] = -1
		if c >= 0 {
			out[i] = where[c]
		}
	}
	return out, nil
}

// unionIndex returns the sorted union of the labels of the series' indexes.
func unionIndex(series []*Series) (*Index, error) {
	levels := indexColumns(series[0].index)
	names := series[0].index.names
	for _, s := range series[1:] {
		other := indexColumns(s.index)
		if len(other) != len(levels) {
			return nil, fmt.Errorf("%w: indexes of %d and %d levels", ErrArgument, len(levels), len(other))
		}
		for k := range levels {
			dt, err := commonType(levels[k].dtype(), other[k].dtype())
			if err != nil {
				return nil, err
			}
			a, _ := cast(levels[k], dt)
			b, _ := cast(other[k], dt)
			levels[k] = a.appendCol(b)
		}
	}
	codes, _, _, _ := jointCodes(levels, nil)
	seen := make(map[int]bool)
	var first []int
	for i, c := range codes {
		if !seen[c] {
			seen[c] = true
			first = append(first, i)
		}
	}
	slices.SortStableFunc(first, func(i, j int) int { return compareRows(levels, i, j) })
	out := make([]column, len(levels))
	for k := range levels {
		out[k] = levels[k].take(first)
	}
	return labelIndex(names, out), nil
}

// compareRows orders rows i and j of key columns lexicographically, missing
// values last.
func compareRows(keys []column, i, j int) int {
	for _, c := range keys {
		ni, nj := c.isNA(i), c.isNA(j)
		switch {
		case ni && nj:
			continue
		case ni:
			return 1
		case nj:
			return -1
		}
		if r := c.compare(i, j); r != 0 {
			return r
		}
	}
	return 0
}

// alignSeries aligns two series on their indexes: unchanged when the indexes
// are equal, otherwise both reindexed to the sorted union of labels.
func alignSeries(a, b *Series) (*Index, column, column, error) {
	if a.index.equal(b.index) {
		return a.index, a.c, b.c, nil
	}
	ix, err := unionIndex([]*Series{a, b})
	if err != nil {
		return nil, nil, nil, err
	}
	ia, err := alignTo(a.index, ix)
	if err != nil {
		return nil, nil, nil, err
	}
	ib, err := alignTo(b.index, ix)
	if err != nil {
		return nil, nil, nil, err
	}
	return ix, a.c.take(ia), b.c.take(ib), nil
}

// factorizeDense codes two int64 columns jointly with a lookup table over
// their range of values, when that range is no wider than a few times their
// length: no hashing at all for the common small-integer key.
func factorizeDense(a, b column) ([]int32, []int32, int, bool) {
	pa, ok1 := a.(*col[int64])
	pb, ok2 := b.(*col[int64])
	if !ok1 || !ok2 || pa.dt != Int64 || pb.dt != Int64 || len(pa.data) == 0 {
		return nil, nil, 0, false
	}
	lo, hi := pa.data[0], pa.data[0]
	for _, d := range [][]int64{pa.data, pb.data} {
		for _, v := range d {
			lo, hi = min(lo, v), max(hi, v)
		}
	}
	limit := uint64(4*(len(pa.data)+len(pb.data)) + 1024)
	if hi-lo < 0 || uint64(hi-lo) > limit {
		return nil, nil, 0, false
	}
	table := make([]int32, hi-lo+1)
	n := int32(0)
	code := func(p *col[int64]) []int32 {
		codes := make([]int32, len(p.data))
		for i, v := range p.data {
			if !p.valid.get(i) {
				codes[i] = -1
				continue
			}
			t := &table[v-lo]
			if *t == 0 {
				n++
				*t = n
			}
			codes[i] = *t - 1
		}
		return codes
	}
	ca := code(pa)
	cb := code(pb)
	return ca, cb, int(n), true
}
