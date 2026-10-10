package dataframe

import (
	"fmt"
	"math"
	"slices"
)

// radixMin is the size below which a comparison sort beats the radix sort.
const radixMin = 512

// sortPerm returns the stable ordering of rows by key columns, the first key
// most significant, each ascending or descending, missing values last for
// every key (pandas' na_position="last"). It is an LSD sort: one stable pass
// per key, least significant first, each pass a radix sort on keys with an
// order-preserving integer form and a merge sort otherwise.
func sortPerm(keys []column, ascending []bool) []int {
	n := keys[0].len()
	perm := make([]int, n)
	for i := range perm {
		perm[i] = i
	}
	for k := len(keys) - 1; k >= 0; k-- {
		perm = stableSortBy(keys[k], perm, ascending[k])
	}
	return perm
}

// stableSortBy reorders perm stably by one key column.
func stableSortBy(c column, perm []int, asc bool) []int {
	present := make([]int, 0, len(perm))
	var missing []int
	for _, i := range perm {
		if c.isNA(i) {
			missing = append(missing, i)
		} else {
			present = append(present, i)
		}
	}
	if keys, ok := radixKeys(c, present, asc); ok && len(present) >= radixMin {
		radixSort(keys, present)
	} else if asc {
		slices.SortStableFunc(present, c.compare)
	} else {
		slices.SortStableFunc(present, func(a, b int) int { return c.compare(b, a) })
	}
	return append(present, missing...)
}

// radixKeys returns order-preserving uint64 keys of the rows at pos, for the
// dtypes that have them, complemented for a descending sort.
func radixKeys(c column, pos []int, asc bool) ([]uint64, bool) {
	keys := make([]uint64, len(pos))
	switch p := c.(type) {
	case *col[float64]:
		for k, i := range pos {
			keys[k] = floatKey(p.data[i])
		}
	case *col[int64]:
		for k, i := range pos {
			keys[k] = uint64(p.data[i]) ^ 1<<63
		}
	case *col[int32]:
		for k, i := range pos {
			keys[k] = uint64(int64(p.data[i])) ^ 1<<63
		}
	default:
		return nil, false
	}
	if !asc {
		for k := range keys {
			keys[k] = ^keys[k]
		}
	}
	return keys, true
}

// floatKey maps a float64 (not NaN) to a uint64 of the same order; -0 and +0
// map to one key, as they compare equal.
func floatKey(f float64) uint64 {
	if f == 0 {
		return 1 << 63
	}
	b := math.Float64bits(f)
	if b>>63 == 1 {
		return ^b
	}
	return b | 1<<63
}

// radixSort sorts keys and carries pos along, stably: an LSD radix sort on
// 11-bit digits that skips the digits all keys share.
func radixSort(keys []uint64, pos []int) {
	const bits, buckets = 11, 1 << 11
	n := len(keys)
	orig := pos
	var hist [6][buckets]int
	for _, k := range keys {
		for d := range 6 {
			hist[d][k>>(d*bits)&(buckets-1)]++
		}
	}
	k2 := make([]uint64, n)
	p2 := make([]int, n)
	for d := range 6 {
		h := &hist[d]
		if h[keys[0]>>(d*bits)&(buckets-1)] == n {
			continue
		}
		sum := 0
		for b := range h {
			h[b], sum = sum, sum+h[b]
		}
		shift := uint(d * bits)
		for i, k := range keys {
			b := k >> shift & (buckets - 1)
			j := h[b]
			h[b]++
			k2[j] = k
			p2[j] = pos[i]
		}
		keys, k2 = k2, keys
		pos, p2 = p2, pos
	}
	// After an odd number of passes the sorted positions sit in the
	// scratch slice; copy them home.
	if &pos[0] != &orig[0] {
		copy(orig, pos)
	}
}

// SortValues returns the series sorted by value, stably, missing values
// last.
func (s *Series) SortValues(ascending bool) *Series {
	return s.take(sortPerm([]column{s.c}, []bool{ascending}))
}

// SortIndex returns the series sorted by its index labels.
func (s *Series) SortIndex(ascending bool) *Series {
	return s.take(sortPerm(indexColumns(s.index), repeatBool(ascending, s.index.NLevels())))
}

// SortValues returns the frame sorted by the given columns, the first the
// most significant. ascending gives the direction: none means all ascending,
// one value applies to every key, otherwise one per key. The sort is stable
// and puts missing values last, like pandas' sort_values(kind="stable").
func (df *DataFrame) SortValues(by []string, ascending ...bool) (*DataFrame, error) {
	if len(by) == 0 {
		return nil, fmt.Errorf("%w: SortValues needs a column", ErrArgument)
	}
	asc, err := directions(ascending, len(by))
	if err != nil {
		return nil, err
	}
	keys := make([]column, len(by))
	for k, name := range by {
		i, err := df.pos(name)
		if err != nil {
			return nil, err
		}
		keys[k] = df.cols[i]
	}
	return df.take(sortPerm(keys, asc)), nil
}

// SortIndex returns the frame sorted by its index labels (every level, the
// first the most significant).
func (df *DataFrame) SortIndex(ascending bool) *DataFrame {
	return df.take(sortPerm(indexColumns(df.index), repeatBool(ascending, df.index.NLevels())))
}

func directions(asc []bool, n int) ([]bool, error) {
	switch len(asc) {
	case 0:
		return repeatBool(true, n), nil
	case 1:
		return repeatBool(asc[0], n), nil
	case n:
		return asc, nil
	}
	return nil, fmt.Errorf("%w: %d directions for %d keys", ErrLength, len(asc), n)
}

func repeatBool(b bool, n int) []bool {
	out := make([]bool, n)
	for i := range out {
		out[i] = b
	}
	return out
}
