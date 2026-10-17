package dataframe

import (
	"fmt"
	"slices"
)

// MergeOptions describes a database-style join of two frames.
type MergeOptions struct {
	// How is "inner" (the default), "left", "right", "outer" or "cross".
	How string
	// On names key columns both frames have. When On, LeftOn and RightOn
	// are all empty, the keys are the columns the frames have in common.
	On []string
	// LeftOn and RightOn name key columns that differ between the frames,
	// pairwise.
	LeftOn, RightOn []string
	// Suffixes are appended to non-key columns both frames have; the
	// default is "_x" and "_y".
	Suffixes [2]string
	// Indicator adds a column "_merge" saying where each row came from:
	// "both", "left_only" or "right_only".
	Indicator bool
}

// Merge joins two frames on key columns, as pandas' merge. Rows come in
// pandas' order: inner and left joins follow the left frame, a right join
// the right frame, an outer join the sorted keys. Missing keys match each
// other, as in pandas. The result has a range index.
func Merge(left, right *DataFrame, opts MergeOptions) (*DataFrame, error) {
	how := opts.How
	if how == "" {
		how = "inner"
	}
	if opts.Suffixes == [2]string{} {
		opts.Suffixes = [2]string{"_x", "_y"}
	}
	lon, ron := opts.LeftOn, opts.RightOn
	if len(opts.On) > 0 {
		lon, ron = opts.On, opts.On
	}
	if how == "cross" {
		lon, ron = nil, nil
	} else if len(lon) == 0 && len(ron) == 0 {
		for _, n := range left.names {
			if slices.Contains(right.names, n) {
				lon = append(lon, n)
			}
		}
		ron = lon
		if len(lon) == 0 {
			return nil, fmt.Errorf("%w: no common columns to merge on", ErrArgument)
		}
	}
	if len(lon) != len(ron) {
		return nil, fmt.Errorf("%w: %d left keys and %d right keys", ErrLength, len(lon), len(ron))
	}
	lk, err := keyColumns(left, lon)
	if err != nil {
		return nil, err
	}
	rk, err := keyColumns(right, ron)
	if err != nil {
		return nil, err
	}
	var li, ri []int
	switch how {
	case "cross":
		for i := range left.Len() {
			for j := range right.Len() {
				li, ri = append(li, i), append(ri, j)
			}
		}
	case "inner", "left", "right", "outer":
		li, ri, err = joinRows(lk, rk, how)
		if err != nil {
			return nil, err
		}
	default:
		return nil, fmt.Errorf("%w: unknown join %q", ErrArgument, how)
	}
	return assemble(left, right, lon, ron, li, ri, opts), nil
}

// Merge joins the frame with another, as the Merge function.
func (df *DataFrame) Merge(right *DataFrame, opts MergeOptions) (*DataFrame, error) {
	return Merge(df, right, opts)
}

func keyColumns(df *DataFrame, names []string) ([]column, error) {
	out := make([]column, len(names))
	for k, n := range names {
		i, err := df.pos(n)
		if err != nil {
			return nil, err
		}
		out[k] = df.cols[i]
	}
	return out, nil
}

// joinRows returns the pairs of left and right rows a join produces (-1 for
// the side a row lacks), in pandas' order.
func joinRows(lk, rk []column, how string) ([]int, []int, error) {
	ca, cb, n, err := jointCodes(lk, rk)
	if err != nil {
		return nil, nil, err
	}
	// Missing keys match each other: give them a code of their own.
	for _, s := range [][]int{ca, cb} {
		for i, c := range s {
			if c < 0 {
				s[i] = n
			}
		}
	}
	n++
	rrows := bucket(cb, n)
	var lrows [][]int // the left rows of each code, for right and outer joins
	if how != "inner" && how != "left" {
		lrows = bucket(ca, n)
	}
	var li, ri []int
	emit := func(c int, i int, fromLeft bool) {
		other := rrows[c]
		if !fromLeft {
			other = lrows[c]
		}
		if len(other) == 0 {
			if fromLeft {
				li, ri = append(li, i), append(ri, -1)
			} else {
				li, ri = append(li, -1), append(ri, i)
			}
			return
		}
		for _, j := range other {
			if fromLeft {
				li, ri = append(li, i), append(ri, j)
			} else {
				li, ri = append(li, j), append(ri, i)
			}
		}
	}
	switch how {
	case "inner", "left":
		total := 0
		for _, c := range ca {
			m := len(rrows[c])
			if m == 0 && how == "left" {
				m = 1
			}
			total += m
		}
		li, ri = make([]int, 0, total), make([]int, 0, total)
		for i, c := range ca {
			if how == "left" || len(rrows[c]) > 0 {
				emit(c, i, true)
			}
		}
	case "right":
		for j, c := range cb {
			emit(c, j, false)
		}
	default:
		// Outer: keys in sorted order; for each, the left rows against
		// the right rows, or each alone.
		keyRow := make([]int, n) // a representative: left row, or ^right row
		order := make([]int, 0, n)
		seen := make([]bool, n)
		for i, c := range ca {
			if !seen[c] {
				seen[c], keyRow[c] = true, i
				order = append(order, c)
			}
		}
		for j, c := range cb {
			if !seen[c] {
				seen[c], keyRow[c] = true, ^j
				order = append(order, c)
			}
		}
		both := make([]column, len(lk))
		for k := range lk {
			dt, _ := commonType(lk[k].dtype(), rk[k].dtype())
			a, _ := cast(lk[k], dt)
			b, _ := cast(rk[k], dt)
			both[k] = a.appendCol(b)
		}
		nl := lk[0].len()
		at := func(c int) int {
			if r := keyRow[c]; r >= 0 {
				return r
			}
			return nl + ^keyRow[c]
		}
		slices.SortStableFunc(order, func(a, b int) int { return compareRows(both, at(a), at(b)) })
		for _, c := range order {
			if len(lrows[c]) == 0 {
				for _, j := range rrows[c] {
					li, ri = append(li, -1), append(ri, j)
				}
				continue
			}
			for _, i := range lrows[c] {
				emit(c, i, true)
			}
		}
	}
	return li, ri, nil
}

// bucket lists the rows of each code, in row order.
func bucket(codes []int, n int) [][]int {
	cnt := make([]int, n+1)
	for _, c := range codes {
		cnt[c+1]++
	}
	for k := range n {
		cnt[k+1] += cnt[k]
	}
	flat := make([]int, len(codes))
	next := slices.Clone(cnt[:n])
	for i, c := range codes {
		flat[next[c]] = i
		next[c]++
	}
	out := make([][]int, n)
	for k := range n {
		out[k] = flat[cnt[k]:cnt[k+1]:cnt[k+1]]
	}
	return out
}

// assemble builds the joined frame from row pairs.
func assemble(left, right *DataFrame, lon, ron []string, li, ri []int, opts MergeOptions) *DataFrame {
	out := &DataFrame{index: NewRangeIndex(len(li))}
	shared := func(n string, theirs *DataFrame) bool {
		return slices.Contains(theirs.names, n) && !sameKeyName(lon, ron, n)
	}
	// When every left row appears once, in order (a left or inner join on
	// keys unique on the right, all matched), the left columns are reused
	// as they are.
	leftAsIs := len(li) == left.Len() && !slices.ContainsFunc(li, func(i int) bool { return i < 0 })
	for k, i := range li {
		if !leftAsIs {
			break
		}
		leftAsIs = i == k
	}
	leftCols := make([]column, len(left.cols))
	eachColumn(len(li), len(left.cols), func(k int) {
		leftCols[k] = left.cols[k]
		if !leftAsIs {
			leftCols[k] = left.cols[k].take(li)
		}
	})
	rightCols := make([]column, len(right.cols))
	eachColumn(len(ri), len(right.cols), func(k int) { rightCols[k] = right.cols[k].take(ri) })
	for k, n := range left.names {
		c := leftCols[k]
		if p := slices.Index(lon, n); p >= 0 && lon[p] == ron[p] && slices.Contains(li, -1) {
			// A key both sides call alike: one column, filled from the
			// right where the left row is absent.
			j, _ := right.pos(ron[p])
			c = coalesce(c, rightCols[j], li)
		}
		name := n
		if shared(n, right) {
			name += opts.Suffixes[0]
		}
		out.names = append(out.names, name)
		out.cols = append(out.cols, c)
	}
	for k, n := range right.names {
		if p := slices.Index(ron, n); p >= 0 && lon[p] == ron[p] {
			continue
		}
		name := n
		if shared(n, left) {
			name += opts.Suffixes[1]
		}
		out.names = append(out.names, name)
		out.cols = append(out.cols, rightCols[k])
	}
	if opts.Indicator {
		ind := make([]string, len(li))
		for r := range li {
			switch {
			case li[r] < 0:
				ind[r] = "right_only"
			case ri[r] < 0:
				ind[r] = "left_only"
			default:
				ind[r] = "both"
			}
		}
		cat, _ := toCategory(newStrCol(ind, nil))
		out.names = append(out.names, "_merge")
		out.cols = append(out.cols, cat)
	}
	return out
}

func sameKeyName(lon, ron []string, n string) bool {
	p := slices.Index(lon, n)
	q := slices.Index(ron, n)
	return p >= 0 && p == q
}

// coalesce takes a's value, or b's where row li is absent.
// The two sides were matched already, so they have a common dtype.
func coalesce(a, b column, li []int) column {
	dt, _ := commonType(a.dtype(), b.dtype())
	a, _ = cast(a, dt)
	b, _ = cast(b, dt)
	idx := make([]int, len(li))
	n := len(li)
	for r := range li {
		idx[r] = r
		if li[r] < 0 {
			idx[r] = n + r
		}
	}
	return a.appendCol(b).take(idx)
}

// Join joins another frame on the index labels (pandas' join): how is
// "left" (the default when empty), "right", "inner" or "outer"; column names
// both frames have take the suffixes.
func (df *DataFrame) Join(other *DataFrame, how, lsuffix, rsuffix string) (*DataFrame, error) {
	if how == "" {
		how = "left"
	}
	for _, n := range df.names {
		if slices.Contains(other.names, n) && lsuffix == "" && rsuffix == "" {
			return nil, fmt.Errorf("%w: columns overlap (%q); give suffixes", ErrArgument, n)
		}
	}
	li, ri, err := joinRows(indexColumns(df.index), indexColumns(other.index), how)
	if err != nil {
		return nil, err
	}
	l := df.ResetIndex(false)
	nl := df.index.NLevels()
	r := &DataFrame{index: NewRangeIndex(other.Len()), names: other.names, cols: other.cols}
	merged := assemble(&DataFrame{index: l.index, names: l.names[nl:], cols: l.cols[nl:]}, r,
		nil, nil, li, ri, MergeOptions{Suffixes: [2]string{lsuffix, rsuffix}})
	levels := make([]column, nl)
	for k := range levels {
		levels[k] = coalesce(l.cols[k].take(li), other.index.level(k).take(ri), li)
	}
	merged.index = labelIndex(df.index.names, levels)
	return merged, nil
}
