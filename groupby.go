package dataframe

import (
	"fmt"
	"math"
	"slices"
	"strings"
)

// GroupBy is a frame split into groups of rows by the values of key
// columns, ready to aggregate. Groups come out sorted by key, and rows whose
// key is missing are left out, as pandas' groupby does by default
// (sort=True, dropna=True). Grouping is hash based: one pass to code the
// keys, then one pass per aggregation.
type GroupBy struct {
	df      *DataFrame
	keys    []string
	codes   []int // group of each row, -1 when a key is missing
	ngroups int
	reps    []int // a row of each group, in group order
	values  []string
	bins    *Index // the result index, when it is not made of key values
}

// GroupBy groups the frame's rows by the named key columns.
func (df *DataFrame) GroupBy(keys ...string) (*GroupBy, error) {
	if len(keys) == 0 {
		return nil, fmt.Errorf("%w: GroupBy needs a key", ErrArgument)
	}
	cols := make([]column, len(keys))
	for k, name := range keys {
		i, err := df.pos(name)
		if err != nil {
			return nil, err
		}
		cols[k] = df.cols[i]
	}
	codes, n, sorted := sortedCodes(cols)
	gb := &GroupBy{df: df, keys: slices.Clone(keys), codes: codes, ngroups: n, reps: sorted}
	for _, name := range df.names {
		if !slices.Contains(keys, name) {
			gb.values = append(gb.values, name)
		}
	}
	return gb, nil
}

// groupCodes codes rows by key tuple.
func groupCodes(cols []column) ([]int, int) {
	codes, _, n, _ := jointCodes(cols, nil)
	return codes, n
}

// NGroups returns the number of groups.
func (g *GroupBy) NGroups() int { return g.ngroups }

// Select restricts the aggregated columns to the named ones (pandas'
// df.groupby(k)[cols]).
func (g *GroupBy) Select(columns ...string) (*GroupBy, error) {
	for _, n := range columns {
		if _, err := g.df.pos(n); err != nil {
			return nil, err
		}
	}
	out := *g
	out.values = slices.Clone(columns)
	return &out, nil
}

// index returns the result index: the group keys, in group order.
func (g *GroupBy) index() *Index {
	if g.bins != nil {
		return g.bins
	}
	levels := make([]column, len(g.keys))
	for k, name := range g.keys {
		i, _ := g.df.pos(name)
		levels[k] = g.df.cols[i].take(g.reps)
	}
	return labelIndex(slices.Clone(g.keys), levels)
}

// numericOnly reports the aggregations pandas applies to numeric columns
// only here (summing strings is not a thing this package does).
func numericOnly(fn string) bool {
	switch fn {
	case "sum", "mean", "std", "var", "median", "prod":
		return true
	}
	return false
}

// aggregate applies one aggregation to every value column it fits.
func (g *GroupBy) aggregate(fn string) (*DataFrame, error) {
	out := &DataFrame{index: g.index()}
	var cols []column
	for _, name := range g.values {
		i, _ := g.df.pos(name)
		c := g.df.cols[i]
		if numericOnly(fn) && !(c.dtype().isReal() || c.dtype() == Bool) {
			continue
		}
		out.names = append(out.names, name)
		cols = append(cols, c)
	}
	out.cols = make([]column, len(cols))
	errs := make([]error, len(cols))
	eachColumn(len(g.codes), len(cols), func(k int) { out.cols[k], errs[k] = g.aggCol(cols[k], fn) })
	for k, err := range errs {
		if err != nil {
			return nil, fmt.Errorf("column %q: %w", out.names[k], err)
		}
	}
	return out, nil
}

// Sum returns the per-group sum of each numeric column.
func (g *GroupBy) Sum() (*DataFrame, error) { return g.aggregate("sum") }

// Mean returns the per-group mean of each numeric column.
func (g *GroupBy) Mean() (*DataFrame, error) { return g.aggregate("mean") }

// Std returns the per-group sample standard deviation.
func (g *GroupBy) Std() (*DataFrame, error) { return g.aggregate("std") }

// Var returns the per-group sample variance.
func (g *GroupBy) Var() (*DataFrame, error) { return g.aggregate("var") }

// Median returns the per-group median.
func (g *GroupBy) Median() (*DataFrame, error) { return g.aggregate("median") }

// Min returns the per-group minimum of each column.
func (g *GroupBy) Min() (*DataFrame, error) { return g.aggregate("min") }

// Max returns the per-group maximum of each column.
func (g *GroupBy) Max() (*DataFrame, error) { return g.aggregate("max") }

// First returns the first present value of each column in each group.
func (g *GroupBy) First() (*DataFrame, error) { return g.aggregate("first") }

// Last returns the last present value of each column in each group.
func (g *GroupBy) Last() (*DataFrame, error) { return g.aggregate("last") }

// Count returns the number of present values of each column in each group.
func (g *GroupBy) Count() (*DataFrame, error) { return g.aggregate("count") }

// NUnique returns the number of distinct present values per group.
func (g *GroupBy) NUnique() (*DataFrame, error) { return g.aggregate("nunique") }

// Size returns the number of rows of each group, as a series named "size".
func (g *GroupBy) Size() *Series {
	n := make([]int64, g.ngroups)
	for _, c := range g.codes {
		if c >= 0 {
			n[c]++
		}
	}
	return &Series{name: "size", index: g.index(), c: newCol(Int64, n, nil)}
}

// AggSpec names the aggregations of one column, for GroupBy.Agg.
type AggSpec struct {
	Column string
	Funcs  []string
}

// Agg returns an AggSpec: the column and the aggregations to apply to it
// ("sum", "mean", "std", "var", "min", "max", "median", "count", "size",
// "first", "last", "nunique", "prod").
func Agg(column string, funcs ...string) AggSpec { return AggSpec{column, funcs} }

// Agg applies several aggregations. A column with one function keeps its
// name; with several, each result is named column_func (pandas makes a
// two-level column index there; this package keeps column names flat).
func (g *GroupBy) Agg(specs ...AggSpec) (*DataFrame, error) {
	out := &DataFrame{index: g.index()}
	for _, sp := range specs {
		i, err := g.df.pos(sp.Column)
		if err != nil {
			return nil, err
		}
		for _, fn := range sp.Funcs {
			r, err := g.aggCol(g.df.cols[i], fn)
			if err != nil {
				return nil, fmt.Errorf("column %q: %w", sp.Column, err)
			}
			name := sp.Column
			if len(sp.Funcs) > 1 {
				name += "_" + fn
			}
			out.names = append(out.names, name)
			out.cols = append(out.cols, r)
		}
	}
	return out, nil
}

// aggCol reduces one column per group.
func (g *GroupBy) aggCol(c column, fn string) (column, error) {
	n := g.ngroups
	switch fn {
	case "sum":
		return g.sum(c)
	case "mean", "std", "var":
		return g.moments(c, fn)
	case "count", "size":
		cnt := make([]int64, n)
		for i, k := range g.codes {
			if k >= 0 && (fn == "size" || !c.isNA(i)) {
				cnt[k]++
			}
		}
		return newCol(Int64, cnt, nil), nil
	case "min", "max", "first", "last":
		return g.pick(c, fn)
	case "median", "prod", "nunique":
		return g.perGroup(c, fn)
	}
	return nil, fmt.Errorf("%w: unknown aggregation %q", ErrArgument, fn)
}

// sum sums per group: int64 for integers and bools, float64 for floats.
func (g *GroupBy) sum(c column) (column, error) {
	n := g.ngroups
	switch p := c.(type) {
	case *col[float64]:
		s := makeCol[float64](Float64, n)
		for i, k := range g.codes {
			if v := p.data[i]; k >= 0 && v == v && p.valid.get(i) {
				s.data[k] += v
			}
		}
		return s, nil
	case *col[int64]:
		if p.dt == Int64 {
			s := makeCol[int64](Int64, n)
			for i, k := range g.codes {
				if k >= 0 && p.valid.get(i) {
					s.data[k] += p.data[i]
				}
			}
			return s, nil
		}
	}
	dt := c.dtype()
	if !(dt.isReal() || dt == Bool) {
		return nil, fmt.Errorf("%w: sum of %v", ErrDType, dt)
	}
	f, _ := floatsOf(c)
	if dt.isFloat() {
		return g.sum(floatCol(f))
	}
	i64 := mustCast(c, Int64)
	return g.sum(i64)
}

// moments computes per-group mean, variance or standard deviation (ddof 1).
func (g *GroupBy) moments(c column, fn string) (column, error) {
	f, ok := floatsOf(c)
	if !ok {
		return nil, fmt.Errorf("%w: %s of %v", ErrDType, fn, c.dtype())
	}
	n := g.ngroups
	sum := make([]float64, n)
	cnt := make([]float64, n)
	for i, k := range g.codes {
		if v := f[i]; k >= 0 && v == v {
			sum[k] += v
			cnt[k]++
		}
	}
	out := makeCol[float64](Float64, n)
	for k := range out.data {
		out.data[k] = sum[k] / cnt[k]
	}
	if fn == "mean" {
		return out, nil
	}
	ss := make([]float64, n)
	for i, k := range g.codes {
		if v := f[i]; k >= 0 && v == v {
			d := v - out.data[k]
			ss[k] += d * d
		}
	}
	for k := range out.data {
		v := math.NaN()
		if cnt[k] > 1 {
			v = ss[k] / (cnt[k] - 1)
		}
		if fn == "std" {
			v = math.Sqrt(v)
		}
		out.data[k] = v
	}
	return out, nil
}

// pick chooses one row per group (the smallest, largest, first or last
// present value) and gathers them, which keeps the column's dtype.
func (g *GroupBy) pick(c column, fn string) (column, error) {
	if (fn == "min" || fn == "max") && (c.dtype() == Complex64 || c.dtype() == Complex128) {
		return nil, fmt.Errorf("%w: %s of %v", ErrDType, fn, c.dtype())
	}
	best := make([]int, g.ngroups)
	for i := range best {
		best[i] = -1
	}
	fp, isFloat := c.(*col[float64])
	for i, k := range g.codes {
		if k < 0 || c.isNA(i) {
			continue
		}
		b := best[k]
		switch {
		case b < 0, fn == "last":
			best[k] = i
		case fn == "first":
		case isFloat:
			if (fn == "min" && fp.data[i] < fp.data[b]) || (fn == "max" && fp.data[i] > fp.data[b]) {
				best[k] = i
			}
		default:
			r := c.compare(i, b)
			if (fn == "min" && r < 0) || (fn == "max" && r > 0) {
				best[k] = i
			}
		}
	}
	return c.take(best), nil
}

// groupRows returns the rows of each group, in row order: rows[off[k]:off[k+1]].
func (g *GroupBy) groupRows() (off, rows []int) {
	off = make([]int, g.ngroups+1)
	for _, k := range g.codes {
		if k >= 0 {
			off[k+1]++
		}
	}
	for k := range g.ngroups {
		off[k+1] += off[k]
	}
	next := slices.Clone(off[:g.ngroups])
	rows = make([]int, off[g.ngroups])
	for i, k := range g.codes {
		if k >= 0 {
			rows[next[k]] = i
			next[k]++
		}
	}
	return off, rows
}

// perGroup computes the aggregations that need each group's values
// together.
func (g *GroupBy) perGroup(c column, fn string) (column, error) {
	off, rows := g.groupRows()
	if fn == "nunique" {
		out := makeCol[int64](Int64, g.ngroups)
		for k := range g.ngroups {
			_, u := c.take(rows[off[k]:off[k+1]]).factorize()
			out.data[k] = int64(u.len())
		}
		return out, nil
	}
	f, ok := floatsOf(c)
	if !ok {
		return nil, fmt.Errorf("%w: %s of %v", ErrDType, fn, c.dtype())
	}
	out := makeCol[float64](Float64, g.ngroups)
	for k := range g.ngroups {
		var vals []float64
		for _, i := range rows[off[k]:off[k+1]] {
			if f[i] == f[i] {
				vals = append(vals, f[i])
			}
		}
		if fn == "median" {
			out.data[k] = quantileOf(vals, 0.5)
		} else {
			p := 1.0
			for _, v := range vals {
				p *= v
			}
			out.data[k] = p
		}
	}
	return out, nil
}

// Groups returns each group's key (a Go value, or a []any tuple for several
// keys) and its rows as a frame, in group order.
func (g *GroupBy) Groups() ([]any, []*DataFrame) {
	off, rows := g.groupRows()
	ix := g.index()
	keys := make([]any, g.ngroups)
	frames := make([]*DataFrame, g.ngroups)
	for k := range g.ngroups {
		keys[k] = ix.Label(k)
		frames[k] = g.df.take(rows[off[k]:off[k+1]])
	}
	return keys, frames
}

// Apply calls f on each group's rows and concatenates the frames it returns,
// in group order.
func (g *GroupBy) Apply(f func(key any, group *DataFrame) (*DataFrame, error)) (*DataFrame, error) {
	keys, frames := g.Groups()
	parts := make([]*DataFrame, len(frames))
	for k, fr := range frames {
		r, err := f(keys[k], fr)
		if err != nil {
			return nil, fmt.Errorf("group %v: %w", keys[k], err)
		}
		parts[k] = r
	}
	return Concat(parts...)
}

// Transform returns, for every row of the frame, its group's aggregate of
// the named column (pandas' groupby(k)[col].transform(fn)); rows whose key
// is missing get a missing value.
func (g *GroupBy) Transform(column, fn string) (*Series, error) {
	i, err := g.df.pos(column)
	if err != nil {
		return nil, err
	}
	agg, err := g.aggCol(g.df.cols[i], fn)
	if err != nil {
		return nil, err
	}
	return &Series{name: column, index: g.df.index, c: agg.take(g.codes)}, nil
}

// String describes the grouping.
func (g *GroupBy) String() string {
	return fmt.Sprintf("GroupBy(by=[%s], groups=%d)", strings.Join(g.keys, ", "), g.ngroups)
}
