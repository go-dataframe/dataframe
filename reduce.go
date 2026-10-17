package dataframe

import (
	"fmt"
	"math"
	"slices"

	"github.com/go-ndarray/ndarray"
)

// The reductions skip missing values, as pandas does by default (skipna).

// presentFloats returns the present values of a real or bool column as
// float64s, and false for another dtype.
func presentFloats(c column) ([]float64, bool) {
	switch p := c.(type) {
	case *col[float64]:
		out := make([]float64, 0, len(p.data))
		for i, v := range p.data {
			if v == v && p.valid.get(i) {
				out = append(out, v)
			}
		}
		return out, true
	case *col[int64]:
		if p.dt != Int64 {
			return nil, false
		}
		out := make([]float64, 0, len(p.data))
		for i, v := range p.data {
			if p.valid.get(i) {
				out = append(out, float64(v))
			}
		}
		return out, true
	}
	f, ok := floatsOf(c)
	if !ok {
		return nil, false
	}
	return slices.DeleteFunc(f, math.IsNaN), true
}

// sumFloat sums with pairwise summation over blocks, as NumPy does, so that
// the rounding matches it closely.
func sumFloat(x []float64) float64 {
	if len(x) <= 128 {
		s := 0.0
		for _, v := range x {
			s += v
		}
		return s
	}
	m := len(x) / 2
	return sumFloat(x[:m]) + sumFloat(x[m:])
}

// colSum returns the sum of a column: int64 for signed integers and bools,
// uint64 for unsigned ones, float64 for floats, complex128 for complex; nil
// for a dtype without a sum.
func colSum(c column) any {
	dt := c.dtype()
	switch {
	case dt.isInt() || dt == Bool:
		var s int64
		var u uint64
		forInts(c, func(i int64, v uint64) { s += i; u += v })
		if dt == Uint8 || dt == Uint16 || dt == Uint32 || dt == Uint64 {
			return u
		}
		return s
	case dt == Complex64 || dt == Complex128:
		var z complex128
		a, _ := c.ndarray()
		for i := range c.len() {
			if !c.isNA(i) {
				z += a.AtComplex(i)
			}
		}
		return z
	}
	f, ok := presentFloats(c)
	if !ok {
		return nil
	}
	return sumFloat(f)
}

// forInts calls f with every present value of an integer or bool column, as
// int64 and as uint64.
func forInts(c column, f func(int64, uint64)) {
	switch p := c.(type) {
	case *col[int64]:
		for i, v := range p.data {
			if p.valid.get(i) {
				f(v, uint64(v))
			}
		}
		return
	}
	a, _ := c.ndarray()
	b := a.AsType(ndarray.Int64)
	u := a.AsType(ndarray.Uint64)
	bi := arrayData[int64](b)
	bu := arrayData[uint64](u)
	for i := range bi {
		if !c.isNA(i) {
			f(bi[i], bu[i])
		}
	}
}

// mean, variance and friends over the present values.

func meanOf(x []float64) float64 {
	if len(x) == 0 {
		return math.NaN()
	}
	return sumFloat(x) / float64(len(x))
}

func varOf(x []float64, ddof int) float64 {
	n := len(x)
	if n-ddof <= 0 {
		return math.NaN()
	}
	m := meanOf(x)
	s := 0.0
	for _, v := range x {
		d := v - m
		s += d * d
	}
	return s / float64(n-ddof)
}

// quantileOf returns the q-quantile with linear interpolation (NumPy's and
// pandas' default), sorting x in place.
func quantileOf(x []float64, q float64) float64 {
	if len(x) == 0 {
		return math.NaN()
	}
	slices.Sort(x)
	pos := q * float64(len(x)-1)
	lo := math.Floor(pos)
	i := int(lo)
	if i+1 >= len(x) {
		return x[len(x)-1]
	}
	frac := pos - lo
	return x[i] + (x[i+1]-x[i])*frac
}

// colMinMax returns the smallest (or largest) present value, boxed; nil when
// there is none or the dtype has no order (complex).
func colMinMax(c column, max bool) any {
	dt := c.dtype()
	if dt == Complex64 || dt == Complex128 {
		return nil
	}
	if p, ok := c.(*col[float64]); ok {
		best, found := 0.0, false
		for i, v := range p.data {
			if v != v || !p.valid.get(i) {
				continue
			}
			if !found || (max && v > best) || (!max && v < best) {
				best, found = v, true
			}
		}
		if !found {
			return nil
		}
		return best
	}
	best := -1
	for i := range c.len() {
		if c.isNA(i) {
			continue
		}
		if best < 0 {
			best = i
			continue
		}
		r := c.compare(i, best)
		if (max && r > 0) || (!max && r < 0) {
			best = i
		}
	}
	if best < 0 {
		return nil
	}
	return c.value(best)
}

// Count returns the number of present values.
func (s *Series) Count() int { return s.Len() - s.c.naCount() }

// Sum returns the sum of the present values as a float64 (NaN for a dtype
// without a sum). SumValue keeps the exact type.
func (s *Series) Sum() float64 { return asFloat(colSum(s.c)) }

// SumValue returns the sum in the dtype pandas gives it: int64 for signed
// integers and bools, uint64 for unsigned ones, float64, or complex128; nil
// for a dtype without a sum.
func (s *Series) SumValue() any { return colSum(s.c) }

// Mean returns the mean of the present values (NaN if there are none or the
// dtype is not real).
func (s *Series) Mean() float64 {
	f, _ := presentFloats(s.c)
	return meanOf(f)
}

// Var returns the sample variance (ddof 1, as pandas).
func (s *Series) Var() float64 { return s.VarDDOF(1) }

// VarDDOF returns the variance with the given delta degrees of freedom.
func (s *Series) VarDDOF(ddof int) float64 {
	f, _ := presentFloats(s.c)
	return varOf(f, ddof)
}

// Std returns the sample standard deviation (ddof 1, as pandas).
func (s *Series) Std() float64 { return math.Sqrt(s.Var()) }

// Median returns the median of the present values.
func (s *Series) Median() float64 { return s.Quantile(0.5) }

// Quantile returns the q-quantile, 0 <= q <= 1, interpolating linearly.
func (s *Series) Quantile(q float64) float64 {
	f, _ := presentFloats(s.c)
	return quantileOf(f, q)
}

// Min returns the smallest present value (a Go value of the series' kind;
// time.Time for datetimes), nil when there is none.
func (s *Series) Min() any { return colMinMax(s.c, false) }

// Max returns the largest present value.
func (s *Series) Max() any { return colMinMax(s.c, true) }

// Prod returns the product of the present values.
func (s *Series) Prod() float64 {
	f, _ := presentFloats(s.c)
	p := 1.0
	for _, v := range f {
		p *= v
	}
	return p
}

// NUnique returns the number of distinct present values.
func (s *Series) NUnique() int {
	_, u := s.c.factorize()
	return u.len()
}

// Unique returns the distinct present values in order of appearance.
func (s *Series) Unique() *Series {
	_, u := s.c.factorize()
	return &Series{name: s.name, index: NewRangeIndex(u.len()), c: u}
}

// ValueCounts returns how often each distinct value occurs, most frequent
// first (ties in order of first appearance), indexed by the values and named
// "count", as pandas' value_counts.
func (s *Series) ValueCounts() *Series {
	codes, u := s.c.factorize()
	counts := make([]int64, u.len())
	for _, k := range codes {
		if k >= 0 {
			counts[k]++
		}
	}
	order := make([]int, len(counts))
	for i := range order {
		order[i] = i
	}
	slices.SortStableFunc(order, func(a, b int) int {
		switch {
		case counts[a] > counts[b]:
			return -1
		case counts[a] < counts[b]:
			return 1
		}
		return 0
	})
	vals := make([]int64, len(order))
	for r, k := range order {
		vals[r] = counts[k]
	}
	ix := labelIndex([]string{s.name}, []column{u.take(order)})
	return &Series{name: "count", index: ix, c: newCol(Int64, vals, nil)}
}

func asFloat(v any) float64 {
	f, ok := toFloat(v)
	if !ok {
		return math.NaN()
	}
	return f
}

// Agg reduces the series by name: "sum", "mean", "std", "var", "min",
// "max", "count", "median", "prod", "nunique", "first", "last" or "size".
func (s *Series) Agg(name string) (any, error) {
	switch name {
	case "sum":
		return colSum(s.c), nil
	case "mean":
		return s.Mean(), nil
	case "std":
		return s.Std(), nil
	case "var":
		return s.Var(), nil
	case "min":
		return s.Min(), nil
	case "max":
		return s.Max(), nil
	case "count":
		return int64(s.Count()), nil
	case "size":
		return int64(s.Len()), nil
	case "median":
		return s.Median(), nil
	case "prod":
		return s.Prod(), nil
	case "nunique":
		return int64(s.NUnique()), nil
	case "first", "last":
		d := s.DropNA()
		if d.Len() == 0 {
			return nil, nil
		}
		if name == "first" {
			return d.c.value(0), nil
		}
		return d.c.value(d.Len() - 1), nil
	}
	return nil, fmt.Errorf("%w: unknown aggregation %q", ErrArgument, name)
}

// numericColumns returns the positions of real-number and bool columns.
func (df *DataFrame) numericColumns() []int {
	var out []int
	for k, c := range df.cols {
		if dt := c.dtype(); dt.isReal() || dt == Bool {
			out = append(out, k)
		}
	}
	return out
}

// reduceFrame applies a reduction to every numeric column (pandas'
// numeric_only=True) and returns a series indexed by column name; it is
// int64 when every result is an integer from an integer column, float64
// otherwise.
func (df *DataFrame) reduceFrame(name string) *Series {
	cols := df.numericColumns()
	names := make([]string, len(cols))
	vals := make([]any, len(cols))
	allInt := true
	for k, j := range cols {
		names[k] = df.names[j]
		v, _ := (&Series{c: df.cols[j]}).Agg(name)
		switch x := v.(type) {
		case int64:
		case bool:
			v = int64(0)
			if x {
				v = int64(1)
			}
		default:
			if i, ok := toInt(v); ok && df.cols[j].dtype().isInt() && (name == "min" || name == "max") {
				v = i
			} else {
				allInt = false
			}
		}
		vals[k] = v
	}
	var c column
	if allInt {
		ints := make([]int64, len(vals))
		for k, v := range vals {
			ints[k] = v.(int64)
		}
		c = newCol(Int64, ints, nil)
	} else {
		fs := make([]float64, len(vals))
		for k, v := range vals {
			fs[k] = asFloat(v)
		}
		c = newCol(Float64, fs, nil)
	}
	ix, _ := NewIndex("", names)
	return &Series{index: ix, c: c}
}

// Sum returns the sum of each numeric column.
func (df *DataFrame) Sum() *Series { return df.reduceFrame("sum") }

// Mean returns the mean of each numeric column.
func (df *DataFrame) Mean() *Series { return df.reduceFrame("mean") }

// Std returns the sample standard deviation of each numeric column.
func (df *DataFrame) Std() *Series { return df.reduceFrame("std") }

// Var returns the sample variance of each numeric column.
func (df *DataFrame) Var() *Series { return df.reduceFrame("var") }

// Min returns the minimum of each numeric column.
func (df *DataFrame) Min() *Series { return df.reduceFrame("min") }

// Max returns the maximum of each numeric column.
func (df *DataFrame) Max() *Series { return df.reduceFrame("max") }

// Median returns the median of each numeric column.
func (df *DataFrame) Median() *Series { return df.reduceFrame("median") }

// Count returns the number of present values in every column.
func (df *DataFrame) Count() *Series {
	vals := make([]int64, len(df.cols))
	for k, c := range df.cols {
		vals[k] = int64(c.len() - c.naCount())
	}
	ix, _ := NewIndex("", df.Columns())
	return &Series{index: ix, c: newCol(Int64, vals, nil)}
}

// Describe returns summary statistics of the numeric columns, as pandas'
// describe(): count, mean, std, min, 25%, 50%, 75%, max. A frame without
// numeric columns is described as text: count, unique, top, freq.
func (df *DataFrame) Describe() *DataFrame {
	var cols []int
	for _, k := range df.numericColumns() {
		if df.cols[k].dtype() != Bool {
			cols = append(cols, k)
		}
	}
	if len(cols) == 0 {
		return df.describeText()
	}
	stats := []string{"count", "mean", "std", "min", "25%", "50%", "75%", "max"}
	ix, _ := NewIndex("", stats)
	out := &DataFrame{index: ix}
	for _, j := range cols {
		f, _ := presentFloats(df.cols[j])
		v := []float64{float64(len(f)), meanOf(f), math.Sqrt(varOf(f, 1)), math.NaN(),
			quantileOf(f, 0.25), quantileOf(f, 0.5), quantileOf(f, 0.75), math.NaN()}
		if len(f) > 0 {
			v[3], v[7] = f[0], f[len(f)-1] // sorted by quantileOf
		}
		out.names = append(out.names, df.names[j])
		out.cols = append(out.cols, newCol(Float64, v, nil))
	}
	return out
}

func (df *DataFrame) describeText() *DataFrame {
	ix, _ := NewIndex("", []string{"count", "unique", "top", "freq"})
	out := &DataFrame{index: ix}
	for k, c := range df.cols {
		s := &Series{name: df.names[k], index: df.index, c: c}
		vc := s.ValueCounts()
		vals := []any{int64(s.Count()), int64(vc.Len()), nil, nil}
		if vc.Len() > 0 {
			vals[2] = vc.index.levels[0].str(0)
			vals[3] = vc.c.value(0)
		}
		strs, valid := make([]string, 4), allValid(4)
		for i, v := range vals {
			if v == nil {
				valid.clear(i)
			} else {
				strs[i] = fmt.Sprint(v)
			}
		}
		col := newStrCol(strs, valid)
		out.names = append(out.names, df.names[k])
		out.cols = append(out.cols, col)
	}
	return out
}
