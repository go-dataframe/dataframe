package dataframe

import (
	"math"
	"math/rand/v2"
	"reflect"
	"slices"
	"testing"
	"time"
)

func TestArithmetic(t *testing.T) {
	i := Must(NewSeries("i", []int{7, -7, 3, 0}))
	if v := Must(i.FloorDiv(2)); !reflect.DeepEqual(v.Items(), []any{int64(3), int64(-4), int64(1), int64(0)}) {
		t.Error("floordiv", v.Items())
	}
	if v := Must(i.Mod(-2)); !reflect.DeepEqual(v.Items(), []any{int64(-1), int64(-1), int64(-1), int64(0)}) {
		t.Error("mod", v.Items())
	}
	if v := Must(i.Mod(0)); v.Count() != 0 {
		t.Error("mod 0 missing")
	}
	if v := Must(i.Pow(2)); v.c.value(1) != int64(49) {
		t.Error("pow")
	}
	if v := Must(i.Pow(-1)); v.c.value(0) != 1.0/7 {
		t.Error("negative pow", v.c.value(0))
	}
	f := Must(NewSeries("f", []float64{7.5, -7.5}))
	if v := Must(f.FloorDiv(2)); v.c.value(1) != -4.0 {
		t.Error("float floordiv")
	}
	if v := Must(f.Pow(2.0)); v.c.value(0) != 56.25 {
		t.Error("float pow")
	}
	if v := Must(f.Mod(2)); v.c.value(1) != 0.5 {
		t.Error("float mod", v.c.value(1))
	}
	if v := Must(f.Sub(0.5)); v.c.value(0) != 7.0 {
		t.Error("sub")
	}
	if v := Must(f.Mul(2)); v.c.value(0) != 15.0 {
		t.Error("mul")
	}
	if v := Must(i.Div(2)); v.c.value(0) != 3.5 {
		t.Error("div")
	}
	if v := Must(i.Add(nil)); v.Count() != 0 {
		t.Error("add missing scalar")
	}
	f32 := Must(NewSeries("g", []float32{1}))
	if v := Must(f32.Add(0.5)); v.DType() != Float32 {
		t.Error("float32 + python float stays float32", v.DType())
	}
	i32 := Must(NewSeries("h", []int32{1}))
	if v := Must(i32.Add(1)); v.DType() != Int32 {
		t.Error("int32 + int stays int32")
	}
	if v := Must(i32.Add(1.5)); v.DType() != Float64 {
		t.Error("int32 + float is float64")
	}
	if v := Must(i32.Add(int64(1) << 40)); v.DType() != Int64 {
		t.Error("int32 + big int", v.DType())
	}
	s := Must(NewSeries("s", []string{"a", "b"}))
	if v := Must(s.Add("!")); v.c.value(1) != "b!" {
		t.Error("string concat scalar")
	}
	_, err := s.Sub("x")
	fails(t, err, ErrDType)
	_, err = s.Add(struct{}{})
	fails(t, err, ErrArgument)
	_, err = s.FloorDiv(1)
	fails(t, err, ErrDType)
	z := Must(NewSeries("z", []complex128{1}))
	_, err = z.Mod(1)
	fails(t, err, ErrDType)
	_, err = i.Add(Must(NewSeriesIndexed("x", []int{1}, Must(NewIndex("", []string{"a"})))))
	fails(t, err, ErrDType)
	if v := Must(i.Neg()); v.c.value(0) != int64(-7) {
		t.Error("neg")
	}
	_, err = s.Neg()
	fails(t, err, ErrDType)
	if v := Must(f.Abs()); v.c.value(1) != 7.5 {
		t.Error("abs")
	}
	_, err = s.Abs()
	fails(t, err, ErrDType)
	m := Must(NewSeriesMasked("m", []float64{4, 9}, []bool{true, false}))
	if v := Must(m.MapFloat(math.Sqrt)); v.c.value(0) != 2.0 || v.c.value(1) != nil {
		t.Error("MapFloat")
	}
	_, err = s.MapFloat(math.Sqrt)
	fails(t, err, ErrDType)
	if v := Must(s.Map(func(x any) any { return x.(string) + "?" })); v.c.value(0) != "a?" {
		t.Error("Map")
	}
	_, err = s.Map(func(x any) any { return struct{}{} })
	fails(t, err, ErrArgument)
	// Integer ops overflow into ndarray errors are not possible; division
	// by a broadcast series of another length is.
	_, err = arithCols(opAdd, newCol(Float64, []float64{1, 2}, nil), newCol(Float64, []float64{1, 2, 3}, nil))
	if err == nil {
		t.Error("shape mismatch")
	}
	if v := Must(arithCols(opFloorDiv, newCol(Int64, []int64{5, 6}, nil), newCol(Int64, []int64{2}, nil))); v.value(1) != int64(3) {
		t.Error("broadcast int floordiv")
	}
	if v := Must(arithCols(opMod, newCol(Float64, []float64{5, 6}, nil), newCol(Float64, []float64{4}, nil))); v.value(1) != 2.0 {
		t.Error("broadcast float mod")
	}
}

func TestApply(t *testing.T) {
	df := sample(t)
	r := Must(df.Apply("r", func(r Row) any {
		if r.Position() == 3 {
			return nil
		}
		return r.Float("a") + r.Float("c") + float64(len(r.Get("b").(string))) + float64(r.Label().(int64))*0
	}))
	if r.c.value(0) != 9.0 || r.c.value(3) != nil || r.Name() != "r" {
		t.Error("Apply", r.Items())
	}
	row := Row{df, 0}
	if row.Get("zz") != nil || !math.IsNaN(row.Float("b")) {
		t.Error("Row on missing column")
	}
	_, err := df.Apply("x", func(Row) any { return struct{}{} })
	fails(t, err, ErrArgument)
	ac := Must(df.ApplyColumns(func(s *Series) (*Series, error) { return s.Shift(1), nil }))
	if ac.cols[2].value(1) != int64(5) {
		t.Error("ApplyColumns")
	}
	_, err = df.ApplyColumns(func(s *Series) (*Series, error) { return s.Abs() })
	fails(t, err, ErrDType)
	_, err = df.ApplyColumns(func(s *Series) (*Series, error) { return s.Head(1), nil })
	fails(t, err, ErrLength)
}

func TestFrameArithmetic(t *testing.T) {
	a := Must(New(Col("p", []float64{1, 2}), Col("q", []int{3, 4})))
	if v := Must(a.Sub(1)); v.cols[1].value(0) != int64(2) {
		t.Error("Sub")
	}
	if v := Must(a.Mul(2)); v.cols[0].value(1) != 4.0 {
		t.Error("Mul")
	}
	if v := Must(a.Div(a)); v.cols[1].value(0) != 1.0 {
		t.Error("Div frames")
	}
	_, err := sample(t).Add(1)
	fails(t, err, ErrDType)
	s := Must(New(Col("p", []string{"x", "y"})))
	_, err = a.Add(s)
	fails(t, err, ErrDType)
	other := Must(a.WithIndex(Must(NewIndex("", []string{"a", "b"}))))
	_, err = a.Add(other)
	fails(t, err, ErrDType)
}

func TestComparisons(t *testing.T) {
	i := Must(NewSeriesMasked("i", []int{1, 2, 3}, []bool{true, true, false}))
	check := func(got *Series, want ...bool) {
		t.Helper()
		b := got.Valid()
		for k, w := range want {
			if got.c.value(k) != w {
				t.Errorf("%v: %v, want %v (valid %v)", got.Name(), got.Items(), want, b)
				return
			}
		}
	}
	check(Must(i.Eq(2)), false, true, false)
	check(Must(i.Ne(2)), true, false, true)
	check(Must(i.Lt(2)), true, false, false)
	check(Must(i.Le(2)), true, true, false)
	check(Must(i.Gt(1)), false, true, false)
	check(Must(i.Ge(2)), false, true, false)
	check(Must(i.Gt(1.5)), false, true, false)
	f := Must(NewSeries("f", []float64{1, math.NaN(), 3}))
	check(Must(f.Eq(1)), true, false, false)
	check(Must(f.Ne(1)), false, true, true)
	check(Must(f.Lt(3)), true, false, false)
	check(Must(f.Le(3)), true, false, true)
	check(Must(f.Ge(3)), false, false, true)
	check(Must(f.Gt(int64(1))), false, false, true)
	fi := Must(NewSeries("j", []int64{1, 2}))
	check(Must(fi.Eq(int64(2))), false, true)
	check(Must(fi.Ne(int64(2))), true, false)
	check(Must(fi.Lt(int64(2))), true, false)
	check(Must(fi.Le(int64(1))), true, false)
	check(Must(fi.Ge(int64(2))), false, true)
	check(Must(fi.Gt(1.5)), false, true)
	check(Must(Must(NewSeries("u", []uint8{1, 2})).Gt(1)), false, true)
	check(Must(Must(NewSeries("i8", []int8{1, 2})).Gt(int8(1))), false, true)
	check(Must(f.Gt(nil)), false, false, false)
	s := Must(NewSeries("s", []string{"a", "b"}))
	check(Must(s.Lt("b")), true, false)
	check(Must(s.Eq(Must(NewSeries("t", []string{"a", "c"})))), true, false)
	_, err := s.Lt(1)
	fails(t, err, ErrDType)
	z := Must(NewSeries("z", []complex128{1, 2}))
	check(Must(z.Eq(1)), true, false)
	_, err = z.Lt(1)
	fails(t, err, ErrDType)
	_, err = s.Eq(struct{}{})
	fails(t, err, ErrArgument)
	ts := Must(NewSeries("t", []time.Time{time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC), time.Date(2021, 1, 1, 0, 0, 0, 0, time.UTC)}))
	check(Must(ts.Gt("2020-06-01")), false, true)
	_, err = ts.Gt("garbage")
	fails(t, err, ErrParse)
	check(Must(Must(NewSeries("t", []bool{true, false})).Eq(true)), true, false)
	check(Must(i.Between(1, 2)), true, true, false)
	_, err = s.Between(1, "z")
	fails(t, err, ErrDType)
	_, err = s.Between("a", 1)
	fails(t, err, ErrDType)
	b := Must(NewSeries("b", []bool{true, false}))
	c := Must(NewSeries("c", []bool{true, true}))
	check(Must(b.Xor(c)), false, true)
	check(Must(b.Or(c)), true, true)
	check(Must(b.Not()), false, true)
	_, err = s.Not()
	fails(t, err, ErrDType)
	_, err = b.And(s)
	fails(t, err, ErrDType)
	_, err = s.And(b)
	fails(t, err, ErrDType)
	_, err = b.And(Must(NewSeriesIndexed("x", []bool{true}, Must(NewIndex("", []string{"a"})))))
	fails(t, err, ErrDType)
	check(Must(s.IsIn()), false, false)
	check(Must(i.IsIn(2, 9)), false, true, false)
	_, err = s.IsIn(struct{}{})
	fails(t, err, ErrArgument)
	_, err = s.IsIn(1)
	fails(t, err, ErrDType)
}

func TestReductions(t *testing.T) {
	f := Must(NewSeriesMasked("f", []float64{1, 2, math.NaN(), 4}, []bool{true, true, true, false}))
	if f.Sum() != 3 || f.Mean() != 1.5 || f.Count() != 2 || f.Min() != 1.0 || f.Max() != 2.0 || f.Prod() != 2 {
		t.Error("float reductions")
	}
	if math.Abs(f.Std()-math.Sqrt(0.5)) > 1e-15 || f.Median() != 1.5 || f.VarDDOF(0) != 0.25 {
		t.Error("float moments")
	}
	u := Must(NewSeries("u", []uint8{200, 100}))
	if u.SumValue() != uint64(300) || u.Max() != uint8(200) {
		t.Error("uint sums as uint64", u.SumValue())
	}
	b := Must(NewSeries("b", []bool{true, true, false}))
	if b.SumValue() != int64(2) || b.Mean() != 2.0/3 {
		t.Error("bool sum")
	}
	z := Must(NewSeriesMasked("z", []complex128{1 + 1i, 2, 5}, []bool{true, true, false}))
	if z.SumValue() != 3+1i || z.Min() != nil || !math.IsNaN(z.Sum()) {
		t.Error("complex sum")
	}
	s := Must(NewSeries("s", []string{"b", "a", "b"}))
	if s.SumValue() != nil || !math.IsNaN(s.Mean()) || s.Min() != "a" || s.Max() != "b" {
		t.Error("string reductions")
	}
	empty := Must(NewSeries("e", []float64{}))
	if empty.Min() != nil || !math.IsNaN(empty.Median()) || !math.IsNaN(empty.Var()) {
		t.Error("empty")
	}
	allNA := Must(NewSeries("n", []any{nil, nil}))
	if allNA.Max() != nil {
		t.Error("all missing")
	}
	i8 := Must(NewSeriesMasked("i8", []int8{1, 2, 3}, []bool{true, false, true}))
	if i8.SumValue() != int64(4) || i8.Mean() != 2 || i8.Min() != int8(1) {
		t.Error("int8")
	}
	ts := Must(NewSeries("t", []time.Time{time.Unix(5, 0), time.Unix(1, 0)}))
	if ts.Min() != time.Unix(1, 0).UTC() || !math.IsNaN(ts.Mean()) {
		t.Error("datetime min")
	}
	for _, name := range []string{"sum", "mean", "std", "var", "min", "max", "count", "size", "median", "prod", "nunique", "first", "last"} {
		if _, err := s.Agg(name); err != nil {
			t.Error(name, err)
		}
	}
	if v, _ := allNA.Agg("first"); v != nil {
		t.Error("first of nothing")
	}
	if v, _ := s.Agg("last"); v != "b" {
		t.Error("last")
	}
	_, err := s.Agg("nope")
	fails(t, err, ErrArgument)
	if vc := s.ValueCounts(); vc.c.value(0) != int64(2) || vc.index.Label(0) != "b" {
		t.Error("ValueCounts")
	}
	df := Must(New(Col("i", []int{3, 1}), Col("u", []uint64{1, 2}), Col("b", []bool{true, false}), Col("s", []string{"x", "y"})))
	if m := df.Max(); m.DType() != Int64 || m.c.value(2) != int64(1) {
		t.Error("frame max ints", m.Items())
	}
	if m := df.Sum(); m.DType() != Float64 {
		t.Error("frame sum with uint64 is float64", m.DType())
	}
	if m := df.Mean(); m.c.value(0) != 2.0 {
		t.Error("frame mean")
	}
	if df.Std().Len() != 3 || df.Var().Len() != 3 || df.Min().Len() != 3 || df.Median().Len() != 3 {
		t.Error("frame reductions")
	}
	txt := Must(New(Col("s", []any{"x", "x", nil}), Col("e", []any{nil, nil, nil})))
	txt.cols[1] = newCol(String, []string{"", "", ""}, newBitmap(3))
	d := txt.Describe()
	if d.cols[0].value(2) != "x" || d.cols[0].value(3) != "2" || d.cols[1].value(2) != nil {
		t.Error("describe text", d)
	}
	nan := Must(New(Col("n", []float64{math.NaN()})))
	if d := nan.Describe(); d.cols[0].value(0) != 0.0 {
		t.Error("describe empty column")
	}
}

func TestWindows(t *testing.T) {
	s := Must(NewSeries("s", []string{"a"}))
	for _, f := range []func() (*Series, error){func() (*Series, error) { return s.Diff(1) },
		func() (*Series, error) { return s.PctChange(1) }, s.CumSum} {
		_, err := f()
		fails(t, err, ErrDType)
	}
	_, err := s.Rolling(2)
	fails(t, err, ErrDType)
	n := Must(NewSeries("n", []int{1, 2}))
	_, err = n.RollingMin(2, 3)
	fails(t, err, ErrArgument)
	df := Must(New(Col("n", []int{1, 2, 3})))
	if sh := df.Shift(-1); sh.cols[0].value(2) != nil {
		t.Error("frame shift")
	}
}

func TestSorting(t *testing.T) {
	df := sample(t)
	_, err := df.SortValues(nil)
	fails(t, err, ErrArgument)
	_, err = df.SortValues([]string{"zz"})
	fails(t, err, ErrColumn)
	_, err = df.SortValues([]string{"a", "c"}, true, false, true)
	fails(t, err, ErrLength)
	s := Must(df.Column("a"))
	if v := s.SortValues(false); v.c.value(0) != 3.0 || v.c.value(3) != nil && !v.c.isNA(3) {
		t.Error("series sort desc")
	}
	if v := s.SortIndex(false); v.index.Label(0) != int64(3) {
		t.Error("series sort index")
	}
	// Radix paths: large columns of each radix dtype, with missing values,
	// against a comparison sort.
	rng := rand.New(rand.NewPCG(1, 2))
	n := 70_000
	fl := make([]float64, n)
	in := make([]int64, n)
	i32 := make([]int32, n)
	valid := make([]bool, n)
	for k := range n {
		fl[k] = math.Round(rng.NormFloat64()*100) / 10
		if k%97 == 0 {
			fl[k] = 0
			if k%2 == 0 {
				fl[k] = math.Copysign(0, -1)
			}
		}
		in[k] = rng.Int64N(1000) - 500
		i32[k] = int32(rng.IntN(50))
		valid[k] = k%13 != 0
	}
	big := Must(New(Col("f", Must(NewSeriesMasked("f", fl, valid))), Col("i", in), Col("j", i32)))
	for _, by := range [][]string{{"f"}, {"i"}, {"j", "f"}, {"j", "i"}} {
		for _, asc := range []bool{true, false} {
			got := Must(big.SortValues(by, asc))
			keys := make([]column, len(by))
			for k, name := range by {
				keys[k] = big.cols[Must(big.pos(name))]
			}
			want := make([]int, n)
			for k := range want {
				want[k] = k
			}
			slices.SortStableFunc(want, func(a, b int) int {
				for _, c := range keys {
					na, nb := c.isNA(a), c.isNA(b)
					switch {
					case na && nb:
						continue
					case na:
						return 1
					case nb:
						return -1
					}
					r := c.compare(a, b)
					if !asc {
						r = -r
					}
					if r != 0 {
						return r
					}
				}
				return 0
			})
			for k := range want {
				if l, _ := toInt(got.index.Label(k)); int(l) != want[k] {
					t.Fatalf("sort %v asc=%v differs at %d: %v vs %d", by, asc, k, got.index.Label(k), want[k])
				}
			}
		}
	}
	// A column with one value only skips every radix pass.
	same := Must(New(Col("x", make([]int64, 1000))))
	if v := Must(same.SortValues([]string{"x"})); v.index.Label(999) != int64(999) {
		t.Error("constant column")
	}
}

func TestParallelSmall(t *testing.T) {
	calls := 0
	parallel(10, func(lo, hi int) { calls += hi - lo })
	eachColumn(10, 1, func(int) { calls++ })
	if calls != 11 {
		t.Error(calls)
	}
}
