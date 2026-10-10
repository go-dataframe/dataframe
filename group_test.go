package dataframe

import (
	"math"
	"reflect"
	"strings"
	"testing"
	"time"
)

func groupSample(t *testing.T) *DataFrame {
	t.Helper()
	return Must(New(
		Col("k", []any{"b", "a", "b", nil, "a"}),
		Col("x", []any{1.0, 2.0, nil, 4.0, 6.0}),
		Col("i", []int8{1, 2, 3, 4, 5}),
		Col("f", []float32{1, 2, 3, 4, 5}),
		Col("u", []bool{true, false, true, true, true}),
		Col("s", []string{"p", "q", "r", "s", "q"}),
		Col("z", []complex128{1, 2, 3, 4, 5}),
	))
}

func TestGroupByAll(t *testing.T) {
	df := groupSample(t)
	_, err := df.GroupBy()
	fails(t, err, ErrArgument)
	_, err = df.GroupBy("zz")
	fails(t, err, ErrColumn)
	g := Must(df.GroupBy("k"))
	if g.NGroups() != 2 || !strings.Contains(g.String(), "groups=2") {
		t.Error("NGroups/String")
	}
	sum := Must(g.Sum())
	if !reflect.DeepEqual(sum.Columns(), []string{"x", "i", "f", "u"}) {
		t.Error("sum columns", sum.Columns())
	}
	if sum.cols[1].value(0) != int64(7) || sum.cols[2].value(1) != 4.0 || sum.cols[3].value(1) != int64(2) {
		t.Error("sum values", sum)
	}
	for name, f := range map[string]func() (*DataFrame, error){"mean": g.Mean, "std": g.Std, "var": g.Var,
		"median": g.Median, "first": g.First, "last": g.Last, "count": g.Count, "nunique": g.NUnique} {
		if r, err := f(); err != nil || r.Len() != 2 {
			t.Error(name, err)
		}
	}
	_, err = g.Min()
	fails(t, err, ErrDType)
	_, err = g.Max()
	fails(t, err, ErrDType)
	sel := Must(g.Select("s", "x"))
	mn := Must(sel.Min())
	if mn.cols[0].value(0) != "q" || mn.cols[1].value(1) != 1.0 {
		t.Error("Select+Min", mn)
	}
	mx := Must(sel.Max())
	if mx.cols[0].value(1) != "r" || mx.cols[1].value(0) != 6.0 {
		t.Error("Max", mx)
	}
	_, err = g.Select("zz")
	fails(t, err, ErrColumn)
	_, err = g.Agg(Agg("zz", "sum"))
	fails(t, err, ErrColumn)
	_, err = g.Agg(Agg("s", "sum"))
	fails(t, err, ErrDType)
	_, err = g.Agg(Agg("s", "mean"))
	fails(t, err, ErrDType)
	_, err = g.Agg(Agg("s", "median"))
	fails(t, err, ErrDType)
	_, err = g.Agg(Agg("x", "nope"))
	fails(t, err, ErrArgument)
	a := Must(g.Agg(Agg("x", "size", "prod", "median"), Agg("s", "nunique")))
	if !reflect.DeepEqual(a.Columns(), []string{"x_size", "x_prod", "x_median", "s"}) || a.cols[1].value(0) != 12.0 {
		t.Error("Agg", a)
	}
	_, err = g.Transform("zz", "sum")
	fails(t, err, ErrColumn)
	_, err = g.Transform("s", "sum")
	fails(t, err, ErrDType)
	tr := Must(g.Transform("x", "max"))
	if tr.c.value(3) != nil || tr.c.value(0) != 1.0 {
		t.Error("Transform", tr.Items())
	}
	keys, frames := g.Groups()
	if keys[0] != "a" || frames[1].Len() != 2 {
		t.Error("Groups")
	}
	ap := Must(g.Apply(func(key any, f *DataFrame) (*DataFrame, error) { return f.Head(1), nil }))
	if ap.Len() != 2 {
		t.Error("Apply")
	}
	_, err = g.Apply(func(key any, f *DataFrame) (*DataFrame, error) { return f.Select("nope") })
	fails(t, err, ErrColumn)
	if sz := g.Size(); sz.c.value(0) != int64(2) {
		t.Error("Size")
	}
	// Multiple keys, one of them sparse integers (hashed, not tabled).
	big := Must(New(Col("a", []int64{1 << 40, 5, 1 << 40}), Col("b", []string{"x", "y", "x"}), Col("v", []float64{1, 2, 3})))
	m := Must(Must(big.GroupBy("a", "b")).Sum())
	if m.Len() != 2 || m.index.NLevels() != 2 || m.cols[0].value(1) != 4.0 {
		t.Error("multi-key", m)
	}
	// Datetime and uint keys, a frame with no value columns.
	dk := Must(New(Col("t", []time.Time{time.Unix(5, 0), time.Unix(1, 0), time.Unix(5, 0)})))
	if c := Must(Must(dk.GroupBy("t")).Count()); c.Len() != 2 || len(c.cols) != 0 {
		t.Error("datetime key")
	}
	// A group whose float values are all missing.
	nan := Must(New(Col("k", []int{1, 1, 2}), Col("v", []float64{math.NaN(), math.NaN(), 1})))
	ng := Must(nan.GroupBy("k"))
	mm := Must(ng.Mean())
	if !mm.cols[0].isNA(0) {
		t.Error("mean of nothing")
	}
	if md := Must(ng.Median()); !md.cols[0].isNA(0) {
		t.Error("median of nothing")
	}
}

func TestConcat(t *testing.T) {
	_, err := Concat()
	fails(t, err, ErrArgument)
	a := Must(New(Col("x", []int{1}), Col("s", []string{"p"})))
	b := Must(New(Col("x", []float64{2.5})))
	c := Must(Concat(a, b))
	if c.cols[0].dtype() != Float64 || c.cols[1].value(1) != nil {
		t.Error("concat promotes and pads", c)
	}
	bad := Must(New(Col("x", []string{"q"})))
	_, err = Concat(a, bad)
	fails(t, err, ErrDType)
	strIx := Must(b.WithIndex(Must(NewIndex("", []string{"r"}))))
	_, err = Concat(a, strIx)
	fails(t, err, ErrDType)
	multi := Must(a.SetIndex("s"))
	multi = Must(multi.WithIndex(labelIndex([]string{"p", "q"}, []column{multi.index.levels[0], multi.index.levels[0]})))
	_, err = Concat(Must(a.SetIndex("s")), multi)
	fails(t, err, ErrArgument)
	// Side by side with equal indexes, and misaligned ones that fail.
	same := Must(ConcatWith(ConcatOptions{Axis: 1}, a, b))
	if len(same.cols) != 3 {
		t.Error("axis 1 same index")
	}
	_, err = ConcatWith(ConcatOptions{Axis: 1}, a, strIx)
	fails(t, err, ErrDType)
	dup := Must(b.WithIndex(Must(NewIndex("", []int{0}))))
	dup2 := Must(New(Col("y", []int{1, 2})))
	dup2 = Must(dup2.WithIndex(Must(NewIndex("", []int{0, 0}))))
	_, err = ConcatWith(ConcatOptions{Axis: 1}, dup2, Must(dup.WithIndex(Must(NewIndex("", []int{5})))))
	fails(t, err, ErrLabel)
	withNA := Must(New(Col("y", []int{1, 2})))
	withNA = Must(withNA.WithIndex(labelIndex([]string{""}, []column{newCol(Int64, []int64{0, 9}, func() bitmap { b := allValid(2); b.clear(1); return b }())})))
	if r := Must(ConcatWith(ConcatOptions{Axis: 1}, withNA, b)); r.Len() != 1 {
		t.Error("missing labels dropped from the union", r.Len())
	}
	s1 := Must(NewSeries("v", []int{1}))
	s2 := Must(NewSeries("w", []int{2}))
	if s := Must(ConcatSeries(s1, s2)); s.Len() != 2 || s.Name() != "v" {
		t.Error("ConcatSeries")
	}
	_, err = ConcatSeries(s1, Must(NewSeries("w", []string{"x"})))
	fails(t, err, ErrDType)
}

func TestMergeErrors(t *testing.T) {
	l := Must(New(Col("k", []int{1, 2}), Col("v", []float64{1, 2})))
	r := Must(New(Col("j", []int{1}), Col("w", []float64{3})))
	_, err := Merge(l, r, MergeOptions{})
	fails(t, err, ErrArgument)
	_, err = Merge(l, r, MergeOptions{LeftOn: []string{"k"}})
	fails(t, err, ErrLength)
	_, err = Merge(l, r, MergeOptions{On: []string{"zz"}})
	fails(t, err, ErrColumn)
	_, err = Merge(l, r, MergeOptions{LeftOn: []string{"k"}, RightOn: []string{"zz"}})
	fails(t, err, ErrColumn)
	_, err = Merge(l, r, MergeOptions{LeftOn: []string{"k"}, RightOn: []string{"j"}, How: "sideways"})
	fails(t, err, ErrArgument)
	rs := Must(New(Col("k", []string{"1"})))
	_, err = Merge(l, rs, MergeOptions{On: []string{"k"}})
	fails(t, err, ErrDType)
	// Default keys, suffixes, the method form.
	r2 := Must(New(Col("k", []int{2, 2}), Col("v", []float64{7, 8})))
	m := Must(l.Merge(Must(r2.Drop("v")), MergeOptions{}))
	if m.Len() != 2 {
		t.Error("common keys")
	}
	m = Must(Merge(l, r2, MergeOptions{On: []string{"k"}, Suffixes: [2]string{"_l", "_r"}}))
	if !reflect.DeepEqual(m.Columns(), []string{"k", "v_l", "v_r"}) {
		t.Error("suffixes", m.Columns())
	}
	// A right join whose key is a float on one side.
	rf := Must(New(Col("k", []float64{2, 9})))
	m = Must(Merge(l, rf, MergeOptions{On: []string{"k"}, How: "right"}))
	if m.cols[0].dtype() != Float64 || m.cols[0].value(1) != 9.0 {
		t.Error("right join promotes keys", m)
	}
	// Join errors.
	_, err = l.Join(l, "", "", "")
	fails(t, err, ErrArgument)
	_, err = l.Join(Must(r.WithIndex(Must(NewIndex("", []string{"a"})))), "left", "", "")
	fails(t, err, ErrDType)
	if j := Must(l.Join(l, "", "_a", "_b")); !reflect.DeepEqual(j.Columns(), []string{"k_a", "v_a", "k_b", "v_b"}) {
		t.Error("join suffixes", j.Columns())
	}
}

func TestReshapeErrors(t *testing.T) {
	_, err := Must(New()).Stack()
	fails(t, err, ErrArgument)
	if tr := Must(New()).T(); tr.Len() != 0 {
		t.Error("T of empty")
	}
	s := Must(NewSeries("s", []int{1}))
	_, err = s.Unstack()
	fails(t, err, ErrArgument)
	long := Must(New(Col("r", []string{"a", "a"}), Col("c", []string{"x", "x"}), Col("v", []int{1, 2})))
	_, err = long.Pivot("r", "c", "v")
	fails(t, err, ErrArgument)
	_, err = long.Pivot("zz", "c", "v")
	fails(t, err, ErrColumn)
	_, err = long.Pivot("r", "c", "zz")
	fails(t, err, ErrColumn)
	_, err = long.PivotTable("zz", "c", "v", "")
	fails(t, err, ErrColumn)
	_, err = long.PivotTable("r", "c", "zz", "sum")
	fails(t, err, ErrColumn)
	if p := Must(long.PivotTable("r", "c", "v", "")); p.cols[0].value(0) != 1.5 {
		t.Error("pivot table default mean")
	}
	wide := Must(Must(New(Col("r", []string{"a", "b"}), Col("c", []string{"x", "y"}), Col("v", []int{1, 2}), Col("w", []float64{3, 4}))).SetIndex("r", "c"))
	u := Must(wide.Unstack())
	if !reflect.DeepEqual(u.Columns(), []string{"v_x", "v_y", "w_x", "w_y"}) {
		t.Error("frame unstack", u.Columns())
	}
	_, err = Must(New(Col("v", []int{1}))).Unstack()
	fails(t, err, ErrArgument)
	_, err = Must(New()).Unstack()
	fails(t, err, ErrArgument)
	withNA := Must(New(Col("r", []any{"a", nil}), Col("c", []string{"x", "y"}), Col("v", []int{1, 2})))
	if p := Must(withNA.Pivot("r", "c", "v")); p.Len() != 1 {
		t.Error("missing keys skipped")
	}
	_, err = long.Melt([]string{"zz"}, nil)
	fails(t, err, ErrColumn)
	_, err = long.Melt(nil, []string{"zz"})
	fails(t, err, ErrColumn)
	if m := Must(long.Melt([]string{"r", "c", "v"}, nil)); m.Len() != 0 {
		t.Error("melt with no value columns")
	}
	if m := Must(long.Melt([]string{"r"}, nil)); m.Len() != 4 || m.cols[2].dtype() != String {
		t.Error("melt mixed values to strings", m)
	}
	mixed := Must(New(Col("t", []time.Time{time.Unix(0, 0)}), Col("s", []string{"x"})))
	if tr := mixed.T(); tr.cols[0].dtype() != String {
		t.Error("T mixed")
	}
	cats := Must(Must(New(Col("c", []string{"x"}))).AsType(map[string]DType{"c": Category}))
	if tr := cats.T(); tr.cols[0].dtype() != String {
		t.Error("T categories")
	}
}
