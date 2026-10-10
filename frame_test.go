package dataframe

import (
	"math"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/go-ndarray/ndarray"
)

func sample(t *testing.T) *DataFrame {
	t.Helper()
	return Must(New(
		Col("a", []float64{3, 1, math.NaN(), 2}),
		Col("b", []string{"x", "y", "z", "w"}),
		Col("c", []int{5, 3, 1, 4}),
	))
}

func TestSeriesBasics(t *testing.T) {
	_, err := NewSeries("x", 3)
	fails(t, err, ErrArgument)
	_, err = NewSeriesIndexed("x", 3, NewRangeIndex(1))
	fails(t, err, ErrArgument)
	_, err = NewSeriesIndexed("x", []int{1, 2}, NewRangeIndex(1))
	fails(t, err, ErrLength)
	ix := Must(NewIndex("k", []string{"p", "q"}))
	s := Must(NewSeriesIndexed("v", []int{10, 20}, ix))
	if s.Name() != "v" || s.Rename("w").Name() != "w" || s.Index() != ix || s.Len() != 2 || s.DType() != Int64 {
		t.Error("accessors")
	}
	if v := Must(s.Get("q")); v != int64(20) {
		t.Error("Get", v)
	}
	_, err = s.Get("zz")
	fails(t, err, ErrLabel)
	if v := Must(s.At(-1)); v != int64(20) {
		t.Error("At -1")
	}
	_, err = s.At(5)
	fails(t, err, ErrIndex)
	_, err = s.SetIndex(NewRangeIndex(3))
	fails(t, err, ErrLength)
	if r := Must(s.SetIndex(NewRangeIndex(2))); !r.Index().IsRange() {
		t.Error("SetIndex")
	}
	if !s.ResetIndex().Index().IsRange() {
		t.Error("ResetIndex")
	}
	if a := Must(s.Values()); a.DType() != ndarray.Int64 {
		t.Error("Values")
	}
	str := Must(NewSeries("s", []string{"a"}))
	_, err = str.Values()
	fails(t, err, ErrDType)
	_, err = str.Floats()
	fails(t, err, ErrDType)
	if f := Must(s.Floats()); f[1] != 20 {
		t.Error("Floats")
	}
	m := Must(NewSeriesMasked("m", []float64{1, 2}, []bool{false, true}))
	if !reflect.DeepEqual(m.Strings(), []string{"", "2.0"}) || !reflect.DeepEqual(m.Items(), []any{nil, 2.0}) {
		t.Error("Strings/Items", m.Strings(), m.Items())
	}
	if !reflect.DeepEqual(m.Valid(), []bool{false, true}) {
		t.Error("Valid")
	}
	if Must(NewSeriesMasked("m", []float64{1}, []bool{true})).c.validity() != nil {
		t.Error("all-valid mask kept a bitmap")
	}
	_, err = NewSeriesMasked("m", []float64{1}, []bool{true, false})
	fails(t, err, ErrLength)
	_, err = NewSeriesMasked("m", 1, nil)
	fails(t, err, ErrArgument)
	if h := s.Head(1); h.Len() != 1 || h.Tail(-1).Len() != 0 {
		t.Error("Head/Tail")
	}
	if s.Tail(1).c.value(0) != int64(20) || s.Head(-1).Len() != 1 {
		t.Error("Tail")
	}
	if cp := s.Copy(); cp.c == s.c || cp.c.value(1) != int64(20) {
		t.Error("Copy")
	}
	if na := m.IsNA(); na.c.value(0) != true {
		t.Error("IsNA")
	}
	if nn := m.NotNA(); nn.c.value(0) != false {
		t.Error("NotNA")
	}
	if f := s.ToFrame(); f.names[0] != "v" {
		t.Error("ToFrame")
	}
	_, err = str.bools()
	fails(t, err, ErrDType)
}

func TestFrameConstruction(t *testing.T) {
	_, err := New(Col("a", 3))
	fails(t, err, ErrArgument)
	_, err = New(Col("a", []int{1}), Col("b", []int{1, 2}))
	fails(t, err, ErrLength)
	ix := Must(NewIndex("k", []string{"p", "q"}))
	s := Must(NewSeriesIndexed("v", []int{1, 2}, ix))
	df := Must(New(Col("v", s), Col("w", []float64{1, 2})))
	if df.Index() != ix {
		t.Error("series index adopted")
	}
	empty := Must(New())
	if r, c := empty.Shape(); r != 0 || c != 0 {
		t.Error("empty")
	}
	if Must(FromSeries()).Len() != 0 {
		t.Error("FromSeries()")
	}
	s2 := Must(NewSeriesIndexed("u", []int{7, 8}, Must(NewIndex("k", []string{"q", "r"}))))
	u := Must(FromSeries(s, s2))
	if u.Len() != 3 || u.cols[0].isNA(2) != true || u.cols[1].isNA(0) != true {
		t.Error("FromSeries union", u)
	}
	same := Must(FromSeries(s, s.Rename("z")))
	if same.Index() != ix {
		t.Error("FromSeries same index")
	}
	_, err = FromSeries(s, Must(NewSeriesIndexed("x", []int{1}, Must(NewIndex("", []int{1})))))
	fails(t, err, ErrDType)
	dupIx := Must(NewIndex("k", []string{"p", "p"}))
	_, err = FromSeries(Must(NewSeriesIndexed("x", []int{1, 2}, dupIx)), s2)
	fails(t, err, ErrLabel)
	two := Must(NewIndex("a", []int{1}))
	two = labelIndex([]string{"a", "b"}, []column{two.levels[0], two.levels[0]})
	_, err = FromSeries(Must(NewSeriesIndexed("x", []int{1}, two)), s2)
	fails(t, err, ErrArgument)

	m, _ := ndarray.FromData([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	fa := Must(FromArray(m))
	if !reflect.DeepEqual(fa.Columns(), []string{"0", "1", "2"}) || fa.cols[2].value(1) != 6.0 {
		t.Error("FromArray", fa)
	}
	fb := Must(FromArray(m, "x", "y", "z"))
	if fb.names[1] != "y" {
		t.Error("FromArray names")
	}
	_, err = FromArray(m, "x")
	fails(t, err, ErrLength)
	v, _ := ndarray.FromData([]float64{1}, 1)
	_, err = FromArray(v)
	fails(t, err, ErrArgument)
}

func TestFrameAccessors(t *testing.T) {
	df := sample(t)
	if !reflect.DeepEqual(df.DTypes(), []DType{Float64, String, Int64}) {
		t.Error("DTypes")
	}
	if d := df.DTypesSeries(); d.c.value(1) != "string" || d.index.Label(2) != "c" {
		t.Error("DTypesSeries")
	}
	_, err := df.Column("zz")
	fails(t, err, ErrColumn)
	if c := Must(df.ColumnAt(-1)); c.Name() != "c" {
		t.Error("ColumnAt")
	}
	_, err = df.ColumnAt(3)
	fails(t, err, ErrIndex)
	_, err = df.Select("a", "zz")
	fails(t, err, ErrColumn)
	_, err = df.Drop("zz")
	fails(t, err, ErrColumn)
	if d := Must(df.Drop("b")); !reflect.DeepEqual(d.Columns(), []string{"a", "c"}) {
		t.Error("Drop")
	}
	w := Must(df.WithColumn("d", 1.5))
	if w.cols[3].value(2) != 1.5 || len(df.cols) != 3 {
		t.Error("WithColumn scalar")
	}
	w = Must(df.WithColumn("a", []int{1, 2, 3, 4}))
	if w.cols[0].dtype() != Int64 {
		t.Error("WithColumn replace")
	}
	_, err = df.WithColumn("a", []int{1})
	fails(t, err, ErrLength)
	_, err = df.WithColumn("a", struct{}{})
	fails(t, err, ErrArgument)
	sorted := Must(df.SortValues([]string{"c"}))
	c := Must(sorted.Column("c"))
	back := Must(df.WithColumn("c2", c))
	if back.cols[3].value(0) != int64(5) {
		t.Error("WithColumn aligns a series")
	}
	same := Must(df.WithColumn("c3", Must(df.Column("c"))))
	if same.cols[3] != df.cols[2] {
		t.Error("WithColumn same index shares")
	}
	dup := Must(NewSeriesIndexed("d", []int{1, 2}, Must(NewIndex("", []int{0, 0}))))
	_, err = df.WithColumn("d", dup)
	fails(t, err, ErrLabel)
	r := df.Rename(map[string]string{"a": "A", "nope": "x"})
	if r.names[0] != "A" || df.names[0] != "a" {
		t.Error("Rename")
	}
	sc := Must(df.SetColumns("p", "q", "r"))
	if sc.names[2] != "r" {
		t.Error("SetColumns")
	}
	_, err = df.SetColumns("p")
	fails(t, err, ErrLength)
	if len(df.Series()) != 3 {
		t.Error("Series")
	}
	cp := df.Copy()
	if cp.cols[0] == df.cols[0] || cp.cols[0].value(0) != 3.0 {
		t.Error("Copy")
	}
}

func TestFrameValues(t *testing.T) {
	df := sample(t)
	_, err := df.Values()
	fails(t, err, ErrDType)
	num := Must(df.Select("a", "c"))
	v := Must(num.Values())
	if !reflect.DeepEqual(v.Shape(), []int{4, 2}) || v.At(1, 1) != 3 || !math.IsNaN(v.At(2, 0)) {
		t.Error("Values", v)
	}
	ints := Must(New(Col("i", []int32{1, 2}), Col("j", []any{1, nil})))
	iv := Must(ints.Values())
	if iv.DType() != ndarray.Float64 || !math.IsNaN(iv.At(1, 1)) {
		t.Error("int with missing promotes to float", iv.DType())
	}
	z := Must(New(Col("z", []complex128{1, 2})))
	if Must(z.Values()).DType() != ndarray.Complex128 {
		t.Error("complex Values")
	}
	e := Must(New())
	if Must(e.Values()).Size() != 0 {
		t.Error("empty Values")
	}
}

func TestIndexAPI(t *testing.T) {
	r := NewRangeIndex(3)
	if !r.IsRange() || r.NLevels() != 1 || r.Name() != "" || r.DType() != Int64 {
		t.Error("range basics")
	}
	if !reflect.DeepEqual(r.Labels(), []any{int64(0), int64(1), int64(2)}) {
		t.Error("Labels")
	}
	_, err := r.Get(3)
	fails(t, err, ErrLabel)
	_, err = r.Get("x")
	fails(t, err, ErrLabel)
	if Must(r.Values()).Size() != 3 {
		t.Error("range Values")
	}
	_, err = r.Rename("a", "b")
	fails(t, err, ErrLength)
	rn := Must(r.Rename("n"))
	if rn.Name() != "n" || !reflect.DeepEqual(rn.Names(), []string{"n"}) {
		t.Error("Rename")
	}
	_, err = r.Level(1)
	fails(t, err, ErrIndex)
	if lv := Must(r.Level(0)); lv.Len() != 3 {
		t.Error("Level")
	}
	_, err = NewIndex("x", 3)
	fails(t, err, ErrArgument)
	s := Must(NewIndex("s", []string{"a", "b", "a"}))
	if p := Must(s.Get("a")); !reflect.DeepEqual(p, []int{0, 2}) {
		t.Error("Get dup", p)
	}
	_, err = s.Values()
	fails(t, err, ErrDType)
	m := labelIndex([]string{"k", "j"}, []column{newCol(String, []string{"a", "a", "b"}, nil), newCol(Int64, []int64{1, 2, 1}, nil)})
	if p := Must(m.Get([]any{"a", 2})); !reflect.DeepEqual(p, []int{1}) {
		t.Error("multi Get", p)
	}
	if p := Must(m.Get("a")); len(p) != 2 {
		t.Error("multi prefix Get", p)
	}
	_, err = m.Get([]any{"a", 2, 3})
	fails(t, err, ErrLabel)
	_, err = m.Get([]any{"c"})
	fails(t, err, ErrLabel)
	_, err = m.Get([]any{"a", "x"})
	fails(t, err, ErrLabel)
	if !reflect.DeepEqual(m.Label(1), []any{"a", int64(2)}) {
		t.Error("multi Label")
	}
	_, err = m.Values()
	fails(t, err, ErrDType)
	if !strings.HasPrefix(m.String(), "MultiIndex([(a, 1), (a, 2), (b, 1)]") {
		t.Error(m.String())
	}
	if r.String() != "RangeIndex(start=0, stop=3, step=1)" {
		t.Error(r.String())
	}
	if got := s.String(); got != "Index(['a', 'b', 'a'], dtype='string', name='s')" {
		t.Error(got)
	}
	long := Must(NewIndex("", make([]int, 25)))
	if !strings.Contains(long.String(), "..., ") {
		t.Error("long index elided")
	}
	dt := Must(NewIndex("", []time.Time{time.Unix(0, 0)}))
	if !strings.HasPrefix(dt.String(), "DatetimeIndex(['1970-01-01 00:00:00']") {
		t.Error(dt.String())
	}
	if string(r.Display()["text/plain"]) != r.String() {
		t.Error("Display")
	}
	// take and slice of a stepped range, equality.
	st := &Index{n: 4, start: 10, step: 2, names: []string{""}}
	if tk := st.take([]int{1, 3}); !tk.IsRange() || tk.Label(1) != int64(16) {
		t.Error("take as range", tk.Label(1))
	}
	if tk := st.take([]int{3, 1}); tk.IsRange() || tk.Label(0) != int64(16) {
		t.Error("take reversed")
	}
	if tk := st.take([]int{-1, 0}); tk.level(0).isNA(0) != true {
		t.Error("take missing")
	}
	if p := Must(st.Get(14)); p[0] != 2 {
		t.Error("stepped Get")
	}
	_, err = st.Get(13)
	fails(t, err, ErrLabel)
	if sl := st.slice(1, 3); sl.Label(0) != int64(12) {
		t.Error("slice")
	}
	if !st.equal(st.slice(0, 4)) || st.equal(r) || NewRangeIndex(0).equal(&Index{n: 0, start: 5, step: 1, names: []string{""}}) == false {
		t.Error("equal range")
	}
	if s.equal(Must(NewIndex("s", []string{"a", "b", "c"}))) || !s.equal(Must(NewIndex("s", []string{"a", "b", "a"}))) {
		t.Error("equal labels")
	}
	if s.equal(m) {
		t.Error("equal levels")
	}
	if _, ok := asRange([]int{-1}); ok {
		t.Error("asRange negative")
	}
	if _, ok := asRange([]int{2, 2}); ok {
		t.Error("asRange zero step")
	}
	if r, ok := asRange(nil); !ok || r.step != 1 {
		t.Error("asRange empty")
	}
	if r, ok := asRange([]int{4}); !ok || r.start != 4 {
		t.Error("asRange single")
	}
}

func TestSelection(t *testing.T) {
	df := Must(sample(t).SetIndex("b"))
	if g := Must(df.Iloc(All())); g.Len() != 4 {
		t.Error("All")
	}
	if g := Must(df.Iloc(SliceFrom(2))); g.Len() != 2 {
		t.Error("SliceFrom")
	}
	if g := Must(df.Iloc(Slice(3, 1))); g.Len() != 0 {
		t.Error("empty slice")
	}
	_, err := df.Iloc(Pos(9))
	fails(t, err, ErrIndex)
	_, err = df.Iloc(Labels("x"))
	fails(t, err, ErrArgument)
	_, err = df.Iloc(All(), Labels("x"))
	fails(t, err, ErrArgument)
	_, err = df.Iloc(All(), LabelRange("x", "y"))
	fails(t, err, ErrArgument)
	if g := Must(df.Iloc(All(), Pos(1))); g.names[0] != "c" {
		t.Error("column positions")
	}
	_, err = df.Loc(Pos(0))
	fails(t, err, ErrArgument)
	_, err = df.Loc(All(), "zz")
	fails(t, err, ErrColumn)
	if g := Must(df.Loc(Labels("z"), "c")); g.cols[0].value(0) != int64(1) {
		t.Error("Loc cell")
	}
	_, err = df.Loc(Labels("nope"))
	fails(t, err, ErrLabel)
	_, err = df.Loc(LabelRange("a", "b"))
	fails(t, err, ErrLabel)
	if g := Must(df.Loc(LabelRange(nil, "y"))); g.Len() != 2 {
		t.Error("open from")
	}
	if g := Must(df.Loc(LabelRange("y", nil))); g.Len() != 3 {
		t.Error("open to")
	}
	sorted := df.SortIndex(true)
	if g := Must(sorted.Loc(LabelRange("a", "xa"))); g.Len() != 2 {
		t.Error("absent bounds", g.Len())
	}
	_, err = sorted.Loc(LabelRange(1, nil))
	fails(t, err, ErrLabel)
	_, err = sorted.Loc(LabelRange(nil, 1))
	fails(t, err, ErrLabel)
	withNA := labelIndex([]string{""}, []column{newCol(Int64, []int64{1, 2}, b3()[:1])})
	withNA.levels[0] = withNA.levels[0].withValidity(func() bitmap { b := allValid(2); b.clear(0); return b }())
	if withNA.monotonic() {
		t.Error("monotonic with missing")
	}
	mi := Must(sample(t).SetIndex("b", "c"))
	_, err = mi.Loc(LabelRange([]any{"q"}, nil))
	fails(t, err, ErrLabel)
	_, err = df.Filter([]bool{true})
	fails(t, err, ErrLength)
	_, err = df.Filter(3)
	fails(t, err, ErrArgument)
	_, err = df.Filter(Must(NewSeries("m", []int{1, 2, 3, 4})))
	fails(t, err, ErrDType)
	if g := Must(df.Filter([]bool{true, false, true, false})); g.Len() != 2 {
		t.Error("Filter bools")
	}
	short := Must(NewSeries("m", []bool{true}))
	_, err = sample(t).Filter(short)
	fails(t, err, ErrLabel)
	_, err = sample(t).Iloc(All(), Mask(short))
	fails(t, err, ErrLength)
	if g := Must(sample(t).Iloc(All(), Mask([]bool{true, false, true}))); len(g.cols) != 2 {
		t.Error("column mask")
	}
	lm := Must(NewSeriesIndexed("m", []bool{false, true, false, false}, Must(NewIndex("", []string{"x", "z", "y", "w"}))))
	if g := Must(df.Filter(lm)); g.Len() != 1 || g.index.Label(0) != "z" {
		t.Error("aligned mask")
	}
	dm := Must(NewSeriesIndexed("m", []bool{true, true}, Must(NewIndex("", []string{"z", "z"}))))
	_, err = df.Filter(dm)
	fails(t, err, ErrLabel)
	// Series selection.
	s := Must(df.Column("c"))
	if v := Must(s.Iloc(Slice(1, 2))); v.c.value(0) != int64(3) {
		t.Error("Series Iloc")
	}
	_, err = s.Iloc(Labels("x"))
	fails(t, err, ErrArgument)
	if v := Must(s.Loc(Labels("w"))); v.c.value(0) != int64(4) {
		t.Error("Series Loc")
	}
	_, err = s.Loc(Slice(0, 1))
	fails(t, err, ErrArgument)
	_, err = s.Loc(Labels("nope"))
	fails(t, err, ErrLabel)
	if v := Must(s.Filter([]bool{true, true, false, false})); v.Len() != 2 {
		t.Error("Series Filter")
	}
	if v := Must(df.At("y", "c")); v != int64(3) {
		t.Error("At")
	}
	_, err = df.At("y", "zz")
	fails(t, err, ErrColumn)
	if v := Must(df.IAt(0, 0)); v != 3.0 {
		t.Error("IAt")
	}
	_, err = df.IAt(0, 7)
	fails(t, err, ErrIndex)
}

func TestIndexOps(t *testing.T) {
	df := sample(t)
	_, err := df.SetIndex()
	fails(t, err, ErrArgument)
	_, err = df.SetIndex("zz")
	fails(t, err, ErrColumn)
	_, err = df.WithIndex(NewRangeIndex(2))
	fails(t, err, ErrLength)
	w := Must(df.WithIndex(Must(NewIndex("i", []int{9, 8, 7, 6}))))
	if w.index.Label(0) != int64(9) {
		t.Error("WithIndex")
	}
	mi := Must(df.SetIndex("b", "c"))
	unnamed := Must(mi.WithIndex(Must(mi.index.Rename("", ""))))
	if r := unnamed.ResetIndex(false); r.names[0] != "level_0" || r.names[1] != "level_1" {
		t.Error("unnamed multi reset", r.names)
	}
	_, err = df.Reindex(3)
	fails(t, err, ErrArgument)
	ri := Must(df.Reindex(NewRangeIndex(2)))
	if ri.Len() != 2 {
		t.Error("Reindex index")
	}
	_, err = df.Reindex([]string{"a"})
	fails(t, err, ErrDType)
	dup := Must(df.WithIndex(Must(NewIndex("", []int{1, 1, 2, 3}))))
	_, err = dup.Reindex([]int{1})
	fails(t, err, ErrLabel)
	rc := df.ReindexColumns("c", "q")
	if rc.names[1] != "q" || rc.cols[1].naCount() != 4 {
		t.Error("ReindexColumns")
	}
	s := Must(df.Column("c"))
	if r := Must(s.Reindex([]int{3, 9})); r.c.value(0) != int64(4) || r.c.value(1) != nil {
		t.Error("Series Reindex")
	}
	_, err = s.Reindex("x")
	fails(t, err, ErrArgument)
}

func TestMissing(t *testing.T) {
	df := Must(New(Col("p", []any{1.0, nil, nil, 4.0}), Col("q", []any{nil, nil, "c", "d"})))
	if na := df.IsNA(); na.cols[0].value(1) != true {
		t.Error("IsNA")
	}
	if nn := df.NotNA(); nn.cols[1].value(3) != true {
		t.Error("NotNA")
	}
	_, err := df.DropNA("zz")
	fails(t, err, ErrColumn)
	full := Must(New(Col("x", []int{1})))
	if Must(full.DropNA()) != full {
		t.Error("DropNA unchanged")
	}
	if d := df.DropNAColumns(); len(d.cols) != 0 {
		t.Error("DropNAColumns")
	}
	if d := full.DropNAColumns(); len(d.cols) != 1 {
		t.Error("DropNAColumns keep")
	}
	s := Must(df.Column("p"))
	if d := s.DropNA(); d.Len() != 2 {
		t.Error("Series DropNA")
	}
	if d := full.Series()[0].DropNA(); d.Len() != 1 {
		t.Error("Series DropNA unchanged")
	}
	if f := Must(s.FillNA(0)); f.c.value(1) != 0.0 {
		t.Error("Series FillNA")
	}
	_, err = s.FillNA("x")
	fails(t, err, ErrDType)
	f := df.FillNA("z")
	if f.cols[1].value(0) != "z" || f.cols[0].value(1) != nil {
		t.Error("frame FillNA skips mismatched columns")
	}
	_, err = df.FillNAColumns(map[string]any{"zz": 1})
	fails(t, err, ErrColumn)
	_, err = df.FillNAColumns(map[string]any{"p": "x"})
	fails(t, err, ErrDType)
	if ff := s.FFill(); ff.c.value(2) != 1.0 {
		t.Error("Series FFill")
	}
	if bf := s.BFill(); bf.c.value(1) != 4.0 {
		t.Error("Series BFill")
	}
}

func TestAsType(t *testing.T) {
	s := Must(NewSeriesMasked("f", []float64{1.5, math.NaN(), 3}, []bool{true, true, false}))
	i := Must(s.AsType(Int64))
	if i.c.value(0) != int64(1) || i.c.value(1) != nil || i.c.value(2) != nil {
		t.Error("float to int", i.Items())
	}
	_, err := s.AsType(Category + 10)
	fails(t, err, ErrDType)
	str := Must(NewSeries("s", []any{" 12 ", nil, "7"}))
	if v := Must(str.AsType(Int32)); v.DType() != Int32 || v.c.value(0) != int32(12) {
		t.Error("string to int32")
	}
	if v := Must(str.AsType(Float64)); v.c.value(2) != 7.0 {
		t.Error("string to float")
	}
	if v := Must(str.AsType(Uint64)); v.c.value(2) != uint64(7) {
		t.Error("string to uint64")
	}
	big := Must(NewSeries("b", []string{"18446744073709551615"}))
	if v := Must(big.AsType(Uint64)); v.c.value(0) != uint64(math.MaxUint64) {
		t.Error("big uint64", v.Items())
	}
	if v := Must(Must(NewSeries("b", []string{"True", "false"})).AsType(Bool)); v.c.value(1) != false {
		t.Error("string to bool")
	}
	if v := Must(Must(NewSeries("z", []string{"1+2i"})).AsType(Complex128)); v.c.value(0) != 1+2i {
		t.Error("string to complex")
	}
	if v := Must(Must(NewSeries("d", []string{"2020-01-01"})).AsType(Datetime)); v.c.value(0) != time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC) {
		t.Error("string to datetime")
	}
	for dt, bad := range map[DType]string{Int64: "1.5", Bool: "yes", Complex128: "z", Float64: "x", Datetime: "nope"} {
		_, err := Must(NewSeries("x", []string{bad})).AsType(dt)
		fails(t, err, ErrParse)
	}
	dts := Must(NewSeries("t", []time.Time{time.Unix(1, 0)}))
	if v := Must(dts.AsType(Int64)); v.c.value(0) != int64(1e9) {
		t.Error("datetime to int")
	}
	if v := Must(Must(NewSeries("n", []int64{2e9})).AsType(Datetime)); v.c.value(0) != time.Unix(2, 0).UTC() {
		t.Error("int to datetime")
	}
	cat := Must(Must(NewSeries("c", []any{"b", nil, "a"})).AsType(Category))
	if cat.c.(*col[int32]).levels[0] != "a" || cat.c.value(1) != nil {
		t.Error("to category")
	}
	if v := Must(cat.AsType(String)); v.c.value(0) != "b" {
		t.Error("category to string")
	}
	if v := Must(Must(Must(NewSeries("c", []string{"1", "2"})).AsType(Category)).AsType(Int64)); v.c.value(1) != int64(2) {
		t.Error("category to int")
	}
	if v := Must(cat.AsType(Category)); v.c.value(2) != "a" {
		t.Error("category to category")
	}
	df := Must(New(Col("a", []int{1}), Col("b", []string{"x"})))
	_, err = df.AsType(map[string]DType{"zz": Int64})
	fails(t, err, ErrColumn)
	_, err = df.AsType(map[string]DType{"b": Int64})
	fails(t, err, ErrParse)
	if a := Must(df.AsTypeAll(String)); a.cols[0].value(0) != "1" {
		t.Error("AsTypeAll")
	}
	f32 := Must(NewSeries("f", []float32{1.25}))
	if r := f32.Round(1); r.c.value(0) != float32(1.2) {
		t.Error("float32 round", r.c.value(0))
	}
	if r := Must(NewSeries("i", []int{15})).Round(-1); r.c.value(0) != int64(15) {
		t.Error("int round unchanged")
	}
	if r := Must(NewSeries("f", []float64{2.5, 1250})).Round(0); r.c.value(0) != 2.0 {
		t.Error("round half even")
	}
}
