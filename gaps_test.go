package dataframe

import (
	"math"
	"strings"
	"testing"
	"time"
)

// Edge cases that the feature tests do not reach on their own.

func TestEdgeAlignment(t *testing.T) {
	if dt, err := commonType(String, Category); err != nil || dt != String {
		t.Error("string/category")
	}
	cat := Must(Must(New(Col("k", []string{"a", "b"}), Col("v", []int{1, 2}))).AsType(map[string]DType{"k": Category}))
	str := Must(New(Col("k", []string{"b"}), Col("w", []int{9})))
	if m := Must(Merge(cat, str, MergeOptions{On: []string{"k"}})); m.Len() != 1 || m.cols[0].value(0) != "b" {
		t.Error("merge category with string", m)
	}
	// Three keys of many distinct values overflow the tuple-code space and
	// are compacted between keys.
	n := 50_000
	a := make([]int64, n)
	b := make([]int64, n)
	c := make([]int64, n)
	for i := range n {
		a[i], b[i], c[i] = int64(i)*1_000_003, int64(n-i)*999_983, int64(i%7)
	}
	df := Must(New(Col("a", a), Col("b", b), Col("c", c)))
	if g := Must(df.GroupBy("a", "b", "c")); g.NGroups() != n {
		t.Error("compacted codes", g.NGroups())
	}
	keys := []column{newCol(Int64, []int64{1, 1}, b2()), newCol(Int64, []int64{3, 3}, nil)}
	if compareRows(keys, 0, 1) != 0 {
		t.Error("equal rows with missing keys")
	}
	dup := Must(NewSeriesIndexed("d", []int{1, 2}, Must(NewIndex("", []string{"x", "x"}))))
	other := Must(NewSeriesIndexed("o", []int{1}, Must(NewIndex("", []string{"y"}))))
	_, err := dup.Add(other)
	fails(t, err, ErrLabel)
	_, err = other.Add(dup)
	fails(t, err, ErrLabel)
	tcol := Must(NewSeries("t", []time.Time{time.Unix(0, 0)}))
	if _, ok := floatsOf(tcol.c); ok {
		t.Error("floatsOf datetime")
	}
	_, err = Must(NewSeries("s", []string{"x"})).Gt(time.Unix(0, 0))
	fails(t, err, ErrParse)
	_, err = Must(NewSeries("i", []int64{1})).Gt(time.Unix(0, 0))
	fails(t, err, ErrDType)
	if sumFloat(make([]float64, 300)) != 0 || Must(NewSeries("x", make([]float64, 300))).Sum() != 0 {
		t.Error("pairwise sum")
	}
	if Must(NewSeries("x", []any{nil, nil})).c.(*col[float64]) == nil {
		t.Error()
	}
	allNA := Must(NewSeriesMasked("i", []int{1, 2}, []bool{false, false}))
	if allNA.Min() != nil {
		t.Error("min of all-missing ints")
	}
	if _, err := parseTime("2024-01-xx", ""); err == nil {
		t.Error("bad day")
	}
	days := Must(New(Col("d", []time.Time{time.Unix(0, 0), time.Unix(86400, 0)})))
	if s := days.String(); !strings.Contains(s, "1970-01-02") || strings.Contains(s, "00:00:00") {
		t.Error("date-only display", s)
	}
	p := Must(Must(NewSeriesMasked("t", []time.Time{time.Unix(0, 0)}, []bool{false})).ToPeriod("M"))
	if !strings.Contains(p.String(), "NaT") {
		t.Error("period NaT")
	}
	r := Must(New(Col("v", []int{1, 2, 3})))
	_, err = r.Loc(LabelRange(0.5, 2))
	fails(t, err, ErrLabel)
	if math.IsNaN(0) {
		t.Error()
	}
}

func b2() bitmap {
	b := allValid(2)
	b.clear(0)
	b.clear(1)
	return b
}

func TestQueryErrorsInside(t *testing.T) {
	df := Must(New(Col("a", []int{1, 2})))
	for _, bad := range []string{"a > 1 and [1]", "[1] or a > 1", "not [1]", "not )", "[1] == a", "[1] + a",
		"- )", "a ** )", "abs([1])", "( )", "abs( )", "a > 1 and )"} {
		if _, err := df.Query(bad); err == nil {
			t.Errorf("%q: no error", bad)
		}
	}
}

func TestParallelFactorize(t *testing.T) {
	n := 300_000
	s := make([]string, n)
	valid := make([]bool, n)
	for i := range s {
		s[i] = string(rune('a' + (i*7)%26))
		valid[i] = i%11 != 0
	}
	col := Must(NewSeriesMasked("s", s, valid)).c
	codes, u := col.factorize()
	pa, _, pu := col.factorizeWith(col.slice(0, 0))
	if u.len() != pu.len() {
		t.Fatal("unique counts differ")
	}
	for i := range codes {
		if int32(codes[i]) != pa[i] {
			t.Fatalf("code %d: %d vs %d", i, codes[i], pa[i])
		}
	}
	// Two keys, the first missing on some rows.
	df := Must(New(Col("a", []any{nil, "x", "x"}), Col("b", []int{1, 2, 2})))
	if g := Must(df.GroupBy("a", "b")); g.NGroups() != 1 {
		t.Error("missing first key", g.NGroups())
	}
}

func TestCombine(t *testing.T) {
	acc := []int{-1, 1, 2}
	combine(acc, []int32{0, -1, 1}, 2)
	if acc[0] != -1 || acc[1] != -1 || acc[2] != 5 {
		t.Error(acc)
	}
}

func TestMaskWithMissing(t *testing.T) {
	df := Must(New(Col("a", []int{1, 2, 3})))
	m := Must(NewSeriesMasked("m", []bool{true, true, false}, []bool{true, false, true}))
	if g := Must(df.Filter(m)); g.Len() != 1 {
		t.Error("missing mask entries are false")
	}
}
