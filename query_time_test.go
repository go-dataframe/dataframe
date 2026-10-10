package dataframe

import (
	"math"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestQueryLanguage(t *testing.T) {
	df := Must(New(
		Col("a", []int{1, 2, 3, 4}),
		Col("b b", []float64{-1.5, 2, math.NaN(), 4}),
		Col("s", []string{"x", "y", "x", "z"}),
		Col("f", []bool{true, false, true, false}),
	))
	rows := func(expr string, vars ...map[string]any) []any {
		t.Helper()
		r := Must(df.Query(expr, vars...))
		return Must(r.Column("a")).Items()
	}
	for expr, want := range map[string][]any{
		"`b b` > 0":                       {int64(2), int64(4)},
		"abs(`b b`) > 1.6":                {int64(2), int64(4)},
		"notna(`b b`) and a >= 2":         {int64(2), int64(4)},
		"isnull(`b b`) or notnull(a) & f": {int64(1), int64(3)},
		"`b b`.notna() and not f":         {int64(2), int64(4)},
		"isna(`b b`)":                     {int64(3)},
		"s == \"x\" and a != 3":           {int64(1)},
		"s in ['y', 'z'] and a < 4":       {int64(2)},
		"a in [1.0, 4] | f == True":       {int64(1), int64(3), int64(4)},
		"f == false":                      {int64(2), int64(4)},
		"-a < -2.5":                       {int64(3), int64(4)},
		"+a > 3":                          {int64(4)},
		"a * 1_0 >= 3e1":                  {int64(3), int64(4)},
		"a / 2 == .5":                     {int64(1)},
		"2 ** a == 8":                     {int64(3)},
		"(a + 1) * 2 > 9":                 {int64(4)},
		"a in @keep":                      {int64(2)},
		"s in @names":                     {int64(2)},
		"a > @lim":                        {int64(4)},
		"a in @fl":                        {int64(1)},
		"a == @ser":                       {int64(1), int64(2), int64(3), int64(4)},
		"index >= 2":                      {int64(3), int64(4)},
		"s == 'it\\'s' or a == 1":         {int64(1)},
		"a in [@lim]":                     {int64(3)},
		"a == 2 or a in 3":                {int64(2), int64(3)},
	} {
		vars := map[string]any{"keep": []int{2}, "names": []string{"y"}, "lim": 3, "fl": []float64{1},
			"ser": Must(df.Column("a"))}
		vars["any"] = []any{1}
		if got := rows(expr, vars); !reflect.DeepEqual(got, want) {
			t.Errorf("%s: %v, want %v", expr, got, want)
		}
	}
	for _, bad := range []string{"a >", "a > 'x", "a > `b", "a $ 1", "a", "a > 1 )", "zz > 1",
		"@nope > 1", "a.foo()", "a.isna(1)", "a.", "a.isna", "nosuch(a)", "abs(s) > 1", "s + 1 > 0",
		"a > 'x'", "(a > 1", "[1, 2", "a in [1, 'x']", "1e > 2", ")", "s < 1 < 2", "not s", "s and f",
		"f and s", "-s > 1", "a ** s > 1", "a // s > 1", "a > 1 or s", "a in [x]", "a in [1", "a + ", "isna(s, a)"} {
		if _, err := df.Query(bad); err == nil {
			t.Errorf("%q: no error", bad)
		}
	}
	v := Must(df.Eval("a * 2"))
	if v.c.value(3) != int64(8) {
		t.Error("Eval")
	}
	if c := Must(df.Eval("3")); c.Len() != 4 {
		t.Error("constant Eval")
	}
	_, err := df.Eval("@x", map[string]any{"x": struct{}{}})
	fails(t, err, ErrQuery)
	named := Must(df.SetIndex("s"))
	if r := Must(named.Query("s == 'x'")); r.Len() != 2 {
		t.Error("query on index name")
	}
	if ls, ok := listOf([]any{1}); !ok || len(ls) != 1 {
		t.Error("listOf any")
	}
}

func TestTimeParsing(t *testing.T) {
	ok := map[string]string{
		"2024":                      "2024-01-01 00:00:00",
		"2024-03":                   "2024-03-01 00:00:00",
		"2024-02-29T12:30":          "2024-02-29 12:30:00",
		"2024-02-29 12:30:15.25":    "2024-02-29 12:30:15.250000",
		"2024-02-29 12:30:15,5Z":    "2024-02-29 12:30:15.500000",
		"2024-01-01T01:00:00+01:00": "2024-01-01 00:00:00",
		"2024-01-01T01:00:00-0130":  "2024-01-01 02:30:00",
		"2024-01-01T01:00:00+02":    "2023-12-31 23:00:00",
		"2024/01/02":                "2024-01-02 00:00:00",
		"01/02/2024":                "2024-01-02 00:00:00",
		"Jan 2, 2024":               "2024-01-02 00:00:00",
		"1969-12-31 23:59:59":       "1969-12-31 23:59:59",
		"1700-03-01":                "1700-03-01 00:00:00",
		"-001-01-01T00:00:00Z":      "",
	}
	for in, want := range ok {
		ns, err := parseTime(in, "")
		if want == "" {
			continue
		}
		if err != nil || formatTime(ns) != want {
			t.Errorf("%q: %v %q, want %q", in, err, formatTime(ns), want)
		}
	}
	for _, bad := range []string{"20x4", "2024-1", "2024-13-01", "2023-02-29", "2024-01-01X", "2024-01-01 1:00",
		"2024-01-01 25:00", "2024-01-01 12:00:61", "2024-01-01 12:00:00.", "2024-01-01 12:00+1", "2024-01-01 12:00+ab",
		"2024-01-01 12:00 ", "2024-01-01 12:00+0a:00", "2024-01-01 12:00:6", "2024-01-32", "2024-0a", "1600-01-01", "1600/01/01", "3000-01-01T00:00:00.5+00:00"} {
		if _, err := parseTime(bad, ""); err == nil {
			t.Errorf("%q parsed", bad)
		}
	}
	if ns, err := parseTime("31/12/2023", "%d/%m/%Y"); err != nil || formatTime(ns) != "2023-12-31 00:00:00" {
		t.Error("strftime")
	}
	if ns, err := parseTime("2023-12-31", "2006-01-02"); err != nil || ns != 1703980800e9 {
		t.Error("go layout")
	}
	_, err := parseTime("x", "2006")
	fails(t, err, ErrParse)
	_, err = parseTime("1500", "2006")
	fails(t, err, ErrParse)
	for d := int64(-800000); d < 800000; d += 997 {
		y, m, dd := civilFromDays(d)
		if daysFromCivil(y, m, dd) != d {
			t.Fatalf("civil round trip %d", d)
		}
	}
	if Must(Timestamp("2020-05-06")).Day() != 6 {
		t.Error("Timestamp")
	}
	if Must(Timestamp("06/05/2020", "%d/%m/%Y")).Month() != time.May {
		t.Error("Timestamp layout")
	}
	s := Must(NewSeries("s", []any{"2020-01-01", nil}))
	if d := Must(ToDatetime(s)); d.c.value(1) != nil {
		t.Error("ToDatetime missing")
	}
	_, err = ToDatetime(Must(NewSeries("s", []string{"nope"})))
	fails(t, err, ErrParse)
	if d := Must(ToDatetime(Must(NewSeries("n", []int64{86400e9})))); d.c.value(0) != time.Unix(86400, 0).UTC() {
		t.Error("ToDatetime ints")
	}
	if d := Must(ToDatetime(Must(NewSeries("f", []float64{1e9})))); d.c.value(0) != time.Unix(1, 0).UTC() {
		t.Error("ToDatetime floats")
	}
	_, err = ToDatetime(Must(NewSeries("b", []bool{true})))
	fails(t, err, ErrDType)
	_, err = ToDatetime(Must(NewSeries("z", []complex128{1})))
	fails(t, err, ErrDType)
	dts := Must(NewSeries("t", []time.Time{time.Unix(0, 0)}))
	if d := Must(ToDatetime(dts)); d.DType() != Datetime {
		t.Error("ToDatetime of datetimes")
	}
}

func TestTimeSeries(t *testing.T) {
	ts := Must(NewSeriesMasked("t", []time.Time{
		time.Date(2024, 2, 29, 13, 5, 7, 0, time.UTC),
		time.Date(1969, 12, 31, 23, 0, 0, 0, time.UTC),
		time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC),
	}, []bool{true, true, false}))
	want := map[string][]any{
		"Y": {"2024", "1969", nil}, "Q": {"2024Q1", "1969Q4", nil}, "M": {"2024-02", "1969-12", nil},
		"W": {"2024-02-26/2024-03-03", "1969-12-29/1970-01-04", nil}, "D": {"2024-02-29", "1969-12-31", nil},
		"h": {"2024-02-29 13:00", "1969-12-31 23:00", nil}, "min": {"2024-02-29 13:05", "1969-12-31 23:00", nil},
		"s": {"2024-02-29 13:05:07", "1969-12-31 23:00:00", nil},
	}
	for f, w := range want {
		p := Must(ts.ToPeriod(f))
		if got := p.Items(); !reflect.DeepEqual(got, w) {
			t.Errorf("%s: %v", f, got)
		}
		back := Must(p.ToTimestamp())
		if back.c.isNA(2) != true || Must(Must(back.Iloc(Pos(0))).At(0)).(time.Time).After(Must(ts.At(0)).(time.Time)) {
			t.Errorf("%s: ToTimestamp", f)
		}
		pc := p.c.(*col[int64])
		if one, ok := pc.convert("2024-02-29 13:05:07"); !ok || one.str(0) != w[0] {
			t.Errorf("%s: period from string", f)
		}
	}
	_, err := ts.ToPeriod("fortnight")
	fails(t, err, ErrArgument)
	_, err = Must(NewSeries("x", []int{1})).ToPeriod("M")
	fails(t, err, ErrDType)
	_, err = Must(NewSeries("x", []int{1})).ToTimestamp()
	fails(t, err, ErrDType)
	_, err = Must(NewSeries("x", []int{1})).Dt()
	fails(t, err, ErrDType)
	d := Must(ts.Dt())
	if d.Minute().c.value(0) != int64(5) || d.Second().c.value(0) != int64(7) || d.Normalize().c.value(0) != time.Date(2024, 2, 29, 0, 0, 0, 0, time.UTC) {
		t.Error("fields")
	}
	if d.Normalize().c.value(1) != time.Date(1969, 12, 31, 0, 0, 0, 0, time.UTC) {
		t.Error("normalize before epoch")
	}
	if d.Strftime("%Y/%m").c.value(0) != "2024/02" || d.Strftime("2006").c.value(1) != "1969" {
		t.Error("Strftime")
	}
	df := Must(New(Col("t", ts), Col("v", []float64{1, 2, 3})))
	idx := Must(df.SetIndex("t"))
	pf := Must(idx.ToPeriod("M"))
	if pf.index.DType() != Period {
		t.Error("frame ToPeriod")
	}
	_, err = df.ToPeriod("M")
	fails(t, err, ErrDType)
	_, err = Must(df.SetIndex("t", "v")).ToPeriod("M")
	fails(t, err, ErrArgument)
	_, err = df.Resample("D")
	fails(t, err, ErrArgument)
	_, err = df.ResampleOn("zz", "D")
	fails(t, err, ErrColumn)
	_, err = df.ResampleOn("v", "D")
	fails(t, err, ErrDType)
	_, err = df.ResampleOn("t", "3ME")
	fails(t, err, ErrArgument)
	_, err = df.ResampleOn("t", "0h")
	fails(t, err, ErrArgument)
	r := Must(df.ResampleOn("t", "YS"))
	sum := Must(r.Sum())
	if sum.Len() != 56 || sum.index.Label(0) != time.Date(1969, 1, 1, 0, 0, 0, 0, time.UTC) || sum.cols[0].value(55) != 1.0 {
		t.Error("YS", sum.Len(), sum.index.Label(0))
	}
	for name, f := range map[string]func() (*DataFrame, error){"min": r.Min, "max": r.Max, "first": r.First,
		"last": r.Last, "std": r.Std, "median": r.Median, "mean": r.Mean, "count": r.Count} {
		if g, err := f(); err != nil || g.Len() != 56 {
			t.Error(name, err)
		}
	}
	if a := Must(r.Agg(Agg("v", "sum", "count"))); a.names[1] != "v_count" {
		t.Error("Agg")
	}
	if r.Size().Len() != 56 {
		t.Error("Size")
	}
	for _, rule := range []string{"QS", "MS", "W-SUN", "A", "AS", "Y", "M", "Q"} {
		if _, err := df.ResampleOn("t", rule); err != nil {
			t.Error(rule, err)
		}
	}
	day := Must(New(Col("t", []time.Time{time.Unix(0, 0), time.Unix(3600, 0)}), Col("v", []int{1, 2})))
	for _, rule := range []string{"2D", "30s", "T", "H", "S"} {
		if _, err := day.ResampleOn("t", rule); err != nil {
			t.Error(rule, err)
		}
	}
	empty := Must(New(Col("t", Must(NewSeriesMasked("t", []time.Time{time.Unix(0, 0)}, []bool{false}))), Col("v", []int{1})))
	if e := Must(Must(empty.ResampleOn("t", "D")).Sum()); e.Len() != 0 {
		t.Error("all-missing times")
	}
	if floorDiv64(-1, 2) != -1 || floorDiv64(3, 2) != 1 || floorDiv64(-4, 2) != -2 {
		t.Error("floorDiv64")
	}
}

func TestDisplay(t *testing.T) {
	df := Must(New(
		Col("f", []float64{1e12 + 0.5, math.Inf(1), math.Inf(-1)}),
		Col("g", []float32{1.5, 2, 3}),
		Col("t", []time.Time{time.Unix(0, 0), time.Unix(3600, 0), time.Unix(7200, 0)}),
		Col("n", []any{nil, int64(1), int64(2)}),
		Col("u", []string{"é", "ü", "<b>"}),
	))
	s := df.String()
	for _, want := range []string{"1.000000e+12", "inf", "-inf", "1.5", "1970-01-01 01:00:00", "<NA>"} {
		if !strings.Contains(s, want) {
			t.Errorf("String lacks %q:\n%s", want, s)
		}
	}
	h := df.HTML()
	if !strings.Contains(h, "&lt;b&gt;") || !strings.Contains(h, "<th>f</th>") {
		t.Error("HTML", h)
	}
	if m := df.Display(); string(m["text/plain"]) != s || string(m["text/html"]) != h {
		t.Error("Display")
	}
	nat := Must(New(Col("t", Must(NewSeriesMasked("t", []time.Time{time.Unix(0, 0)}, []bool{false})))))
	if !strings.Contains(nat.String(), "NaT") {
		t.Error("NaT")
	}
	if e := Must(New()); !strings.HasPrefix(e.String(), "Empty DataFrame") {
		t.Error("empty frame")
	}
	// Truncation in both directions, with a named index.
	var cols []Column
	for k := range 25 {
		cols = append(cols, Col("c"+string(rune('a'+k)), make([]int, 70)))
	}
	wide := Must(New(cols...))
	wide = Must(wide.WithIndex(Must(Must(NewIndex("", make([]int, 70))).Rename("row"))))
	ws := wide.String()
	if !strings.Contains(ws, "[70 rows x 25 columns]") || !strings.Contains(ws, "...") {
		t.Error("truncated String")
	}
	wh := wide.HTML()
	if !strings.Contains(wh, "70 rows × 25 columns") || !strings.Contains(wh, "<th>row</th>") {
		t.Error("truncated HTML")
	}
	one := Must(New(Col("x", make([]int, 70))))
	one = Must(one.WithIndex(Must(NewIndex("", make([]int, 70)))))
	if !strings.Contains(one.String(), "\n. ") {
		t.Error("narrow dots", one.String())
	}
	tiny := Must(New(Col("x", make([]int, 70))))
	tiny = Must(tiny.WithIndex(Must(NewIndex("", make([]bool, 70)))))
	_ = tiny.String()
	// A series: named index, truncated, unnamed, empty.
	ser := Must(NewSeriesIndexed("v", make([]float64, 70), Must(Must(NewIndex("", make([]int, 70))).Rename("k"))))
	ss := ser.String()
	if !strings.HasPrefix(ss, "k\n") || !strings.Contains(ss, "Length: 70") {
		t.Error("series String", ss)
	}
	if !strings.Contains(ser.HTML(), "<table") || string(ser.Display()["text/plain"]) != ss {
		t.Error("series HTML/Display")
	}
	if e := Must(NewSeries("", []int{})); e.String() != "Series([], dtype: int64)" {
		t.Error(e.String())
	}
	if dts := df.DTypesSeries(); !strings.HasSuffix(dts.String(), "dtype: string") {
		t.Error(dts.String())
	}
	if d := dotsFor(1); d != "." {
		t.Error(d)
	}
}
