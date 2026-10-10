package dataframe

import (
	"bytes"
	_ "embed"
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"testing"
	"time"
)

// The fixtures are pandas' answers to the operations below, recorded by
// testdata/gen_fixtures.py. Each case gives input frames, an operation and
// what pandas returned; the test runs the same operation here and compares
// values, dtypes and index labels. Where this package deliberately differs
// from pandas (see docs/design.md), the comparison allows exactly that
// difference and no other.

//go:embed testdata/fixtures.json
var fixturesJSON []byte

type fxLevel struct {
	DType  string `json:"dtype"`
	Values []any  `json:"values"`
}

type fxIndex struct {
	Range  []json.Number `json:"range"`
	Names  []string      `json:"names"`
	Levels []fxLevel     `json:"levels"`
}

type fxCol struct {
	Name   string `json:"name"`
	DType  string `json:"dtype"`
	Values []any  `json:"values"`
}

type fxFrame struct {
	Columns []fxCol `json:"columns"`
	Index   fxIndex `json:"index"`
}

type fxSeries struct {
	Name   string  `json:"name"`
	DType  string  `json:"dtype"`
	Values []any   `json:"values"`
	Index  fxIndex `json:"index"`
}

type fxCase struct {
	Name   string         `json:"name"`
	Op     string         `json:"op"`
	Args   map[string]any `json:"args"`
	Inputs []fxFrame      `json:"inputs"`
	Expect struct {
		Kind   string    `json:"kind"`
		Frame  *fxFrame  `json:"frame"`
		Series *fxSeries `json:"series"`
		Text   string    `json:"text"`
		Value  any       `json:"value"`
		Error  string    `json:"error"`
	} `json:"expect"`
}

func loadFixtures(t *testing.T) []fxCase {
	t.Helper()
	var doc struct {
		Cases []fxCase `json:"cases"`
	}
	dec := json.NewDecoder(bytes.NewReader(fixturesJSON))
	dec.UseNumber()
	if err := dec.Decode(&doc); err != nil {
		t.Fatal(err)
	}
	return doc.Cases
}

// fxColumn builds a column from fixture values.
func fxColumn(t *testing.T, dtype string, vals []any) column {
	t.Helper()
	n := len(vals)
	valid := allValid(n)
	missing := false
	for i, v := range vals {
		if v == nil {
			valid.clear(i)
			missing = true
		}
	}
	var c column
	switch dtype {
	case "float64", "float32":
		x := makeCol[float64](Float64, n)
		for i, v := range vals {
			x.data[i] = fxFloat(v)
		}
		c = x
	case "bool":
		x := makeCol[bool](Bool, n)
		for i, v := range vals {
			x.data[i], _ = v.(bool)
		}
		c = x
	case "string", "category", "period":
		x := makeCol[string](String, n)
		for i, v := range vals {
			x.data[i], _ = v.(string)
		}
		c = x
	case "datetime":
		x := makeCol[int64](Datetime, n)
		for i, v := range vals {
			if v != nil {
				x.data[i], _ = v.(json.Number).Int64()
			}
		}
		c = x
	default:
		x := makeCol[int64](Int64, n)
		for i, v := range vals {
			if v != nil {
				x.data[i], _ = v.(json.Number).Int64()
			}
		}
		c = x
	}
	if missing {
		c = c.withValidity(valid)
	}
	dt, err := ParseDType(dtype)
	if err == nil && dt != c.dtype() && dtype != "period" {
		if c, err = cast(c, dt); err != nil {
			t.Fatal(err)
		}
	}
	return c
}

func fxFloat(v any) float64 {
	switch x := v.(type) {
	case json.Number:
		f, _ := x.Float64()
		return f
	case string:
		if x == "inf" {
			return math.Inf(1)
		}
		return math.Inf(-1)
	}
	return math.NaN()
}

func fxIndexOf(t *testing.T, ix fxIndex, n int) *Index {
	if ix.Range != nil {
		start, _ := ix.Range[0].Int64()
		step, _ := ix.Range[2].Int64()
		if start == 0 && step == 1 {
			out := NewRangeIndex(n)
			out.names = ix.Names
			return out
		}
		return &Index{n: n, start: int(start), step: int(step), names: ix.Names}
	}
	levels := make([]column, len(ix.Levels))
	for k, l := range ix.Levels {
		levels[k] = fxColumn(t, l.DType, l.Values)
	}
	return labelIndex(ix.Names, levels)
}

func fxFrameOf(t *testing.T, f fxFrame) *DataFrame {
	t.Helper()
	df := &DataFrame{}
	n := 0
	for _, c := range f.Columns {
		col := fxColumn(t, c.DType, c.Values)
		df.names = append(df.names, c.Name)
		df.cols = append(df.cols, col)
		n = col.len()
	}
	if f.Index.Range != nil {
		a, _ := f.Index.Range[0].Int64()
		b, _ := f.Index.Range[1].Int64()
		s, _ := f.Index.Range[2].Int64()
		n = int((b - a) / s)
	} else if len(f.Index.Levels) > 0 {
		n = len(f.Index.Levels[0].Values)
	}
	df.index = fxIndexOf(t, f.Index, n)
	return df
}

// deviation records a difference from pandas that the design allows.
type deviation struct{ what string }

// sameValue compares one of our values with pandas' (JSON) value.
func fxSame(got any, want any, dt DType) bool {
	if want == nil || got == nil {
		if want == nil && got != nil {
			// pandas' float NaN against our present NaN is still NaN.
			if f, ok := got.(float64); ok && math.IsNaN(f) {
				return true
			}
		}
		return want == nil && got == nil
	}
	switch w := want.(type) {
	case json.Number:
		if tm, ok := got.(time.Time); ok {
			n, _ := w.Int64()
			return tm.UnixNano() == n
		}
		wf, _ := w.Float64()
		gf, ok := toFloat(got)
		if b, isBool := got.(bool); isBool {
			gf, ok = map[bool]float64{false: 0, true: 1}[b], true
		}
		if !ok {
			return false
		}
		return wf == gf || math.Abs(wf-gf) <= 1e-9*math.Max(math.Abs(wf), math.Abs(gf))
	case string:
		if f, ok := got.(float64); ok && (w == "inf" || w == "-inf") {
			return math.IsInf(f, map[string]int{"inf": 1, "-inf": -1}[w])
		}
		switch g := got.(type) {
		case string:
			return g == w
		case bool:
			return map[bool]string{true: "True", false: "False"}[g] == w
		case time.Time:
			return formatTime(g.UnixNano()) == w || g.Format("2006-01-02") == w
		}
		return fmt.Sprint(got) == w
	case bool:
		g, ok := got.(bool)
		return ok && g == w
	}
	return false
}

// dtypeOK reports whether our dtype stands for pandas' one, and whether that
// needs a deviation note.
func dtypeOK(got DType, want string, hasNA bool) (bool, string) {
	g := got.String()
	switch {
	case g == want:
		return true, ""
	case want == "float64" && (got.isInt() || got == Bool) && hasNA:
		return true, "int/bool column with missing values stays " + g + " (pandas: float64)"
	case want == "string" && got == Bool && hasNA:
		return true, "bool column with missing values stays bool (pandas: object)"
	case want == "period" && got == Period:
		return true, ""
	case want == "float64" && got == Float64:
		return true, ""
	}
	return false, ""
}

func fxCheckColumn(t *testing.T, label string, c column, dtype string, vals []any, devs map[string]bool) {
	t.Helper()
	if c.len() != len(vals) {
		t.Errorf("%s: %d values, pandas %d", label, c.len(), len(vals))
		return
	}
	ok, dev := dtypeOK(c.dtype(), dtype, c.naCount() > 0)
	if !ok {
		t.Errorf("%s: dtype %v, pandas %s", label, c.dtype(), dtype)
	}
	if dev != "" {
		devs[dev] = true
	}
	for i, w := range vals {
		if !fxSame(c.value(i), w, c.dtype()) {
			t.Errorf("%s[%d]: %v, pandas %v", label, i, c.value(i), w)
			return
		}
	}
}

func fxCheckIndex(t *testing.T, label string, got *Index, want fxIndex, devs map[string]bool) {
	t.Helper()
	if want.Range != nil {
		start, _ := want.Range[0].Int64()
		step, _ := want.Range[2].Int64()
		for i := range got.Len() {
			l, _ := toInt(got.Label(i))
			if got.NLevels() != 1 || l != start+int64(i)*step {
				t.Errorf("%s: index label %d is %v, pandas range %v", label, i, got.Label(i), want.Range)
				return
			}
		}
		return
	}
	if got.NLevels() != len(want.Levels) {
		t.Errorf("%s: %d index levels, pandas %d", label, got.NLevels(), len(want.Levels))
		return
	}
	for k, l := range want.Levels {
		fxCheckColumn(t, label+" index level "+fmt.Sprint(k), got.level(k), l.DType, l.Values, devs)
		if got.names[k] != want.Names[k] {
			t.Errorf("%s: index name %q, pandas %q", label, got.names[k], want.Names[k])
		}
	}
}

func fxCheckFrame(t *testing.T, got *DataFrame, want *fxFrame, devs map[string]bool) {
	t.Helper()
	var names []string
	for _, c := range want.Columns {
		names = append(names, c.Name)
	}
	if strings.Join(got.names, "|") != strings.Join(names, "|") {
		t.Errorf("columns %v, pandas %v", got.names, names)
		return
	}
	for k, c := range want.Columns {
		fxCheckColumn(t, "column "+c.Name, got.cols[k], c.DType, c.Values, devs)
	}
	fxCheckIndex(t, "frame", got.index, want.Index, devs)
}

func fxCheckSeries(t *testing.T, got *Series, want *fxSeries, devs map[string]bool) {
	t.Helper()
	fxCheckColumn(t, "series", got.c, want.DType, want.Values, devs)
	fxCheckIndex(t, "series", got.index, want.Index, devs)
}

func argStrings(v any) []string {
	var out []string
	for _, x := range v.([]any) {
		out = append(out, x.(string))
	}
	return out
}

func argInt(v any) int {
	n, _ := v.(json.Number).Int64()
	return int(n)
}

func argScalar(v any) any {
	if n, ok := v.(json.Number); ok {
		if i, err := n.Int64(); err == nil {
			return i
		}
		f, _ := n.Float64()
		return f
	}
	return v
}

func argBools(v any) []bool {
	var out []bool
	for _, x := range v.([]any) {
		out = append(out, x.(bool))
	}
	return out
}

// run performs a fixture's operation.
func (c *fxCase) run(t *testing.T, in []*DataFrame) (any, error) {
	df := in[0]
	a := c.Args
	col := func(name string) *Series { return Must(df.Column(name)) }
	switch c.Op {
	case "head":
		return df.Head(argInt(a["n"])), nil
	case "tail":
		return df.Tail(argInt(a["n"])), nil
	case "T":
		return df.T(), nil
	case "round":
		return df.Round(argInt(a["decimals"])), nil
	case "rename":
		m := map[string]string{}
		for k, v := range a["mapping"].(map[string]any) {
			m[k] = v.(string)
		}
		return df.Rename(m), nil
	case "astype":
		m := map[string]DType{}
		for k, v := range a["dtypes"].(map[string]any) {
			m[k] = Must(ParseDType(v.(string)))
		}
		return df.AsType(m)
	case "copy":
		return df.Copy(), nil
	case "iloc_slice":
		return df.Iloc(Slice(argInt(a["start"]), argInt(a["stop"])))
	case "iloc_pos":
		var p []int
		for _, x := range a["pos"].([]any) {
			p = append(p, argInt(x))
		}
		return df.Iloc(Pos(p...))
	case "iloc_cols":
		var p []int
		for _, x := range a["pos"].([]any) {
			p = append(p, argInt(x))
		}
		return df.Iloc(Pos(p...), Slice(argInt(a["cstart"]), argInt(a["cstop"])))
	case "loc_labels":
		var l []any
		for _, x := range a["labels"].([]any) {
			l = append(l, x)
		}
		return df.Loc(Labels(l...))
	case "loc_range":
		return df.Loc(LabelRange(a["start"], a["stop"]))
	case "filter_gt", "filter_eq", "filter_ne":
		s := col(a["column"].(string))
		var m *Series
		var err error
		switch c.Op {
		case "filter_gt":
			m, err = s.Gt(argScalar(a["value"]))
		case "filter_eq":
			m, err = s.Eq(argScalar(a["value"]))
		default:
			m, err = s.Ne(argScalar(a["value"]))
		}
		if err != nil {
			return nil, err
		}
		return df.Filter(m)
	case "set_index":
		return df.SetIndex(argStrings(a["keys"])...)
	case "reset_index":
		return df.ResetIndex(a["drop"].(bool)), nil
	case "sort_values":
		return df.SortValues(argStrings(a["by"]), argBools(a["ascending"])...)
	case "sort_index":
		return df.SortIndex(a["ascending"].(bool)), nil
	case "reindex":
		return df.Reindex(argStrings(a["labels"]))
	case "query":
		vars := map[string]any{}
		if v, ok := a["vars"].(map[string]any); ok {
			for k, x := range v {
				vars[k] = argScalar(x)
			}
		}
		return df.Query(a["expr"].(string), vars)
	case "dropna":
		if a["how"] == "all" {
			return df.DropNAAll(argStrings(a["subset"])...)
		}
		return df.DropNA(argStrings(a["subset"])...)
	case "dropna_thresh":
		return df.DropNAThresh(argInt(a["thresh"]))
	case "fillna":
		return df.FillNA(argScalar(a["value"])), nil
	case "fillna_map":
		m := map[string]any{}
		for k, v := range a["values"].(map[string]any) {
			m[k] = argScalar(v)
		}
		return df.FillNAColumns(m)
	case "ffill":
		return df.FFill(), nil
	case "bfill":
		return df.BFill(), nil
	case "isna":
		return df.IsNA(), nil
	case "reduce":
		switch a["fn"] {
		case "sum":
			return df.Sum(), nil
		case "mean":
			return df.Mean(), nil
		case "std":
			return df.Std(), nil
		case "var":
			return df.Var(), nil
		case "min":
			return df.Min(), nil
		case "max":
			return df.Max(), nil
		case "median":
			return df.Median(), nil
		}
		return df.Count(), nil
	case "describe":
		return df.Describe(), nil
	case "value_counts":
		return col(a["column"].(string)).ValueCounts(), nil
	case "unique":
		return col(a["column"].(string)).Unique(), nil
	case "nunique":
		return int64(col(a["column"].(string)).NUnique()), nil
	case "quantile":
		q, _ := a["q"].(json.Number).Float64()
		return col(a["column"].(string)).Quantile(q), nil
	case "series_arith":
		s1 := Must(in[0].Column("v"))
		s2 := Must(in[1].Column("v"))
		if a["opname"] == "add" {
			return s1.Add(s2)
		}
		return s1.Mul(s2)
	case "col_arith":
		x := col(a["a"].(string))
		var other any
		if b, ok := a["b"]; ok {
			other = col(b.(string))
		} else {
			other = argScalar(a["scalar"])
		}
		ops := map[string]arithOp{"add": opAdd, "sub": opSub, "mul": opMul, "div": opDiv,
			"floordiv": opFloorDiv, "mod": opMod, "pow": opPow}
		return x.arith(ops[a["opname"].(string)], other)
	case "isin":
		var v []any
		for _, x := range a["values"].([]any) {
			v = append(v, x)
		}
		return col(a["column"].(string)).IsIn(v...)
	case "between":
		return col(a["column"].(string)).Between(argScalar(a["lo"]), argScalar(a["hi"]))
	case "frame_arith":
		if len(in) > 1 {
			return df.Add(in[1])
		}
		return df.Add(argScalar(a["scalar"]))
	case "groupby", "groupby_size", "groupby_agg", "groupby_transform":
		g, err := df.GroupBy(argStrings(a["keys"])...)
		if err != nil {
			return nil, err
		}
		switch c.Op {
		case "groupby_size":
			return g.Size(), nil
		case "groupby_transform":
			return g.Transform(a["column"].(string), a["fn"].(string))
		case "groupby_agg":
			var specs []AggSpec
			for _, s := range a["specs"].([]any) {
				p := s.([]any)
				specs = append(specs, Agg(p[0].(string), argStrings(p[1])...))
			}
			r, err := g.Agg(specs...)
			if err != nil {
				return nil, err
			}
			return r.Rename(map[string]string{"y_max": "y_max"}), nil
		}
		return g.aggregate(a["fn"].(string))
	case "concat":
		return ConcatWith(ConcatOptions{Axis: argInt(a["axis"]), IgnoreIndex: a["ignore_index"].(bool),
			Inner: a["inner"].(bool)}, in...)
	case "merge":
		o := MergeOptions{How: a["how"].(string)}
		if v, ok := a["on"]; ok {
			o.On = argStrings(v)
		}
		if v, ok := a["left_on"]; ok {
			o.LeftOn, o.RightOn = argStrings(v), argStrings(a["right_on"])
		}
		o.Indicator, _ = a["indicator"].(bool)
		return Merge(in[0], in[1], o)
	case "join":
		return in[0].Join(in[1], a["how"].(string), "", "")
	case "pivot":
		return df.Pivot(a["index"].(string), a["columns"].(string), a["values"].(string))
	case "pivot_table":
		return df.PivotTable(a["index"].(string), a["columns"].(string), a["values"].(string), a["aggfunc"].(string))
	case "stack":
		return df.Stack()
	case "unstack":
		return col(a["column"].(string)).Unstack()
	case "melt":
		return df.Melt(argStrings(a["id_vars"]), argStrings(a["value_vars"]))
	case "to_datetime":
		if f, ok := a["format"]; ok {
			return ToDatetime(col(a["column"].(string)), f.(string))
		}
		return ToDatetime(col(a["column"].(string)))
	case "resample":
		r, err := df.Resample(a["rule"].(string))
		if err != nil {
			return nil, err
		}
		switch a["fn"] {
		case "sum":
			return r.Sum()
		case "mean":
			return r.Mean()
		case "count":
			return r.Count()
		}
		return r.Max()
	case "to_period":
		p, err := (&Series{index: NewRangeIndex(df.Len()), c: df.index.level(0)}).ToPeriod(a["freq"].(string))
		if err != nil {
			return nil, err
		}
		return p.AsType(String)
	case "dt_fields":
		d := Must(col(a["column"].(string)).Dt())
		return FromSeries(d.Year().Rename("year"), d.Month().Rename("month"), d.Day().Rename("day"),
			d.Hour().Rename("hour"), d.DayOfWeek().Rename("dayofweek"), d.DayOfYear().Rename("dayofyear"))
	case "window":
		v := col("v")
		switch a["fn"] {
		case "shift":
			return v.Shift(argInt(a["n"])), nil
		case "diff":
			return v.Diff(argInt(a["n"]))
		case "pct_change":
			return v.PctChange(argInt(a["n"]))
		case "cumsum":
			return v.CumSum()
		case "cumprod":
			return v.CumProd()
		case "cummax":
			return v.CumMax()
		}
		return v.CumMin()
	case "rolling":
		r, err := col("v").RollingMin(argInt(a["window"]), argInt(a["min_periods"]))
		if err != nil {
			return nil, err
		}
		return map[string]func() *Series{"sum": r.Sum, "mean": r.Mean, "std": r.Std, "min": r.Min,
			"max": r.Max, "median": r.Median}[a["fn"].(string)](), nil
	case "repr":
		return df.String(), nil
	case "repr_series":
		return col(a["column"].(string)).String(), nil
	}
	t.Fatalf("unknown op %q", c.Op)
	return nil, nil
}

// knownTextDiffs lists the repr cases whose text differs from pandas, with
// the reason; every other repr must match exactly.
var knownTextDiffs = map[string]string{}

func TestPandasFixtures(t *testing.T) {
	devs := map[string]bool{}
	for _, c := range loadFixtures(t) {
		t.Run(c.Name, func(t *testing.T) {
			in := make([]*DataFrame, len(c.Inputs))
			for i, f := range c.Inputs {
				in[i] = fxFrameOf(t, f)
			}
			got, err := c.run(t, in)
			if c.Expect.Kind == "error" {
				if err == nil {
					t.Errorf("pandas raises (%s); we returned %v", c.Expect.Error, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("error: %v (pandas answered)", err)
			}
			switch c.Expect.Kind {
			case "frame":
				fxCheckFrame(t, got.(*DataFrame), c.Expect.Frame, devs)
			case "series":
				fxCheckSeries(t, got.(*Series), c.Expect.Series, devs)
			case "text":
				if s := got.(string); s != c.Expect.Text {
					if why, ok := knownTextDiffs[c.Name]; ok {
						devs["display: "+why] = true
						return
					}
					t.Errorf("text differs:\n--- ours\n%s\n--- pandas\n%s", s, c.Expect.Text)
				}
			default:
				if !fxSame(got, c.Expect.Value, Float64) {
					t.Errorf("%v, pandas %v", got, c.Expect.Value)
				}
			}
		})
	}
	for d := range devs {
		t.Logf("deliberate difference exercised: %s", d)
	}
}
