package parquetio

// Test helpers shared with arrowio's tests (copied: test files cannot be
// imported across packages, and a helper package would be code to cover).

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/go-dataframe/dataframe"
)

// fixtures is where gen_fixtures.py writes, next to the IPC fixtures.
const fixtures = "../arrowio/testdata"

// The fixtures are written by pandas and pyarrow
// (../arrowio/testdata/gen_fixtures.py); expected.json says what each must
// read as.

type colSpec struct {
	Name   string
	DType  string
	Values []any
}

type fixture struct {
	Rows  int
	Index struct {
		Range  bool
		Name   string
		Levels []colSpec
	}
	Columns []colSpec
}

func loadExpected(t *testing.T) map[string]fixture {
	t.Helper()
	f, err := os.Open(filepath.Join(fixtures, "expected.json"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	dec := json.NewDecoder(f)
	dec.UseNumber()
	var e struct{ Fixtures map[string]fixture }
	if err := dec.Decode(&e); err != nil {
		t.Fatal(err)
	}
	return e.Fixtures
}

// sameValue compares a series item with an expected JSON value: null is
// missing, datetimes are int64 nanoseconds, numbers compare by value.
func sameValue(got, want any) bool {
	if got == nil || want == nil {
		return got == nil && want == nil
	}
	switch g := got.(type) {
	case time.Time:
		return fmt.Sprint(g.UnixNano()) == fmt.Sprint(want)
	case float32:
		got = float64(g)
	}
	if g, ok := got.(float64); ok {
		w, err := want.(json.Number).Float64()
		return err == nil && (g == w || math.IsNaN(g) && math.IsNaN(w))
	}
	return fmt.Sprint(got) == fmt.Sprint(want)
}

func checkSeries(t *testing.T, where string, s *dataframe.Series, spec colSpec) {
	t.Helper()
	if s.DType().String() != spec.DType {
		t.Errorf("%s: dtype %v, want %s", where, s.DType(), spec.DType)
		return
	}
	items := s.Items()
	if len(items) != len(spec.Values) {
		t.Errorf("%s: %d values, want %d", where, len(items), len(spec.Values))
		return
	}
	for i, v := range items {
		if !sameValue(v, spec.Values[i]) {
			t.Errorf("%s[%d] = %v (%T), want %v", where, i, v, v, spec.Values[i])
		}
	}
}

func checkFixture(t *testing.T, file string, df *dataframe.DataFrame, fx fixture) {
	t.Helper()
	if df.Len() != fx.Rows {
		t.Errorf("%s: %d rows, want %d", file, df.Len(), fx.Rows)
	}
	cols := df.Series()
	if len(cols) != len(fx.Columns) {
		t.Fatalf("%s: columns %v, want %d", file, df.Columns(), len(fx.Columns))
	}
	for i, spec := range fx.Columns {
		if cols[i].Name() != spec.Name {
			t.Errorf("%s: column %d is %q, want %q", file, i, cols[i].Name(), spec.Name)
		}
		checkSeries(t, file+":"+spec.Name, cols[i], spec)
	}
	ix := df.Index()
	if fx.Index.Range {
		if !ix.IsRange() || ix.Name() != fx.Index.Name {
			t.Errorf("%s: index %v named %q, want a range index named %q", file, ix, ix.Name(), fx.Index.Name)
		}
		return
	}
	if ix.IsRange() || ix.NLevels() != len(fx.Index.Levels) {
		t.Fatalf("%s: index %v, want %d levels", file, ix, len(fx.Index.Levels))
	}
	for k, spec := range fx.Index.Levels {
		lv, _ := ix.Level(k)
		if lv.Name() != spec.Name {
			t.Errorf("%s: level %d named %q, want %q", file, k, lv.Name(), spec.Name)
		}
		checkSeries(t, fmt.Sprintf("%s:level %d", file, k), lv, spec)
	}
}

// allTypes returns a frame of every dtype that has an Arrow mapping, with
// missing values in each: masked integers, bools, strings, datetimes and
// categories, NaN floats.
func allTypes(t *testing.T) *dataframe.DataFrame {
	t.Helper()
	valid := []bool{true, false, true, true, false, true, true, true, true, true}
	nan := math.NaN()
	cols := []*dataframe.Series{
		dataframe.Must(dataframe.NewSeries("f64", []float64{1.5, nan, -2, math.Inf(1), 0, 1e300, -1e-300, 3, 4, 5})),
		dataframe.Must(dataframe.NewSeries("f32", []float32{1.5, float32(nan), -2, 0, 1, 2, 3, 4, 5, 6})),
		dataframe.Must(dataframe.NewSeriesMasked("f64m", []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, valid)),
		dataframe.Must(dataframe.NewSeriesMasked("i8", []int8{math.MinInt8, 0, math.MaxInt8, 1, 2, 3, 4, 5, 6, 7}, valid)),
		dataframe.Must(dataframe.NewSeries("i16", []int16{math.MinInt16, 0, math.MaxInt16, 1, 2, 3, 4, 5, 6, 7})),
		dataframe.Must(dataframe.NewSeriesMasked("i32", []int32{math.MinInt32, 0, math.MaxInt32, 1, 2, 3, 4, 5, 6, 7}, valid)),
		dataframe.Must(dataframe.NewSeries("i64", []int64{math.MinInt64, 0, math.MaxInt64, 1, 2, 3, 4, 5, 6, 7})),
		dataframe.Must(dataframe.NewSeriesMasked("u8", []uint8{0, 1, math.MaxUint8, 1, 2, 3, 4, 5, 6, 7}, valid)),
		dataframe.Must(dataframe.NewSeries("u16", []uint16{0, 1, math.MaxUint16, 1, 2, 3, 4, 5, 6, 7})),
		dataframe.Must(dataframe.NewSeriesMasked("u32", []uint32{0, 1, math.MaxUint32, 1, 2, 3, 4, 5, 6, 7}, valid)),
		dataframe.Must(dataframe.NewSeries("u64", []uint64{0, 1, math.MaxUint64, 1, 2, 3, 4, 5, 6, 7})),
		dataframe.Must(dataframe.NewSeries("b", []bool{true, false, true, true, false, false, true, false, true, true})),
		dataframe.Must(dataframe.NewSeriesMasked("bm", []bool{true, false, true, true, false, false, true, false, true, true}, valid)),
		dataframe.Must(dataframe.NewSeriesMasked("s", []string{"alpha", "", "", "ünïcödé", "x", "日本", "a\x00b", "c", "d", "e"}, valid)),
		dataframe.Must(dataframe.NewCategorical("cat", []int32{0, 1, -1, 2, 0, 0, 1, 2, -1, 0}, []string{"lo", "mid", "hi"})),
		dataframe.Must(dataframe.ToDatetime(dataframe.Must(dataframe.NewSeriesMasked("ts", []int64{0, 1, -1, math.MaxInt64, math.MinInt64 + 1, 1577836800e9, 7, 8, 9, 10}, valid)))),
	}
	return dataframe.Must(dataframe.FromSeries(cols...))
}

// equalFrames compares names, dtypes, values (missing as missing), and the
// index: its kind, level names and labels.
func equalFrames(t *testing.T, where string, got, want *dataframe.DataFrame) {
	t.Helper()
	if !slices.Equal(got.Columns(), want.Columns()) {
		t.Fatalf("%s: columns %v, want %v", where, got.Columns(), want.Columns())
	}
	if !slices.Equal(got.DTypes(), want.DTypes()) {
		t.Fatalf("%s: dtypes %v, want %v", where, got.DTypes(), want.DTypes())
	}
	gs, ws := got.Series(), want.Series()
	for i := range gs {
		equalItems(t, where+":"+gs[i].Name(), gs[i].Items(), ws[i].Items())
	}
	gi, wi := got.Index(), want.Index()
	if gi.IsRange() != wi.IsRange() || !slices.Equal(gi.Names(), wi.Names()) {
		t.Fatalf("%s: index %v %q, want %v %q", where, gi, gi.Names(), wi, wi.Names())
	}
	for k := range wi.NLevels() {
		g, _ := gi.Level(k)
		w, _ := wi.Level(k)
		if g.DType() != w.DType() {
			t.Fatalf("%s: index level %d is %v, want %v", where, k, g.DType(), w.DType())
		}
		equalItems(t, fmt.Sprintf("%s:level %d", where, k), g.Items(), w.Items())
	}
}

func equalItems(t *testing.T, where string, got, want []any) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d values, want %d", where, len(got), len(want))
	}
	for i := range got {
		if fmt.Sprint(got[i]) != fmt.Sprint(want[i]) {
			t.Errorf("%s[%d] = %v, want %v", where, i, got[i], want[i])
		}
	}
}
