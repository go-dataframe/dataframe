package arrowio

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/go-dataframe/dataframe"
)

// The fixtures in testdata are written by pandas and pyarrow
// (gen_fixtures.py); expected.json says what each must read as.

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
	f, err := os.Open(filepath.Join("testdata", "expected.json"))
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

type reader func(path string) (*dataframe.DataFrame, error)

func readStreamFile(path string) (*dataframe.DataFrame, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return ReadIPCStream(f)
}

func readFeather(path string) (*dataframe.DataFrame, error) { return ReadFeather(path) }

func TestFixtures(t *testing.T) {
	ex := loadExpected(t)
	byExt := map[string]reader{".feather": readFeather, ".arrow": readFeather, ".arrows": readStreamFile}
	files := map[string][]string{
		"types":             {"types.feather", "types.arrows", "types_zstd.feather", "types_none.feather"},
		"named_index":       {"named_index.feather", "named_index.arrows"},
		"unnamed_index":     {"unnamed_index.feather", "unnamed_index.arrows"},
		"range_index":       {"range_index.feather", "range_index.arrows"},
		"named_range_index": {"named_range_index.feather", "named_range_index.arrows"},
		"multi_index":       {"multi_index.feather", "multi_index.arrows"},
		"datetime_index":    {"datetime_index.feather", "datetime_index.arrows"},
		"empty":             {"empty.feather", "empty.arrows"},
		"batches":           {"batches.arrow", "batches.arrows"},
	}
	if len(files) != len(ex) {
		t.Fatalf("%d fixtures tested, %d in expected.json", len(files), len(ex))
	}
	for name, list := range files {
		fx, ok := ex[name]
		if !ok {
			t.Fatalf("fixture %q not in expected.json", name)
		}
		for _, file := range list {
			t.Run(file, func(t *testing.T) {
				df, err := byExt[filepath.Ext(file)](filepath.Join("testdata", file))
				if err != nil {
					t.Fatal(err)
				}
				checkFixture(t, file, df, fx)
			})
		}
	}
}
