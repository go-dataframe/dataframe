package parquetio

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	df "github.com/go-dataframe/dataframe"
	"github.com/go-dataframe/dataframe/arrowio"
)

// TestWriteReadback writes the files testdata/check_readback.py reads with
// pandas, to check that pandas reads what we write with the right dtypes
// and index. It is a manual check, not run in CI (it needs Python): it
// does nothing unless DATAFRAME_READBACK_DIR names a directory.
//
//	DATAFRAME_READBACK_DIR=/tmp/rb go test -run WriteReadback ./parquetio
//	uv run --with pandas --with numpy --with pyarrow python parquetio/testdata/check_readback.py /tmp/rb
func TestWriteReadback(t *testing.T) {
	out := os.Getenv("DATAFRAME_READBACK_DIR")
	if out == "" {
		t.Skip("set DATAFRAME_READBACK_DIR to write the pandas readback files")
	}
	valid := []bool{true, false, true}
	cols := []*df.Series{
		df.Must(df.NewSeries("f64", []float64{1.5, math.NaN(), 3})),
		df.Must(df.NewSeries("f32", []float32{1, 2, 3})),
		df.Must(df.NewSeries("i8", []int8{-1, 0, 1})),
		df.Must(df.NewSeriesMasked("i64m", []int64{1, 2, 3}, valid)),
		df.Must(df.NewSeriesMasked("u16m", []uint16{1, 2, 3}, valid)),
		df.Must(df.NewSeries("u64", []uint64{0, 1, math.MaxUint64})),
		df.Must(df.NewSeries("b", []bool{true, false, true})),
		df.Must(df.NewSeriesMasked("bm", []bool{true, false, true}, valid)),
		df.Must(df.NewSeriesMasked("s", []string{"a", "", "ü"}, valid)),
		df.Must(df.NewCategorical("cat", []int32{1, -1, 0}, []string{"lo", "hi"})),
		df.Must(df.ToDatetime(df.Must(df.NewSeriesMasked("ts", []int64{1577836800e9, 0, -1}, valid)))),
		df.Must(df.NewSeries("key", []string{"x", "y", "z"})),
		df.Must(df.NewSeries("lvl", []int64{7, 8, 9})),
	}
	base := df.Must(df.FromSeries(cols...))
	multi := df.Must(base.SetIndex("key", "lvl"))
	frames := map[string]*df.DataFrame{
		"range":   df.Must(base.Drop("key", "lvl")),
		"named":   df.Must(base.SetIndex("key")),
		"multi":   df.Must(multi.WithIndex(df.Must(multi.Index().Rename("key", "")))),
		"stepped": df.Must(df.Must(base.Drop("key", "lvl")).Iloc(df.Pos(0, 2))),
		"dtindex": df.Must(base.SetIndex("ts")),
	}
	for name, f := range frames {
		p := filepath.Join(out, name)
		if err := WriteParquetFile(p+".parquet", f); err != nil {
			t.Fatal(err)
		}
		if err := arrowio.WriteFeather(p+".feather", f, arrowio.WithCompression(arrowio.LZ4)); err != nil {
			t.Fatal(err)
		}
		w, err := os.Create(p + ".arrows")
		if err != nil {
			t.Fatal(err)
		}
		if err := arrowio.WriteIPCStream(w, f); err != nil {
			t.Fatal(err)
		}
		w.Close()
	}
	if err := WriteParquetFile(filepath.Join(out, "naive.parquet"), frames["range"], WithTimeZone("")); err != nil {
		t.Fatal(err)
	}
}
