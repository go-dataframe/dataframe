package parquetio

import (
	"bytes"
	"errors"
	"fmt"
	"math"
	"slices"
	"testing"

	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/go-dataframe/dataframe"
	"github.com/go-dataframe/dataframe/arrowio"
)

// probeFrame has n rows of columns whose dictionaries do or do not
// overflow, for each type the probe knows.
func probeFrame(n int) *dataframe.DataFrame {
	i64 := make([]int64, n)
	u64 := make([]uint64, n)
	f64 := make([]float64, n)
	i32 := make([]int32, n)
	f32 := make([]float32, n)
	ts := make([]int64, n)
	low := make([]int64, n)
	late := make([]int64, n)
	mid := make([]int64, n)
	neg := make([]int64, n)
	zeros := make([]float64, n)
	bools := make([]bool, n)
	valid := make([]bool, n)
	for i := range n {
		i64[i] = int64(i) * 7919 % 1000003
		u64[i] = uint64(i) * 104729
		f64[i] = float64(i) / 3
		if i%5 == 0 {
			f64[i] = -float64(i)
		}
		i32[i] = int32(i)
		f32[i] = float32(i) / 2
		ts[i] = 1_600_000_000e9 + int64(i)*1e9
		low[i] = int64(i % 1000)
		// Repeats early on, distinct afterwards: the early look keeps
		// the dictionary, and the writer falls back as it always did.
		late[i] = int64(i)
		if i < 5000 {
			late[i] = int64(i % 10)
		}
		neg[i] = -int64(i)          // -1 is the probe set's all-ones key
		mid[i] = int64(i % 100_000) // distinct early, too few overall
		zeros[i] = math.Copysign(0, float64(i%2*2-1))
		bools[i] = i%2 == 0
		valid[i] = i%9 != 0
	}
	return dataframe.Must(dataframe.FromSeries(
		dataframe.Must(dataframe.NewSeries("i64", i64)),
		dataframe.Must(dataframe.NewSeries("u64", u64)),
		dataframe.Must(dataframe.NewSeries("f64", f64)),
		dataframe.Must(dataframe.NewSeriesMasked("i32", i32, valid)),
		dataframe.Must(dataframe.NewSeries("f32", f32)),
		dataframe.Must(dataframe.ToDatetime(dataframe.Must(dataframe.NewSeries("ts", ts)))),
		dataframe.Must(dataframe.NewSeries("low", low)),
		dataframe.Must(dataframe.NewSeries("late", late)),
		dataframe.Must(dataframe.NewSeries("neg", neg)),
		dataframe.Must(dataframe.NewSeries("mid", mid)),
		dataframe.Must(dataframe.NewSeries("zeros", zeros)),
		dataframe.Must(dataframe.NewSeries("bools", bools)),
	))
}

func TestPlainColumns(t *testing.T) {
	df := probeFrame(300_000)
	rec, err := arrowio.ToRecord(df)
	if err != nil {
		t.Fatal(err)
	}
	defer rec.Release()
	want := []string{"i64", "u64", "f64", "i32", "f32", "ts", "neg"}
	if got := plainColumns(rec, 1<<20); !slices.Equal(got, want) {
		t.Errorf("plainColumns = %v, want %v", got, want)
	}
	// Row groups too short to overflow keep every dictionary.
	if got := plainColumns(rec, 100_000); got != nil {
		t.Errorf("short row groups: %v", got)
	}
	// One short last row group is enough to keep a column's dictionary.
	if got := plainColumns(rec, 200_000); got != nil {
		t.Errorf("a short last row group: %v", got)
	}
	empty, _ := arrowio.ToRecord(df.Head(0))
	defer empty.Release()
	if got := plainColumns(empty, 1<<20); got != nil {
		t.Errorf("no rows: %v", got)
	}
	// Whatever the plan, the frame reads back the same, with every codec.
	for _, o := range []struct {
		c  Compression
		rg int64
	}{{Snappy, 1 << 20}, {Uncompressed, 1 << 20}, {Zstd, 200_000}} {
		equalFrames(t, fmt.Sprintf("%v rows/group %d", o.c, o.rg), roundTrip(t, df, WithCompression(o.c), WithRowGroupLength(o.rg)), df)
	}
}

func TestFloatKey(t *testing.T) {
	if floatKey(math.Copysign(0, -1)) != floatKey(0) {
		t.Error("-0 and 0 count apart")
	}
	if floatKey(math.NaN()) != floatKey(-math.NaN()) || floatKey(math.Float64frombits(0x7ff8000000000001)) != floatKey(math.NaN()) {
		t.Error("NaNs count apart")
	}
	if floatKey(1) == floatKey(2) {
		t.Error("1 and 2 count as one")
	}
}

// stringFrame has string columns that take each path of stringDict.
func stringFrame(n int) *dataframe.DataFrame {
	low := make([]string, n)
	uniq := make([]string, n)
	empty := make([]string, n)
	masked := make([]string, n)
	valid := make([]bool, n)
	for i := range n {
		low[i] = fmt.Sprintf("v%d", i%50)
		uniq[i] = fmt.Sprintf("unique-%08d", i)
		empty[i] = low[i]
		if i%7 == 3 {
			empty[i] = ""
		}
		masked[i] = low[i]
		valid[i] = i%4 != 0
	}
	return dataframe.Must(dataframe.FromSeries(
		dataframe.Must(dataframe.NewSeries("low", low)),
		dataframe.Must(dataframe.NewSeries("uniq", uniq)),
		dataframe.Must(dataframe.NewSeries("empty", empty)),
		dataframe.Must(dataframe.NewSeriesMasked("masked", masked, valid)),
	))
}

func TestStringDict(t *testing.T) {
	df := stringFrame(200_000)
	rec, err := arrowio.ToRecord(df)
	if err != nil {
		t.Fatal(err)
	}
	defer rec.Release()
	cols := rec.Columns()
	for i, want := range []bool{true, false, false, true} {
		d, ok := stringDict(cols[i].(*array.String), 1000, 150_000)
		if ok != want {
			t.Errorf("%s: dictionary %v, want %v", rec.ColumnName(i), ok, want)
			continue
		}
		if !ok {
			continue
		}
		dict := d.(*array.Dictionary)
		src := array.NewSlice(cols[i], 1000, 150_000).(*array.String)
		for k := range src.Len() {
			if src.IsNull(k) != dict.IsNull(k) || !src.IsNull(k) && dict.Dictionary().(*array.String).Value(dict.GetValueIndex(k)) != src.Value(k) {
				t.Fatalf("%s[%d] differs", rec.ColumnName(i), k)
			}
		}
		if n := dict.Dictionary().Len(); n != 50 {
			t.Errorf("%s: %d dictionary values, want 50", rec.ColumnName(i), n)
		}
		src.Release()
		d.Release()
	}
	for _, c := range []Compression{Snappy, Uncompressed} {
		for _, rg := range []int64{1 << 20, 30_000} {
			equalFrames(t, fmt.Sprintf("%v rows/group %d", c, rg), roundTrip(t, df, WithCompression(c), WithRowGroupLength(rg)), df)
		}
	}
}

// A file that fails while its row groups are written reports the failure,
// whichever call meets it.
func TestWriteRowGroupErrors(t *testing.T) {
	df := stringFrame(5000)
	var whole bytes.Buffer
	if err := WriteParquet(&whole, df, WithRowGroupLength(1000)); err != nil {
		t.Fatal(err)
	}
	for n := 0; n < whole.Len(); n += 997 {
		if err := WriteParquet(&failWriter{n}, df, WithRowGroupLength(1000)); !errors.Is(err, errWrite) {
			t.Fatalf("after %d of %d bytes: %v", n, whole.Len(), err)
		}
	}
}
