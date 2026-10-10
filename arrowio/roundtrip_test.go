package arrowio

import (
	"bytes"
	"fmt"
	"math"
	"path/filepath"
	"slices"
	"testing"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/go-dataframe/dataframe"
)

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

type codec struct {
	name  string
	write func(*bytes.Buffer, *dataframe.DataFrame, ...Option) error
	read  func(*bytes.Reader) (*dataframe.DataFrame, error)
}

var codecs = []codec{
	{"ipc-file", func(b *bytes.Buffer, df *dataframe.DataFrame, o ...Option) error { return WriteIPCFile(b, df, o...) },
		func(r *bytes.Reader) (*dataframe.DataFrame, error) { return ReadIPCFile(r) }},
	{"ipc-stream", func(b *bytes.Buffer, df *dataframe.DataFrame, o ...Option) error { return WriteIPCStream(b, df, o...) },
		func(r *bytes.Reader) (*dataframe.DataFrame, error) { return ReadIPCStream(r) }},
}

func roundTrip(t *testing.T, c codec, df *dataframe.DataFrame, opts ...Option) *dataframe.DataFrame {
	t.Helper()
	var b bytes.Buffer
	if err := c.write(&b, df, opts...); err != nil {
		t.Fatalf("%s: write: %v", c.name, err)
	}
	got, err := c.read(bytes.NewReader(b.Bytes()))
	if err != nil {
		t.Fatalf("%s: read: %v", c.name, err)
	}
	return got
}

func TestRoundTripTypes(t *testing.T) {
	df := allTypes(t)
	for _, c := range codecs {
		equalFrames(t, c.name, roundTrip(t, c, df), df)
	}
}

func TestRoundTripCompression(t *testing.T) {
	df := allTypes(t)
	for _, comp := range []Compression{Uncompressed, Zstd, LZ4} {
		for _, c := range codecs {
			equalFrames(t, c.name+" "+comp.String(), roundTrip(t, c, df, WithCompression(comp)), df)
		}
	}
}

func TestRoundTripIndexes(t *testing.T) {
	base := allTypes(t)
	named := dataframe.Must(base.SetIndex("s"))
	ix := dataframe.Must(base.Index().Rename("row"))
	cases := map[string]*dataframe.DataFrame{
		"named range":   dataframe.Must(base.WithIndex(ix)),
		"string index":  named,
		"datetime":      dataframe.Must(base.SetIndex("ts")),
		"category":      dataframe.Must(base.SetIndex("cat")),
		"multi":         dataframe.Must(base.SetIndex("i64", "s")),
		"unnamed multi": dataframe.Must(dataframe.Must(base.SetIndex("i64", "u8")).WithIndex(dataframe.Must(dataframe.Must(base.SetIndex("i64", "u8")).Index().Rename("", "")))),
		"empty":         base.Head(0),
		"one row":       base.Head(1),
	}
	unnamed := dataframe.Must(dataframe.NewIndex("", []int64{9, 8, 7, 6, 5, 4, 3, 2, 1, 0}))
	cases["unnamed"] = dataframe.Must(base.WithIndex(unnamed))
	for name, df := range cases {
		for _, c := range codecs {
			equalFrames(t, c.name+" "+name, roundTrip(t, c, df), df)
		}
	}
}

// A range index that does not start at zero, or steps by more than one,
// is stored as a range descriptor and comes back as labels.
func TestRoundTripRangeDescriptor(t *testing.T) {
	base := allTypes(t)
	stepped := dataframe.Must(base.Iloc(dataframe.Pos(1, 4, 7)))
	if !stepped.Index().IsRange() {
		t.Fatal("Iloc of an arithmetic progression is not a range")
	}
	for _, c := range codecs {
		got := roundTrip(t, c, stepped)
		if got.Index().IsRange() {
			t.Fatalf("%s: got a range index", c.name)
		}
		equalItems(t, c.name, got.Index().Labels(), []any{int64(1), int64(4), int64(7)})
		equalItems(t, c.name, got.Series()[0].Items(), stepped.Series()[0].Items())
	}
}

func TestWithoutIndex(t *testing.T) {
	df := dataframe.Must(allTypes(t).SetIndex("s"))
	for _, c := range codecs {
		got := roundTrip(t, c, df, WithoutIndex())
		if !got.Index().IsRange() || slices.Contains(got.Columns(), "s") {
			t.Fatalf("%s: index %v, columns %v", c.name, got.Index(), got.Columns())
		}
	}
}

func TestTimeZone(t *testing.T) {
	df := allTypes(t)
	for _, tz := range []string{"", "UTC", "Europe/Paris"} {
		rec := dataframe.Must(ToRecord(df, WithTimeZone(tz)))
		f, _ := rec.Schema().FieldsByName("ts")
		if got := f[0].Type.(*arrow.TimestampType).TimeZone; got != tz {
			t.Errorf("zone %q, want %q", got, tz)
		}
		rec.Release()
		for _, c := range codecs {
			equalFrames(t, c.name+" tz "+tz, roundTrip(t, c, df, WithTimeZone(tz)), df)
		}
	}
}

// Missing values travel in the validity bitmaps, and NaN is written as
// null, as pandas writes it.
func TestRecordNulls(t *testing.T) {
	rec := dataframe.Must(ToRecord(allTypes(t)))
	defer rec.Release()
	want := map[string]int{"f64": 1, "f32": 1, "f64m": 2, "i8": 2, "i16": 0, "bm": 2, "s": 2, "cat": 2, "ts": 2, "b": 0}
	for i, f := range rec.Schema().Fields() {
		if n, ok := want[f.Name]; ok && rec.Column(i).NullN() != n {
			t.Errorf("%s: %d nulls, want %d", f.Name, rec.Column(i).NullN(), n)
		}
	}
	df := dataframe.Must(FromRecord(rec))
	equalFrames(t, "FromRecord", df, allTypes(t))
}

// A masked series whose mask has no cleared bit (a head before the first
// missing value) still writes a validity buffer, with no nulls.
func TestMaskWithoutMissing(t *testing.T) {
	df := allTypes(t).Head(1)
	rec := dataframe.Must(ToRecord(df))
	defer rec.Release()
	for i := range rec.Columns() {
		if rec.Column(i).NullN() != 0 {
			t.Errorf("%s: %d nulls", rec.ColumnName(i), rec.Column(i).NullN())
		}
	}
}

// Strings beyond 32-bit offsets are written as large_utf8; the threshold
// is lowered to take that path.
func TestLargeStrings(t *testing.T) {
	defer func(n int) { largeStrings = n }(largeStrings)
	largeStrings = 0
	df := allTypes(t)
	rec := dataframe.Must(ToRecord(df))
	f, _ := rec.Schema().FieldsByName("s")
	if f[0].Type.ID() != arrow.LARGE_STRING {
		t.Fatalf("type %v", f[0].Type)
	}
	rec.Release()
	for _, c := range codecs {
		equalFrames(t, c.name+" large", roundTrip(t, c, df), df)
	}
}

func TestFiles(t *testing.T) {
	df := dataframe.Must(allTypes(t).SetIndex("s"))
	dir := t.TempDir()
	fe := filepath.Join(dir, "x.feather")
	if err := WriteFeather(fe, df, WithCompression(LZ4)); err != nil {
		t.Fatal(err)
	}
	equalFrames(t, "feather file", dataframe.Must(ReadFeather(fe)), df)
}

// Every buffer is released: a checked allocator ends at zero.
func TestAllocator(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.NewGoAllocator())
	defer mem.AssertSize(t, 0)
	df := allTypes(t)
	for _, c := range codecs {
		var b bytes.Buffer
		if err := c.write(&b, df, WithAllocator(mem), WithCompression(Zstd)); err != nil {
			t.Fatal(err)
		}
	}
	var b bytes.Buffer
	_ = WriteIPCFile(&b, df, WithAllocator(mem))
	equalFrames(t, "ipc file", dataframe.Must(ReadIPCFile(bytes.NewReader(b.Bytes()), WithAllocator(mem))), df)
	b.Reset()
	_ = WriteIPCStream(&b, df, WithAllocator(mem))
	equalFrames(t, "ipc stream", dataframe.Must(ReadIPCStream(bytes.NewReader(b.Bytes()), WithAllocator(mem))), df)
}

// A sliced record starts at an offset into its buffers, bitmaps included.
func TestSlicedRecord(t *testing.T) {
	df := allTypes(t)
	rec := dataframe.Must(ToRecord(df, WithoutIndex()))
	defer rec.Release()
	for _, r := range [][2]int64{{1, 7}, {3, 10}, {0, 0}} {
		sl := rec.NewSlice(r[0], r[1])
		got := dataframe.Must(FromRecords(sl.Schema(), []arrow.RecordBatch{sl, sl}))
		sl.Release()
		w := dataframe.Must(df.Iloc(dataframe.Slice(int(r[0]), int(r[1]))))
		for i, s := range got.Series() {
			n := w.Len()
			equalItems(t, fmt.Sprint(r, s.Name()), s.Head(n).Items(), w.Series()[i].Items())
			equalItems(t, fmt.Sprint(r, s.Name(), " again"), s.Tail(n).Items(), w.Series()[i].Items())
		}
	}
}
