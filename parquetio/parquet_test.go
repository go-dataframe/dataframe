package parquetio

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"path/filepath"
	"slices"
	"testing"

	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/go-dataframe/dataframe"
	"github.com/go-dataframe/dataframe/arrowio"
)

func TestFixtures(t *testing.T) {
	ex := loadExpected(t)
	files := map[string][]string{
		"types":             {"types.parquet", "types_zstd.parquet", "types_gzip.parquet", "types_none.parquet"},
		"named_index":       {"named_index.parquet"},
		"unnamed_index":     {"unnamed_index.parquet"},
		"range_index":       {"range_index.parquet"},
		"named_range_index": {"named_range_index.parquet"},
		"multi_index":       {"multi_index.parquet"},
		"datetime_index":    {"datetime_index.parquet"},
		"empty":             {"empty.parquet"},
		"batches":           {"batches.parquet"},
	}
	if len(files) != len(ex) {
		t.Fatalf("%d fixtures tested, %d in expected.json", len(files), len(ex))
	}
	for name, list := range files {
		for _, file := range list {
			t.Run(file, func(t *testing.T) {
				df, err := ReadParquetFile(filepath.Join(fixtures, file))
				if err != nil {
					t.Fatal(err)
				}
				checkFixture(t, file, df, ex[name])
			})
		}
	}
}

func roundTrip(t *testing.T, df *dataframe.DataFrame, opts ...Option) *dataframe.DataFrame {
	t.Helper()
	var b bytes.Buffer
	if err := WriteParquet(&b, df, opts...); err != nil {
		t.Fatalf("write: %v", err)
	}
	got, err := ReadParquet(bytes.NewReader(b.Bytes()))
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	return got
}

func TestRoundTrip(t *testing.T) {
	df := allTypes(t)
	equalFrames(t, "default", roundTrip(t, df), df)
	// Several row groups, so several chunks per column.
	equalFrames(t, "small row groups", roundTrip(t, df, WithRowGroupLength(3)), df)
	for _, c := range []Compression{Snappy, Uncompressed, Gzip, Brotli, Zstd, LZ4} {
		equalFrames(t, c.String(), roundTrip(t, df, WithCompression(c)), df)
	}
	for _, tz := range []string{"", "Europe/Paris"} {
		equalFrames(t, "tz "+tz, roundTrip(t, df, WithTimeZone(tz)), df)
	}
}

func TestRoundTripIndexes(t *testing.T) {
	base := allTypes(t)
	multi := dataframe.Must(base.SetIndex("i64", "s"))
	cases := map[string]*dataframe.DataFrame{
		"named range": dataframe.Must(base.WithIndex(dataframe.Must(base.Index().Rename("row")))),
		"string":      dataframe.Must(base.SetIndex("s")),
		"datetime":    dataframe.Must(base.SetIndex("ts")),
		"multi":       dataframe.Must(multi.WithIndex(dataframe.Must(multi.Index().Rename("", "s")))),
		"empty":       base.Head(0),
	}
	for name, df := range cases {
		equalFrames(t, name, roundTrip(t, df), df)
	}
	stepped := dataframe.Must(base.Iloc(dataframe.Pos(1, 4, 7)))
	got := roundTrip(t, stepped)
	equalItems(t, "stepped", got.Index().Labels(), []any{int64(1), int64(4), int64(7)})
	got = roundTrip(t, cases["string"], WithoutIndex())
	if !got.Index().IsRange() || slices.Contains(got.Columns(), "s") {
		t.Fatalf("WithoutIndex: index %v, columns %v", got.Index(), got.Columns())
	}
}

func TestFiles(t *testing.T) {
	df := dataframe.Must(allTypes(t).SetIndex("s"))
	path := filepath.Join(t.TempDir(), "x.parquet")
	if err := WriteParquetFile(path, df); err != nil {
		t.Fatal(err)
	}
	equalFrames(t, "file", dataframe.Must(ReadParquetFile(path)), df)
}

// Every buffer is released: a checked allocator ends at zero.
func TestAllocator(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.NewGoAllocator())
	defer mem.AssertSize(t, 0)
	df := allTypes(t)
	var b bytes.Buffer
	if err := WriteParquet(&b, df, WithAllocator(mem), WithCompression(Zstd)); err != nil {
		t.Fatal(err)
	}
	equalFrames(t, "checked", dataframe.Must(ReadParquet(bytes.NewReader(b.Bytes()), WithAllocator(mem))), df)
}

func TestCompressionNames(t *testing.T) {
	if s := LZ4.String(); s != "lz4" {
		t.Errorf("String = %q", s)
	}
	if s := Compression(-1).String(); s != "Compression(-1)" {
		t.Errorf("String = %q", s)
	}
	for _, c := range []Compression{-1, 42} {
		if err := WriteParquet(&bytes.Buffer{}, allTypes(t), WithCompression(c)); !errors.Is(err, arrowio.ErrUnsupported) {
			t.Errorf("%v: %v", c, err)
		}
	}
}

// failWriter accepts n bytes, then fails.
type failWriter struct{ n int }

var errWrite = errors.New("write failed")

func (w *failWriter) Write(p []byte) (int, error) {
	if len(p) > w.n {
		return 0, errWrite
	}
	w.n -= len(p)
	return len(p), nil
}

func TestWriteErrors(t *testing.T) {
	df := allTypes(t)
	// Failing on the magic number, then on the first row group.
	for _, n := range []int{0, 8} {
		if err := WriteParquet(&failWriter{n}, df); err == nil {
			t.Errorf("after %d bytes: no error", n)
		}
	}
	complexFrame := dataframe.Must(dataframe.New(dataframe.Col("z", []complex128{1i})))
	if err := WriteParquet(&bytes.Buffer{}, complexFrame); !errors.Is(err, arrowio.ErrUnsupported) {
		t.Errorf("complex column: %v", err)
	}
	if err := WriteParquetFile(filepath.Join(t.TempDir(), "x"), complexFrame); !errors.Is(err, arrowio.ErrUnsupported) {
		t.Errorf("complex column to a file: %v", err)
	}
	if err := WriteParquetFile(filepath.Join(t.TempDir(), "missing", "x"), df); err == nil {
		t.Error("a file in a missing directory")
	}
}

// bodyFailReader serves the footer but fails to read the data before it,
// so that opening succeeds and reading the columns fails.
type bodyFailReader struct {
	*bytes.Reader
	footer int64
}

func (r *bodyFailReader) ReadAt(p []byte, off int64) (int, error) {
	if off < r.footer {
		return 0, io.ErrUnexpectedEOF
	}
	return r.Reader.ReadAt(p, off)
}

func TestReadErrors(t *testing.T) {
	if _, err := ReadParquet(bytes.NewReader([]byte("not a parquet file at all"))); err == nil {
		t.Error("garbage")
	}
	if _, err := ReadParquetFile(filepath.Join(t.TempDir(), "none")); err == nil {
		t.Error("missing file")
	}
	// The footer length is the int32 before the trailing "PAR1".
	var b bytes.Buffer
	_ = WriteParquet(&b, allTypes(t))
	data := b.Bytes()
	flen := int64(binary.LittleEndian.Uint32(data[len(data)-8:]))
	r := &bodyFailReader{bytes.NewReader(data), int64(len(data)) - 8 - flen}
	if _, err := ReadParquet(r); err == nil {
		t.Error("an unreadable body")
	}
	// A type with no mapping fails the conversion after the decoding.
	if _, err := ReadParquetFile(filepath.Join(fixtures, "bad_list.parquet")); !errors.Is(err, arrowio.ErrUnsupported) {
		t.Errorf("list column: %v", err)
	}
}
