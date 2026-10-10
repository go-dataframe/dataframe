package arrowio

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"path/filepath"
	"strings"
	"testing"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/go-dataframe/dataframe"
)

func TestBadFixtures(t *testing.T) {
	for file, want := range map[string]error{
		"bad_list.feather":               ErrUnsupported,
		"bad_int_dictionary.feather":     ErrUnsupported,
		"bad_timestamp_overflow.feather": ErrOutOfRange,
	} {
		if _, err := ReadFeather(filepath.Join("testdata", file)); !errors.Is(err, want) {
			t.Errorf("%s: %v, want %v", file, err, want)
		}
	}
}

func complexFrame(t *testing.T) *dataframe.DataFrame {
	return dataframe.Must(dataframe.New(dataframe.Col("z", []complex128{1i})))
}

func TestUnsupportedDType(t *testing.T) {
	df := complexFrame(t)
	if _, err := ToRecord(df); !errors.Is(err, ErrUnsupported) {
		t.Fatalf("complex column: %v", err)
	}
	// A complex index level fails after the data columns were converted.
	ok := dataframe.Must(dataframe.New(dataframe.Col("x", []float64{1})))
	ix := dataframe.Must(dataframe.NewIndex("z", []complex64{1i}))
	if _, err := ToRecord(dataframe.Must(ok.WithIndex(ix))); !errors.Is(err, ErrUnsupported) {
		t.Fatalf("complex index: %v", err)
	}
	for _, c := range codecs {
		if err := c.write(&bytes.Buffer{}, df); !errors.Is(err, ErrUnsupported) {
			t.Errorf("%s: %v", c.name, err)
		}
	}
}

func TestBadCompression(t *testing.T) {
	df := allTypes(t)
	for _, c := range codecs {
		for _, comp := range []Compression{Compression(-1), Compression(42)} {
			if err := c.write(&bytes.Buffer{}, df, WithCompression(comp)); !errors.Is(err, ErrUnsupported) {
				t.Errorf("%s %v: %v", c.name, comp, err)
			}
		}
	}
	if s := Compression(-1).String(); s != "Compression(-1)" {
		t.Errorf("String = %q", s)
	}
	if s := LZ4.String(); s != "lz4" {
		t.Errorf("String = %q", s)
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
	for _, n := range []int{0, 8} {
		if err := WriteIPCFile(&failWriter{n}, df); !errors.Is(err, errWrite) {
			t.Errorf("ipc file after %d bytes: %v", n, err)
		}
		if err := WriteIPCStream(&failWriter{n}, df); !errors.Is(err, errWrite) {
			t.Errorf("ipc stream after %d bytes: %v", n, err)
		}
	}
	bad := filepath.Join(t.TempDir(), "missing", "x")
	if err := WriteFeather(bad, df); err == nil {
		t.Error("feather to a missing directory")
	}
	if err := WriteFeather(filepath.Join(t.TempDir(), "x"), complexFrame(t)); !errors.Is(err, ErrUnsupported) {
		t.Errorf("feather of a complex column: %v", err)
	}
}

// bodyFailReader serves the footer of a file (everything from footer on)
// but fails to read the body before it, so that opening succeeds and
// reading the data fails.
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
	df := allTypes(t)
	garbage := bytes.NewReader([]byte("this is not a columnar file at all"))
	if _, err := ReadIPCFile(garbage); err == nil {
		t.Error("ipc file of garbage")
	}
	if _, err := ReadIPCStream(strings.NewReader("")); err == nil {
		t.Error("empty ipc stream")
	}
	if _, err := ReadFeather(filepath.Join(t.TempDir(), "none")); err == nil {
		t.Error("missing file")
	}

	// A stream cut in the middle of a record batch.
	var b bytes.Buffer
	_ = WriteIPCStream(&b, df)
	if _, err := ReadIPCStream(bytes.NewReader(b.Bytes()[:b.Len()-100])); err == nil {
		t.Error("truncated ipc stream")
	}

	// An IPC file whose footer reads but whose record batch does not:
	// the footer length is the int32 before the trailing magic.
	// No dictionary column: dictionaries are read when the file opens.
	b.Reset()
	_ = WriteIPCFile(&b, dataframe.Must(df.Drop("cat")))
	data := b.Bytes()
	flen := int64(binary.LittleEndian.Uint32(data[len(data)-10:]))
	r := &bodyFailReader{bytes.NewReader(data), int64(len(data)) - 10 - flen}
	if _, err := ReadIPCFile(r); err == nil {
		t.Error("ipc file with an unreadable body")
	}
}

// record returns a one-column record of int64s with the given "pandas"
// metadata.
func record(t *testing.T, pandas string) arrow.RecordBatch {
	t.Helper()
	b := array.NewInt64Builder(memory.DefaultAllocator)
	defer b.Release()
	b.AppendValues([]int64{1, 2}, nil)
	col := b.NewArray()
	defer col.Release()
	md := arrow.NewMetadata([]string{"pandas"}, []string{pandas})
	schema := arrow.NewSchema([]arrow.Field{{Name: "k", Type: arrow.PrimitiveTypes.Int64, Nullable: true}}, &md)
	return array.NewRecordBatch(schema, []arrow.Array{col}, 2)
}

func TestPandasMetadata(t *testing.T) {
	for name, md := range map[string]string{
		"not json":      `{`,
		"number":        `{"index_columns": [3]}`,
		"other kind":    `{"index_columns": [{"kind": "list"}]}`,
		"range + field": `{"index_columns": [{"kind": "range", "start": 0, "stop": 2, "step": 1}, "k"]}`,
		"missing field": `{"index_columns": ["nope"]}`,
	} {
		rec := record(t, md)
		if _, err := FromRecord(rec); !errors.Is(err, ErrMetadata) {
			t.Errorf("%s: %v", name, err)
		}
		rec.Release()
	}
	// Level names: from the metadata (null is unnamed), else the field
	// name unless it is pandas' placeholder for an unnamed level.
	for md, want := range map[string]string{
		`{"index_columns": ["k"], "columns": [{"name": "key", "field_name": "k"}]}`: "key",
		`{"index_columns": ["k"], "columns": [{"name": null, "field_name": "k"}]}`:  "",
		`{"index_columns": ["k"], "columns": []}`:                                   "k",
		`{"index_columns": []}`: "",
	} {
		rec := record(t, md)
		df, err := FromRecord(rec)
		rec.Release()
		if err != nil || df.Index().Name() != want {
			t.Errorf("%s: index %v named %q, %v; want %q", md, df.Index(), df.Index().Name(), err, want)
		}
	}
	rec := record(t, `{"index_columns": []}`)
	defer rec.Release()
	if df, _ := FromRecord(rec); !df.Index().IsRange() || len(df.Columns()) != 1 {
		t.Errorf("no index columns: %v %v", df.Index(), df.Columns())
	}
	if got := levelName("__index_level_3__", nil); got != "" {
		t.Errorf("placeholder level name %q", got)
	}
}

func TestFromTable(t *testing.T) {
	df := allTypes(t)
	rec := dataframe.Must(ToRecord(df))
	defer rec.Release()
	tbl := array.NewTableFromRecords(rec.Schema(), []arrow.RecordBatch{rec, rec})
	defer tbl.Release()
	got := dataframe.Must(FromTable(tbl))
	if got.Len() != 2*df.Len() {
		t.Fatalf("%d rows", got.Len())
	}
	tail := got.Tail(df.Len())
	for i, s := range tail.Series() {
		equalItems(t, s.Name(), s.Items(), df.Series()[i].Items())
	}
}

// indexArray returns an Arrow integer array of type dt over values, the
// last one null.
func indexArray[T any](dt arrow.DataType, values []T) arrow.Array {
	valid := []uint64{1<<uint(len(values)-1) - 1}
	return fixed(dt, values, valid, len(values))
}

// Dictionary indices of every integer type are read; a dictionary may hold
// a null entry, and a value coded to it is missing, as is a null slot.
func TestDictionaryIndices(t *testing.T) {
	mem := memory.DefaultAllocator
	vb := array.NewStringBuilder(mem)
	defer vb.Release()
	vb.AppendValues([]string{"a", "", "b"}, []bool{true, false, true})
	dict := vb.NewArray()
	defer dict.Release()
	for _, idx := range []arrow.Array{
		indexArray(arrow.PrimitiveTypes.Int8, []int8{0, 1, 2, 2, -1}),
		indexArray(arrow.PrimitiveTypes.Int16, []int16{0, 1, 2, 2, -1}),
		indexArray(arrow.PrimitiveTypes.Int32, []int32{0, 1, 2, 2, -1}),
		indexArray(arrow.PrimitiveTypes.Int64, []int64{0, 1, 2, 2, -1}),
		indexArray(arrow.PrimitiveTypes.Uint8, []uint8{0, 1, 2, 2, 9}),
		indexArray(arrow.PrimitiveTypes.Uint16, []uint16{0, 1, 2, 2, 9}),
		indexArray(arrow.PrimitiveTypes.Uint32, []uint32{0, 1, 2, 2, 9}),
		indexArray(arrow.PrimitiveTypes.Uint64, []uint64{0, 1, 2, 2, 9}),
	} {
		dt := &arrow.DictionaryType{IndexType: idx.DataType(), ValueType: arrow.BinaryTypes.String}
		col := array.NewDictionaryArray(dt, idx, dict)
		schema := arrow.NewSchema([]arrow.Field{{Name: "d", Type: dt, Nullable: true}}, nil)
		rec := array.NewRecordBatch(schema, []arrow.Array{col}, 5)
		df := dataframe.Must(FromRecord(rec))
		equalItems(t, dt.String(), df.Series()[0].Items(), []any{"a", nil, "b", "b", nil})
		rec.Release()
		col.Release()
		idx.Release()
	}
}
