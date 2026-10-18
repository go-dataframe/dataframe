package parquetio

import (
	"bytes"
	"encoding/base64"
	"errors"
	"testing"

	"github.com/apache/arrow-go/v18/parquet/file"
	"github.com/apache/arrow-go/v18/parquet/metadata"
	"github.com/go-dataframe/dataframe"
	"github.com/go-dataframe/dataframe/arrowio"
)

// footer builds the metadata of a one-column, one-row-group file of size
// bytes whose chunk declares info, with the key-value metadata kv.
func footer(t *testing.T, info metadata.ChunkMetaInfo, rows int, size int64, kv ...string) *metadata.FileMetaData {
	t.Helper()
	var b bytes.Buffer
	df := dataframe.Must(dataframe.New(dataframe.Col("x", []int64{1})))
	if err := WriteParquet(&b, df, WithoutIndex()); err != nil {
		t.Fatal(err)
	}
	pf, err := file.NewParquetReader(bytes.NewReader(b.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	kvm := metadata.KeyValueMetadata{}
	for i := 0; i+1 < len(kv); i += 2 {
		_ = kvm.Append(kv[i], kv[i+1])
	}
	fb := metadata.NewFileMetadataBuilder(pf.MetaData().Schema, nil, kvm)
	rg := fb.AppendRowGroup()
	rg.SetNumRows(rows)
	if err := rg.NextColumnChunk().Finish(info, false, false, metadata.EncodingStats{}); err != nil {
		t.Fatal(err)
	}
	if err := rg.Finish(0, 0); err != nil {
		t.Fatal(err)
	}
	md, err := fb.Finish()
	if err != nil {
		t.Fatal(err)
	}
	md.SetSourceFileSize(size)
	return md
}

// A column chunk that does not lie within the file, or a row group of
// negative rows, is refused before any chunk is read.
func TestCheckChunks(t *testing.T) {
	ok := metadata.ChunkMetaInfo{NumValues: 1, DataPageOffset: 4, CompressedSize: 10, UncompressedSize: 10}
	if err := checkChunks(footer(t, ok, 1, 100)); err != nil {
		t.Fatalf("a chunk within the file: %v", err)
	}
	for name, c := range map[string]struct {
		info metadata.ChunkMetaInfo
		rows int
	}{
		"past the end":   {metadata.ChunkMetaInfo{DataPageOffset: 50, CompressedSize: 60}, 1},
		"negative size":  {metadata.ChunkMetaInfo{DataPageOffset: 4, CompressedSize: -1}, 1},
		"offset past":    {metadata.ChunkMetaInfo{DataPageOffset: 200, CompressedSize: 1}, 1},
		"dict page past": {metadata.ChunkMetaInfo{DataPageOffset: 90, DictPageOffset: 4, CompressedSize: 99}, 1},
		"negative rows":  {ok, -1},
	} {
		if err := checkChunks(footer(t, c.info, c.rows, 100)); !errors.Is(err, arrowio.ErrCorrupt) {
			t.Errorf("%s: %v", name, err)
		}
	}
}

// A stored Arrow schema that does not lie within its bytes is refused; no
// stored schema, or one that does not even decode, is pqarrow's affair.
func TestCheckArrowSchema(t *testing.T) {
	info := metadata.ChunkMetaInfo{DataPageOffset: 4, CompressedSize: 1}
	if err := checkArrowSchema(footer(t, info, 1, 100)); err != nil {
		t.Errorf("no ARROW:schema: %v", err)
	}
	if err := checkArrowSchema(footer(t, info, 1, 100, "ARROW:schema", "!!not base64!!")); err != nil {
		t.Errorf("an undecodable ARROW:schema: %v", err)
	}
	bad := base64.StdEncoding.EncodeToString([]byte{0xFF, 0xFF, 0xFF, 0xFF, 8, 0, 0, 0, 0xff, 0xff, 0xff, 0x7f, 0, 0, 0, 0})
	if err := checkArrowSchema(footer(t, info, 1, 100, "ARROW:schema", bad)); !errors.Is(err, arrowio.ErrCorrupt) {
		t.Errorf("a corrupt ARROW:schema: %v", err)
	}
}

// A file whose footer declares a chunk past its end is refused by
// ReadParquet before it reads any chunk.
func TestReadChecksFooter(t *testing.T) {
	md := footer(t, metadata.ChunkMetaInfo{DataPageOffset: 4, CompressedSize: 1 << 40}, 1, 0)
	var b bytes.Buffer
	b.WriteString("PAR1")
	n, err := md.WriteTo(&b, nil)
	if err != nil {
		t.Fatal(err)
	}
	b.Write([]byte{byte(n), byte(n >> 8), byte(n >> 16), byte(n >> 24)})
	b.WriteString("PAR1")
	if _, err := ReadParquet(bytes.NewReader(b.Bytes())); !errors.Is(err, arrowio.ErrCorrupt) {
		t.Errorf("a chunk past the end: %v", err)
	}
}

type panicky struct{ *bytes.Reader }

func (panicky) ReadAt([]byte, int64) (int, error) { panic("read at") }

// A panic of the decoders is an ErrCorrupt.
func TestReadPanics(t *testing.T) {
	if _, err := ReadParquet(panicky{bytes.NewReader(make([]byte, 64))}); !errors.Is(err, arrowio.ErrCorrupt) {
		t.Errorf("a panicking reader: %v", err)
	}
}
