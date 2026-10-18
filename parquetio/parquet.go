package parquetio

import (
	"context"
	"encoding/base64"
	"fmt"
	"io"
	"os"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/parquet"
	"github.com/apache/arrow-go/v18/parquet/compress"
	"github.com/apache/arrow-go/v18/parquet/file"
	"github.com/apache/arrow-go/v18/parquet/metadata"
	"github.com/apache/arrow-go/v18/parquet/pqarrow"
	"github.com/go-dataframe/dataframe"
	"github.com/go-dataframe/dataframe/arrowio"
	"github.com/go-dataframe/dataframe/internal/arrowfb"
	"github.com/go-dataframe/dataframe/internal/rawcol"
)

// WriteParquet writes a frame as a Parquet file, laid out as
// arrowio.ToRecord lays it out (index columns and "pandas" metadata
// included), with the Arrow schema stored in the file so that a reader gets
// the exact Arrow types back (dictionary columns, nanosecond timestamps
// with their zone). The writer is not closed.
func WriteParquet(w io.Writer, df *dataframe.DataFrame, opts ...Option) error {
	cfg := newConfig(opts)
	codec, err := cfg.compression.codec()
	if err != nil {
		return err
	}
	rec, err := arrowio.ToRecord(df, cfg.record...)
	if err != nil {
		return err
	}
	defer rec.Release()
	pprops := []parquet.WriterProperty{
		parquet.WithCompression(codec),
		parquet.WithMaxRowGroupLength(cfg.rowGroup),
		parquet.WithAllocator(cfg.mem),
	}
	tuned := codec != compress.Codecs.Uncompressed // see plan.go
	if tuned {
		for _, name := range plainColumns(rec, cfg.rowGroup) {
			pprops = append(pprops, parquet.WithDictionaryPath(parquet.ColumnPath{name}, false))
		}
	}
	props := parquet.NewWriterProperties(pprops...)
	aprops := pqarrow.NewArrowWriterProperties(pqarrow.WithStoreSchema(), pqarrow.WithAllocator(cfg.mem))
	// The Parquet writer closes a sink that is an io.Closer; the caller's
	// writer stays the caller's to close.
	fw, err := pqarrow.NewFileWriter(rec.Schema(), struct{ io.Writer }{w}, props, aprops)
	if err != nil {
		return err
	}
	if err := writeRowGroups(fw, rec, cfg.rowGroup, tuned); err != nil {
		fw.Close()
		return err
	}
	return fw.Close()
}

// writeRowGroups writes rec as fw.Write does, one row group per group rows
// (and one empty row group for no rows), except that with tuned, each
// string chunk whose distinct values fit the dictionary goes to the writer
// as a dictionary array (see plan.go). The file's Arrow schema is rec's
// whatever the arrays, so that the column reads back as strings.
func writeRowGroups(fw *pqarrow.FileWriter, rec arrow.RecordBatch, group int64, tuned bool) error {
	n := rec.NumRows()
	for lo := int64(0); lo < n || lo == 0; lo += group {
		hi := min(lo+group, n)
		if err := fw.NewRowGroupChecked(); err != nil {
			return err
		}
		for _, col := range rec.Columns() {
			if err := writeChunk(fw, col, lo, hi, tuned); err != nil {
				return err
			}
		}
	}
	return nil
}

// writeChunk writes rows [lo, hi) of col as the next column of the row
// group.
func writeChunk(fw *pqarrow.FileWriter, col arrow.Array, lo, hi int64, tuned bool) error {
	if s, ok := col.(*array.String); ok && tuned && hi > lo {
		if d, ok := stringDict(s, int(lo), int(hi)); ok {
			defer d.Release()
			return fw.WriteColumnData(d)
		}
	}
	part := array.NewSlice(col, lo, hi)
	defer part.Release()
	return fw.WriteColumnData(part)
}

// WriteParquetFile writes a frame to path as WriteParquet does.
func WriteParquetFile(path string, df *dataframe.DataFrame, opts ...Option) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := WriteParquet(f, df, opts...); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// ReadParquet reads a Parquet file, every row group, decoding the columns in
// parallel, and restores the index from "pandas" metadata as
// arrowio.FromRecord does. The only option it takes is WithAllocator.
func ReadParquet(r arrowio.ReadAtSeeker, opts ...Option) (df *dataframe.DataFrame, err error) {
	// The Parquet decoders index buffers by what the file declares; a panic
	// of theirs on a malformed file is an arrowio.ErrCorrupt.
	defer func() {
		if p := recover(); p != nil {
			df, err = nil, fmt.Errorf("%w: %v", arrowio.ErrCorrupt, p)
		}
	}()
	return readParquet(r, opts)
}

func readParquet(r arrowio.ReadAtSeeker, opts []Option) (*dataframe.DataFrame, error) {
	cfg := newConfig(opts)
	pf, err := file.NewParquetReader(r, file.WithReadProps(parquet.NewReaderProperties(cfg.mem)))
	if err != nil {
		return nil, err
	}
	if err := checkFooter(pf.MetaData()); err != nil {
		return nil, err
	}
	// A string column stored dictionary-encoded throughout is read as
	// dictionary arrays (the indices, and each distinct value once) rather
	// than have every value's bytes copied into one buffer, then laid back
	// out as strings sharing the dictionary's bytes. The manifest of a
	// first reader, which reads nothing, says which columns are strings.
	props := pqarrow.ArrowReadProperties{Parallel: true, BatchSize: 1 << 16}
	var (
		asString []int
		tbl      arrow.Table
	)
	fr, err := pqarrow.NewFileReader(pf, props, cfg.mem)
	if err == nil {
		for i, f := range fr.Manifest.Fields {
			switch f.Field.Type.ID() {
			case arrow.STRING, arrow.LARGE_STRING:
				if dictEncoded(pf.MetaData(), f.ColIndex) {
					props.SetReadDict(f.ColIndex, true)
					asString = append(asString, i)
				}
			}
		}
		if asString != nil {
			fr, _ = pqarrow.NewFileReader(pf, props, cfg.mem)
		}
		tbl, err = fr.ReadTable(context.Background())
	}
	if err != nil {
		return nil, err
	}
	defer tbl.Release()
	// The table's schema carries the metadata of the stored Arrow schema
	// only; pandas puts its own in the file's key-value metadata.
	kv := pf.MetaData().KeyValueMetadata()
	md := arrow.NewMetadata(kv.Keys(), kv.Values())
	schema := arrow.NewSchema(tbl.Schema().Fields(), &md)
	cols := make([]arrow.Column, tbl.NumCols())
	for i := range cols {
		cols[i] = *arrow.NewColumn(schema.Field(i), tbl.Column(i).Data())
	}
	withMeta := array.NewTable(schema, cols, tbl.NumRows())
	defer withMeta.Release()
	for i := range cols {
		cols[i].Release()
	}
	df, err := rawcol.FromOwnedTable(withMeta, cfg.mem, asString)
	if err != nil {
		return nil, err
	}
	return df.(*dataframe.DataFrame), nil
}

// dictEncoded reports whether every data page of column col, in every row
// group, is dictionary-encoded, as the page encoding statistics tell: then
// reading the column as a dictionary hashes nothing. (A plain page would be
// hashed into the dictionary value by value.)
func dictEncoded(md *metadata.FileMetaData, col int) bool {
	for g := range md.NumRowGroups() {
		cc, err := md.RowGroup(g).ColumnChunk(col)
		if err != nil || !cc.HasDictionaryPage() || len(cc.EncodingStats()) == 0 {
			return false
		}
		for _, st := range cc.EncodingStats() {
			const dataPage, dataPageV2 = 0, 3 // the Thrift PageType values
			data := int(st.PageType) == dataPage || int(st.PageType) == dataPageV2
			if data && st.Encoding != parquet.Encodings.RLEDict && st.Encoding != parquet.Encodings.PlainDict {
				return false
			}
		}
	}
	return true
}

// ReadParquetFile reads the Parquet file at path as ReadParquet does.
func ReadParquetFile(path string, opts ...Option) (*dataframe.DataFrame, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return ReadParquet(f, opts...)
}

// checkFooter checks what a file's footer declares before anything else
// is read: its column chunks and its stored Arrow schema.
func checkFooter(md *metadata.FileMetaData) error {
	if err := checkChunks(md); err != nil {
		return err
	}
	return checkArrowSchema(md)
}

// checkChunks checks what the footer declares of each column chunk before
// any is read: the reader allocates a chunk's declared size at once, so a
// chunk must lie within the file, its offsets and sizes not negative, and
// a row group with columns no negative number of rows.
func checkChunks(md *metadata.FileMetaData) error {
	size := md.GetSourceFileSize()
	for g := range md.NumRowGroups() {
		rg := md.RowGroup(g)
		for c := range rg.NumColumns() {
			cc, _ := rg.ColumnChunk(c) // c is in range: no error
			start, n := cc.DataPageOffset(), cc.TotalCompressedSize()
			if cc.HasDictionaryPage() && cc.DictionaryPageOffset() > 0 && cc.DictionaryPageOffset() < start {
				start = cc.DictionaryPageOffset()
			}
			if rg.NumRows() < 0 || start < 0 || n < 0 || start > size || n > size-start {
				return fmt.Errorf("%w: row group %d (%d rows) column %d: %d bytes at %d in a file of %d",
					arrowio.ErrCorrupt, g, rg.NumRows(), c, n, start, size)
			}
		}
	}
	return nil
}

// checkArrowSchema checks the Arrow schema a file stores in its
// ARROW:schema metadata (an IPC message, base64-encoded, as pqarrow
// decodes it) before pqarrow reads it: see internal/arrowfb.
func checkArrowSchema(md *metadata.FileMetaData) error {
	v := md.KeyValueMetadata().FindValue("ARROW:schema")
	if v == nil {
		return nil
	}
	var decoded []byte
	var err error
	if len(*v)%4 == 0 {
		decoded, err = base64.StdEncoding.DecodeString(*v)
	}
	if len(decoded) == 0 || err != nil {
		if decoded, err = base64.RawStdEncoding.DecodeString(*v); err != nil {
			return nil // pqarrow refuses it
		}
	}
	if err := arrowfb.CheckFramed(decoded); err != nil {
		return fmt.Errorf("%w: ARROW:schema: %v", arrowio.ErrCorrupt, err)
	}
	return nil
}
