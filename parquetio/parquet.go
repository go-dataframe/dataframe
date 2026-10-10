package parquetio

import (
	"context"
	"io"
	"os"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/parquet"
	"github.com/apache/arrow-go/v18/parquet/file"
	"github.com/apache/arrow-go/v18/parquet/pqarrow"
	"github.com/go-dataframe/dataframe"
	"github.com/go-dataframe/dataframe/arrowio"
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
	props := parquet.NewWriterProperties(
		parquet.WithCompression(codec),
		parquet.WithMaxRowGroupLength(cfg.rowGroup),
		parquet.WithAllocator(cfg.mem),
	)
	aprops := pqarrow.NewArrowWriterProperties(pqarrow.WithStoreSchema(), pqarrow.WithAllocator(cfg.mem))
	// The Parquet writer closes a sink that is an io.Closer; the caller's
	// writer stays the caller's to close.
	fw, err := pqarrow.NewFileWriter(rec.Schema(), struct{ io.Writer }{w}, props, aprops)
	if err != nil {
		return err
	}
	if err := fw.Write(rec); err != nil {
		fw.Close()
		return err
	}
	return fw.Close()
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
func ReadParquet(r arrowio.ReadAtSeeker, opts ...Option) (*dataframe.DataFrame, error) {
	cfg := newConfig(opts)
	pf, err := file.NewParquetReader(r, file.WithReadProps(parquet.NewReaderProperties(cfg.mem)))
	if err != nil {
		return nil, err
	}
	var tbl arrow.Table
	fr, err := pqarrow.NewFileReader(pf, pqarrow.ArrowReadProperties{Parallel: true, BatchSize: 1 << 16}, cfg.mem)
	if err == nil {
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
	return arrowio.FromTable(withMeta)
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
