package arrowio

import (
	"io"
	"os"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/ipc"
	"github.com/go-dataframe/dataframe"
)

// ReadAtSeeker is what the random-access readers (the Arrow IPC file
// format, Parquet) read from; an *os.File and a *bytes.Reader are ones.
type ReadAtSeeker interface {
	io.Reader
	io.ReaderAt
	io.Seeker
}

// WriteIPCFile writes a frame in the Arrow IPC file format (Feather v2), as
// ToRecord lays it out. The options are WithCompression (LZ4 or Zstd; none
// by default), WithTimeZone, WithoutIndex and WithAllocator.
func WriteIPCFile(w io.Writer, df *dataframe.DataFrame, opts ...Option) error {
	cfg := newConfig(opts)
	iopts, err := cfg.ipcOptions()
	if err != nil {
		return err
	}
	rec, err := toRecord(df, cfg)
	if err != nil {
		return err
	}
	defer rec.Release()
	fw, _ := ipc.NewFileWriter(w, append(iopts, ipc.WithSchema(rec.Schema()))...)
	if err := fw.Write(rec); err != nil {
		fw.Close()
		return err
	}
	return fw.Close()
}

// WriteIPCStream writes a frame in the Arrow IPC streaming format, as
// WriteIPCFile does.
func WriteIPCStream(w io.Writer, df *dataframe.DataFrame, opts ...Option) error {
	cfg := newConfig(opts)
	iopts, err := cfg.ipcOptions()
	if err != nil {
		return err
	}
	rec, err := toRecord(df, cfg)
	if err != nil {
		return err
	}
	defer rec.Release()
	sw := ipc.NewWriter(w, append(iopts, ipc.WithSchema(rec.Schema()))...)
	if err := sw.Write(rec); err != nil {
		sw.Close()
		return err
	}
	return sw.Close()
}

// ReadIPCFile reads a frame from the Arrow IPC file format (Feather v2, as
// pandas' to_feather writes it), every record batch in order. The only
// option it takes is WithAllocator.
func ReadIPCFile(r ReadAtSeeker, opts ...Option) (*dataframe.DataFrame, error) {
	cfg := newConfig(opts)
	fr, err := ipc.NewFileReader(r, ipc.WithAllocator(cfg.mem))
	if err != nil {
		return nil, err
	}
	defer fr.Close()
	recs := make([]arrow.RecordBatch, 0, fr.NumRecords())
	defer releaseAll(&recs)
	for i := range fr.NumRecords() {
		rec, err := fr.RecordBatch(i)
		if err != nil {
			return nil, err
		}
		rec.Retain()
		recs = append(recs, rec)
	}
	return fromRecords(fr.Schema(), recs, owns(cfg.mem))
}

// ReadIPCStream reads a frame from the Arrow IPC streaming format, every
// record batch in order. The only option it takes is WithAllocator.
func ReadIPCStream(r io.Reader, opts ...Option) (*dataframe.DataFrame, error) {
	cfg := newConfig(opts)
	sr, err := ipc.NewReader(r, ipc.WithAllocator(cfg.mem))
	if err != nil {
		return nil, err
	}
	defer sr.Release()
	var recs []arrow.RecordBatch
	defer releaseAll(&recs)
	for sr.Next() {
		rec := sr.RecordBatch()
		rec.Retain()
		recs = append(recs, rec)
	}
	if err := sr.Err(); err != nil {
		return nil, err
	}
	return fromRecords(sr.Schema(), recs, owns(cfg.mem))
}

func releaseAll(recs *[]arrow.RecordBatch) {
	for _, r := range *recs {
		r.Release()
	}
}

// ReadFeather reads the Arrow IPC file (Feather v2) at path.
func ReadFeather(path string, opts ...Option) (*dataframe.DataFrame, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return ReadIPCFile(f, opts...)
}

// WriteFeather writes a frame to path in the Arrow IPC file format (Feather
// v2), as WriteIPCFile does.
func WriteFeather(path string, df *dataframe.DataFrame, opts ...Option) error {
	return writePath(path, func(w io.Writer) error { return WriteIPCFile(w, df, opts...) })
}

// writePath creates path and writes it with write, removing nothing on
// failure but reporting the first error, including the one from Close.
func writePath(path string, write func(io.Writer) error) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := write(f); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}
