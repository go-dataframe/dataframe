package parquetio

import (
	"fmt"

	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/apache/arrow-go/v18/parquet/compress"
	"github.com/go-dataframe/dataframe/arrowio"
)

// Compression selects the codec of a written Parquet file.
type Compression int

// The codecs.
const (
	// Snappy is the default, as pandas writes it.
	Snappy Compression = iota
	Uncompressed
	Gzip
	Brotli
	Zstd
	// LZ4 is Parquet's LZ4_RAW.
	LZ4
)

var compressionNames = [...]string{"snappy", "uncompressed", "gzip", "brotli", "zstd", "lz4"}

// String returns the codec's name.
func (c Compression) String() string {
	if c < 0 || int(c) >= len(compressionNames) {
		return fmt.Sprintf("Compression(%d)", int(c))
	}
	return compressionNames[c]
}

// codec maps the compression to the Parquet codec.
func (c Compression) codec() (compress.Compression, error) {
	switch c {
	case Snappy:
		return compress.Codecs.Snappy, nil
	case Uncompressed:
		return compress.Codecs.Uncompressed, nil
	case Gzip:
		return compress.Codecs.Gzip, nil
	case Brotli:
		return compress.Codecs.Brotli, nil
	case Zstd:
		return compress.Codecs.Zstd, nil
	case LZ4:
		return compress.Codecs.Lz4Raw, nil
	}
	return 0, fmt.Errorf("%w: compression %v", arrowio.ErrUnsupported, c)
}

// Option configures a write or a read.
type Option func(*config)

type config struct {
	compression Compression
	rowGroup    int64
	mem         memory.Allocator
	record      []arrowio.Option // how the frame becomes a record
}

func newConfig(opts []Option) *config {
	c := &config{compression: Snappy, rowGroup: 1 << 20, mem: memory.DefaultAllocator}
	for _, o := range opts {
		o(c)
	}
	return c
}

// WithCompression sets the codec of a written file (default Snappy).
func WithCompression(c Compression) Option { return func(o *config) { o.compression = c } }

// WithRowGroupLength sets the most rows a row group holds (default 1<<20,
// as pyarrow uses).
func WithRowGroupLength(n int64) Option { return func(o *config) { o.rowGroup = n } }

// WithTimeZone sets the time zone written with datetime columns, as
// arrowio.WithTimeZone does (default "UTC"; "" writes naive timestamps).
func WithTimeZone(tz string) Option {
	return func(o *config) { o.record = append(o.record, arrowio.WithTimeZone(tz)) }
}

// WithoutIndex leaves the row labels out of a written file, as
// arrowio.WithoutIndex does.
func WithoutIndex() Option {
	return func(o *config) { o.record = append(o.record, arrowio.WithoutIndex()) }
}

// WithAllocator sets the Arrow memory allocator used to read and write
// (default memory.DefaultAllocator).
func WithAllocator(mem memory.Allocator) Option { return func(o *config) { o.mem = mem } }
