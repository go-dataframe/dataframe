package arrowio

import (
	"errors"
	"fmt"

	"github.com/apache/arrow-go/v18/arrow/ipc"
	"github.com/apache/arrow-go/v18/arrow/memory"
)

// Errors returned by the package, wrapped with detail; test with errors.Is.
// Errors from the Arrow decoders themselves (a corrupt file, a
// failing writer) are returned as they come.
var (
	// ErrUnsupported reports a dtype or an Arrow type with no mapping, or an
	// option a format does not offer.
	ErrUnsupported = errors.New("arrowio: unsupported")
	// ErrOutOfRange reports a value that does not fit the dtype it maps to,
	// such as a timestamp in seconds beyond the nanosecond range.
	ErrOutOfRange = errors.New("arrowio: value out of range")
	// ErrMetadata reports "pandas" schema metadata that does not parse or
	// names columns the data lacks.
	ErrMetadata = errors.New("arrowio: bad pandas metadata")
	// ErrCorrupt reports Arrow data that is not what it declares: buffers
	// shorter than the values they hold, offsets out of order or out of
	// bounds, a dictionary index past its dictionary, columns of
	// different lengths, or a decoder that failed on it.
	ErrCorrupt = errors.New("arrowio: corrupt data")
)

// Compression selects the codec of an Arrow IPC file or stream. The Arrow
// IPC formats offer only LZ4 (the LZ4 frame format) and Zstd; Parquet has
// its own codecs, in package parquetio.
type Compression int

// The codecs.
const (
	// Uncompressed is the default.
	Uncompressed Compression = iota
	LZ4
	Zstd
)

var compressionNames = [...]string{"uncompressed", "lz4", "zstd"}

// String returns the codec's name.
func (c Compression) String() string {
	if c < 0 || int(c) >= len(compressionNames) {
		return fmt.Sprintf("Compression(%d)", int(c))
	}
	return compressionNames[c]
}

// Option configures a write or a read.
type Option func(*config)

type config struct {
	compression Compression
	timeZone    string
	index       bool
	mem         memory.Allocator
}

func newConfig(opts []Option) *config {
	c := &config{
		compression: Uncompressed,
		timeZone:    "UTC",
		index:       true,
		mem:         memory.DefaultAllocator,
	}
	for _, o := range opts {
		o(c)
	}
	return c
}

// WithCompression sets the codec of a written IPC file or stream.
func WithCompression(c Compression) Option { return func(o *config) { o.compression = c } }

// WithTimeZone sets the time zone written with datetime columns (default
// "UTC"). The values are UTC instants whatever the zone: it only tells a
// reader how to show them. The empty zone writes naive timestamps, which
// pandas reads as datetime64[ns] without a zone.
func WithTimeZone(tz string) Option { return func(o *config) { o.timeZone = tz } }

// WithoutIndex leaves the row labels out of a written file (pandas'
// index=False): no index columns and no index metadata, so a reader sees a
// default range index.
func WithoutIndex() Option { return func(o *config) { o.index = false } }

// WithAllocator sets the Arrow memory allocator used to read and write
// (default memory.DefaultAllocator).
func WithAllocator(mem memory.Allocator) Option { return func(o *config) { o.mem = mem } }

// ipcOptions maps the configuration to Arrow IPC writer options.
func (c *config) ipcOptions() ([]ipc.Option, error) {
	opts := []ipc.Option{ipc.WithAllocator(c.mem)}
	switch c.compression {
	case Uncompressed:
		return opts, nil
	case LZ4:
		return append(opts, ipc.WithLZ4()), nil
	case Zstd:
		return append(opts, ipc.WithZstd()), nil
	}
	return nil, fmt.Errorf("%w: compression %v in Arrow IPC (only lz4 and zstd)", ErrUnsupported, c.compression)
}
