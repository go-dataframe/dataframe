// Package arrowio reads and writes frames as Apache Arrow data: the Arrow
// IPC file format (Feather v2) and the Arrow IPC streaming format, and
// converts frames to and from Arrow record batches and tables; package
// parquetio builds Parquet on these conversions. It is built on
// github.com/apache/arrow-go, the Apache Arrow project's own pure-Go
// implementation, and needs no cgo.
//
// Files written here read back in pandas (through pyarrow) as the same
// frame, index included, and files pandas writes read here with their
// index: both sides use the "pandas" schema metadata that records which
// columns hold the index levels, or the range a default index spans.
//
// # Type mapping
//
//	dtype       written as                    read from
//	float64     double                        double
//	float32     float                         float
//	int8..64    int8..int64                   int8..int64
//	uint8..64   uint8..uint64                 uint8..uint64
//	bool        bool                          bool
//	string      utf8 (large_utf8 past 2 GiB)  utf8, large_utf8, null
//	datetime    timestamp[ns, tz=UTC]         timestamp[s|ms|us|ns], any zone
//	category    dictionary<int32, utf8>       dictionary<any int, utf8|large_utf8>
//	period      utf8, its text form           -
//	complex     ErrUnsupported                -
//
// Missing values travel in the Arrow validity bitmaps both ways. A float
// column has NaN as its missing value, as in pandas: NaN is written as null
// (pandas writes it so too), and a null float reads as NaN. Arrow keeps a
// non-null NaN apart from null; here both read as a missing value. A null in
// any other column becomes a masked value. In the "pandas" metadata, an
// integer or bool column with missing values is described as pandas'
// nullable Int64/boolean dtype, so pandas reads it back without turning it
// into floats.
//
// A datetime column holds UTC instants. It is written with the zone "UTC"
// (WithTimeZone changes the zone, or drops it to write naive timestamps);
// on read the zone is ignored, since Arrow timestamps are UTC instants
// whatever their zone, and a naive timestamp is read as if it were UTC.
// Units coarser than nanoseconds are scaled, and a value outside the
// nanosecond range (years 1677-2262) is an ErrOutOfRange.
//
// A period column is written as its text, what Series.Strings gives: Arrow has no
// calendar-period type, and pandas itself stores periods through a pyarrow
// extension type that other Arrow readers do not know. It reads back as a
// string column.
//
// # The index
//
// A default range index is written as metadata only (a range descriptor),
// as pandas does. Any other index is written as one column per level, after
// the data columns, named after the level, or __index_level_k__ when the
// level is unnamed. On read the metadata is honoured: a range descriptor
// starting at 0 with step 1 gives a range index; any other range gives an
// int64 index of the same labels; index columns become the levels of a
// (possibly multi-level) index. Without metadata the frame gets a range
// index and every column is a data column.
package arrowio
