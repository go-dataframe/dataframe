// Package rawcol lets the I/O subpackages (arrowio, parquetio) build series
// over storage they own, and read a series' storage, without the copies the
// public constructors and accessors make. Package dataframe sets the
// functions when it is initialised; a caller imports dataframe as well, so
// they are always set. Series travel as any to avoid an import cycle: each
// one is a *dataframe.Series.
package rawcol

import "github.com/apache/arrow-go/v18/arrow/memory"

var (
	// Series returns a series named name over data, a []T of one of the
	// numeric column element types ([]float64 ... []uint64) or []string, with
	// the validity bitmap valid (nil when nothing is missing), both shared
	// rather than copied. With datetime, data is an []int64 of nanoseconds
	// and the series is a datetime one. The caller must not touch data or
	// valid afterwards.
	Series func(name string, data any, valid []uint64, datetime bool) any

	// Categorical returns a category series named name over codes into
	// levels, sharing codes, each in range and 0 where missing, with the
	// validity bitmap valid (nil when nothing is missing), shared too, with
	// its bits past the last value set.
	Categorical func(name string, codes []int32, levels []string, valid []uint64) any

	// Strings returns a string series' storage and validity bitmap, shared
	// (read only), and false for any other dtype. A missing slot may hold
	// any string.
	Strings func(s any) (data []string, valid []uint64, ok bool)

	// Codes returns a category series' codes, validity bitmap and levels,
	// shared (read only). A missing slot may hold any code.
	Codes func(s any) (codes []int32, valid []uint64, levels []string)
)

// FromOwnedTable converts an arrow.Table read with the allocator mem, which
// the caller owns and gives up, to a *dataframe.DataFrame, as
// arrowio.FromTable does, letting the frame keep the table's buffers where
// that is safe. The columns at the positions asString are dictionary
// arrays of strings read for string columns: they become string columns,
// not category ones. Package arrowio sets it.
var FromOwnedTable func(tbl any, mem memory.Allocator, asString []int) (any, error)
