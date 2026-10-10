package arrowio

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/bitutil"
	"github.com/go-dataframe/dataframe"
	"github.com/go-ndarray/ndarray"
)

// FromRecord converts an Arrow record batch to a frame: one column per
// field, the index restored from "pandas" schema metadata when there is
// some (a range index otherwise). The values are copied: the frame does not
// keep the record alive, and the caller still owns it.
func FromRecord(rec arrow.RecordBatch) (*dataframe.DataFrame, error) {
	return FromRecords(rec.Schema(), []arrow.RecordBatch{rec})
}

// FromRecords converts a sequence of record batches sharing one schema (an
// Arrow IPC stream, say) to one frame, as FromRecord does. Category columns
// whose dictionary changes from one batch to the next are unified.
func FromRecords(schema *arrow.Schema, recs []arrow.RecordBatch) (*dataframe.DataFrame, error) {
	chunks := make([][]arrow.Array, schema.NumFields())
	n := 0
	for _, r := range recs {
		for i := range chunks {
			chunks[i] = append(chunks[i], r.Column(i))
		}
		n += int(r.NumRows())
	}
	return fromChunks(schema, chunks, n)
}

// FromTable converts an Arrow table, whose columns may each be split in
// several chunks, to a frame, as FromRecord does.
func FromTable(tbl arrow.Table) (*dataframe.DataFrame, error) {
	return fromChunks(tbl.Schema(), tableChunks(tbl), int(tbl.NumRows()))
}

func tableChunks(tbl arrow.Table) [][]arrow.Array {
	chunks := make([][]arrow.Array, tbl.NumCols())
	for i := range chunks {
		chunks[i] = tbl.Column(i).Data().Chunks()
	}
	return chunks
}

func fromChunks(schema *arrow.Schema, chunks [][]arrow.Array, n int) (*dataframe.DataFrame, error) {
	spec, err := decodePandas(schema)
	if err != nil {
		return nil, err
	}
	cols := make([]dataframe.Column, len(chunks))
	for i, f := range schema.Fields() {
		s, err := toSeries(f, chunks[i], n)
		if err != nil {
			return nil, fmt.Errorf("column %q: %w", f.Name, err)
		}
		cols[i] = dataframe.Col(f.Name, s)
	}
	df, _ := dataframe.New(cols...)
	if spec == nil {
		return df, nil
	}
	return applyIndex(df, spec)
}

// toSeries converts the chunks of one column to a series of n values.
func toSeries(f arrow.Field, chunks []arrow.Array, n int) (*dataframe.Series, error) {
	name := f.Name
	valid := nullBits(chunks, n)
	switch t := f.Type.(type) {
	case *arrow.Float64Type:
		a, d := alloc(ndarray.Float64, n, ndarray.Data[float64])
		copyChunks(d, chunks, func(c arrow.Array) []float64 { return c.(*array.Float64).Float64Values() })
		nanFill(d, valid)
		return dataframe.NewSeries(name, a)
	case *arrow.Float32Type:
		a, d := alloc(ndarray.Float32, n, ndarray.Data[float32])
		copyChunks(d, chunks, func(c arrow.Array) []float32 { return c.(*array.Float32).Float32Values() })
		nanFill(d, valid)
		return dataframe.NewSeries(name, a)
	case *arrow.Int8Type:
		return fixedSeries(name, ndarray.Int8, n, valid, chunks, ndarray.Data[int8], func(c arrow.Array) []int8 { return c.(*array.Int8).Int8Values() })
	case *arrow.Int16Type:
		return fixedSeries(name, ndarray.Int16, n, valid, chunks, ndarray.Data[int16], func(c arrow.Array) []int16 { return c.(*array.Int16).Int16Values() })
	case *arrow.Int32Type:
		return fixedSeries(name, ndarray.Int32, n, valid, chunks, ndarray.Data[int32], func(c arrow.Array) []int32 { return c.(*array.Int32).Int32Values() })
	case *arrow.Int64Type:
		return fixedSeries(name, ndarray.Int64, n, valid, chunks, ndarray.Data[int64], func(c arrow.Array) []int64 { return c.(*array.Int64).Int64Values() })
	case *arrow.Uint8Type:
		return fixedSeries(name, ndarray.Uint8, n, valid, chunks, ndarray.Data[uint8], func(c arrow.Array) []uint8 { return c.(*array.Uint8).Uint8Values() })
	case *arrow.Uint16Type:
		return fixedSeries(name, ndarray.Uint16, n, valid, chunks, ndarray.Data[uint16], func(c arrow.Array) []uint16 { return c.(*array.Uint16).Uint16Values() })
	case *arrow.Uint32Type:
		return fixedSeries(name, ndarray.Uint32, n, valid, chunks, ndarray.Data[uint32], func(c arrow.Array) []uint32 { return c.(*array.Uint32).Uint32Values() })
	case *arrow.Uint64Type:
		return fixedSeries(name, ndarray.Uint64, n, valid, chunks, ndarray.Data[uint64], func(c arrow.Array) []uint64 { return c.(*array.Uint64).Uint64Values() })
	case *arrow.BooleanType:
		a, d := alloc(ndarray.Bool, n, ndarray.Data[bool])
		off := 0
		for _, c := range chunks {
			b := c.(*array.Boolean)
			for i := range b.Len() {
				d[off+i] = b.Value(i)
			}
			off += b.Len()
		}
		return dataframe.NewSeriesBitmap(name, a, valid)
	case *arrow.TimestampType:
		return timestamps(name, t.Unit, n, valid, chunks)
	case *arrow.StringType, *arrow.LargeStringType:
		return dataframe.NewSeriesBitmap(name, readStrings(chunks, n), valid)
	case *arrow.DictionaryType:
		return categories(name, t, n, chunks)
	case *arrow.NullType:
		// pyarrow writes an object column of nothing but None as nulls;
		// pandas reads it back as an object column, here a string one.
		return dataframe.NewSeriesBitmap(name, make([]string, n), make([]uint64, (n+63)>>6))
	}
	return nil, fmt.Errorf("%w: Arrow type %v", ErrUnsupported, f.Type)
}

// alloc returns an array of n zeros of an ndarray dtype and its storage,
// which a series built from the array shares.
func alloc[T any](dt ndarray.DType, n int, data func(*ndarray.Array) ([]T, bool)) (*ndarray.Array, []T) {
	a, _ := ndarray.ZerosOf(dt, n)
	d, _ := data(a)
	return a, d
}

// copyChunks copies each chunk's values after the previous ones.
func copyChunks[T any](d []T, chunks []arrow.Array, values func(arrow.Array) []T) {
	off := 0
	for _, c := range chunks {
		off += copy(d[off:], values(c))
	}
}

// fixedSeries reads an integer column into a series sharing one fresh array.
func fixedSeries[T any](name string, dt ndarray.DType, n int, valid []uint64, chunks []arrow.Array,
	data func(*ndarray.Array) ([]T, bool), values func(arrow.Array) []T) (*dataframe.Series, error) {
	a, d := alloc(dt, n, data)
	copyChunks(d, chunks, values)
	return dataframe.NewSeriesBitmap(name, a, valid)
}

// nullBits returns the validity bitmap of the chunks laid end to end, or
// nil when none has a null. The chunks' Arrow bitmaps are copied whole into
// one byte bitmap, read as little-endian words.
func nullBits(chunks []arrow.Array, n int) []uint64 {
	var b []byte
	off := 0
	for _, c := range chunks {
		if c.NullN() > 0 {
			if b == nil {
				b = bytes.Repeat([]byte{0xff}, 8*((n+63)>>6))
			}
			if bm := c.NullBitmapBytes(); bm != nil {
				bitutil.CopyBitmap(bm, c.Data().Offset(), c.Len(), b, off)
			} else {
				// A null-type array has no bitmap: every slot is null.
				bitutil.SetBitsTo(b, int64(off), int64(c.Len()), false)
			}
		}
		off += c.Len()
	}
	if b == nil {
		return nil
	}
	v := make([]uint64, len(b)/8)
	for i := range v {
		v[i] = binary.LittleEndian.Uint64(b[8*i:])
	}
	return v
}

func isValid(v []uint64, i int) bool { return v == nil || v[i>>6]>>(uint(i)&63)&1 == 1 }

// nanFill writes NaN in the slots of null floats: a float column marks its
// missing values in band, as pandas does.
func nanFill[T float32 | float64](d []T, valid []uint64) {
	for i := range d {
		if !isValid(valid, i) {
			d[i] = T(math.NaN())
		}
	}
}

// timestamps reads a timestamp column of any unit, converted to
// nanoseconds. Arrow timestamps are UTC instants whatever their zone (a
// naive one is wall time, read as UTC), so the zone itself is dropped.
func timestamps(name string, unit arrow.TimeUnit, n int, valid []uint64, chunks []arrow.Array) (*dataframe.Series, error) {
	a, d := alloc(ndarray.Int64, n, ndarray.Data[int64])
	off := 0
	for _, c := range chunks {
		for _, v := range c.(*array.Timestamp).TimestampValues() {
			d[off] = int64(v)
			off++
		}
	}
	if m := int64(unit.Multiplier()); m != 1 {
		for i, v := range d {
			switch {
			case !isValid(valid, i):
				d[i] = 0
			case v > math.MaxInt64/m || v < math.MinInt64/m:
				return nil, fmt.Errorf("%w: timestamp %d%s is outside the nanosecond range", ErrOutOfRange, v, unit)
			default:
				d[i] = v * m
			}
		}
	}
	s, _ := dataframe.NewSeriesBitmap(name, a, valid)
	return dataframe.ToDatetime(s)
}

// readStrings reads utf8 or large_utf8 chunks. Each chunk's bytes are copied
// once into one Go string that the values slice, so the result neither
// allocates per value nor points into Arrow buffers.
func readStrings(chunks []arrow.Array, n int) []string {
	out := make([]string, n)
	off := 0
	for _, c := range chunks {
		if c.Len() == 0 {
			continue
		}
		switch c := c.(type) {
		case *array.String:
			sliceStrings(out[off:], c.ValueOffsets(), c.ValueBytes())
		case *array.LargeString:
			sliceStrings(out[off:], c.ValueOffsets(), c.ValueBytes())
		}
		off += c.Len()
	}
	return out
}

func sliceStrings[O int32 | int64](out []string, offs []O, b []byte) {
	buf := string(b)
	base := offs[0]
	for i := range len(offs) - 1 {
		out[i] = buf[offs[i]-base : offs[i+1]-base]
	}
}

// categories reads a dictionary column with string values as a category
// column. Chunks with different dictionaries are unified: the categories
// are the first chunk's, then each new value in order of appearance.
func categories(name string, t *arrow.DictionaryType, n int, chunks []arrow.Array) (*dataframe.Series, error) {
	switch t.ValueType.(type) {
	case *arrow.StringType, *arrow.LargeStringType:
	default:
		return nil, fmt.Errorf("%w: dictionary of %v (only string categories)", ErrUnsupported, t.ValueType)
	}
	var levels []string
	pos := map[string]int32{}
	codes := make([]int32, n)
	off := 0
	for _, c := range chunks {
		d := c.(*array.Dictionary)
		dict := d.Dictionary()
		values := readStrings([]arrow.Array{dict}, dict.Len())
		remap := make([]int32, len(values))
		for k, v := range values {
			if dict.IsNull(k) {
				remap[k] = -1
				continue
			}
			p, ok := pos[v]
			if !ok {
				p = int32(len(levels))
				pos[v] = p
				levels = append(levels, v)
			}
			remap[k] = p
		}
		out := codes[off : off+d.Len()]
		switch idx := d.Indices().(type) {
		case *array.Int8:
			remapCodes(out, idx.Int8Values(), remap)
		case *array.Int16:
			remapCodes(out, idx.Int16Values(), remap)
		case *array.Int32:
			remapCodes(out, idx.Int32Values(), remap)
		case *array.Int64:
			remapCodes(out, idx.Int64Values(), remap)
		case *array.Uint8:
			remapCodes(out, idx.Uint8Values(), remap)
		case *array.Uint16:
			remapCodes(out, idx.Uint16Values(), remap)
		case *array.Uint32:
			remapCodes(out, idx.Uint32Values(), remap)
		case *array.Uint64:
			remapCodes(out, idx.Uint64Values(), remap)
		}
		if d.NullN() > 0 {
			for i := range out {
				if d.IsNull(i) {
					out[i] = -1
				}
			}
		}
		off += d.Len()
	}
	return dataframe.NewCategorical(name, codes, levels)
}

// remapCodes maps dictionary indices to category codes. A null slot may
// hold any index, so an index outside the dictionary is left to the caller,
// which marks null slots missing.
func remapCodes[T int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64](out []int32, idx []T, remap []int32) {
	for i, k := range idx {
		if uint64(k) < uint64(len(remap)) {
			out[i] = remap[k]
		}
	}
}
