package arrowio

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	mbits "math/bits"
	"runtime"
	"slices"
	"strings"
	"sync"
	"unsafe"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/bitutil"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/go-dataframe/dataframe"
	"github.com/go-dataframe/dataframe/internal/rawcol"
	"github.com/go-ndarray/ndarray"
)

func init() {
	rawcol.FromOwnedTable = func(tbl any, mem memory.Allocator, asString []int) (any, error) {
		t := tbl.(arrow.Table)
		return fromChunks(t.Schema(), tableChunks(t), int(t.NumRows()), owns(mem), asString...)
	}
}

// owns reports whether records read with mem may hand their buffers over to
// the frame rather than have them copied. The Go allocator's buffers are
// ordinary Go memory, which the garbage collector keeps alive while the
// frame points into it, and which Release never frees or reuses (its Free
// does nothing). Any other allocator may free or reuse its memory.
func owns(mem memory.Allocator) bool {
	_, ok := mem.(*memory.GoAllocator)
	return ok
}

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
	return fromRecords(schema, recs, false)
}

// fromRecords is FromRecords; with own, the records' buffers may be handed
// over to the frame (see fromChunks).
func fromRecords(schema *arrow.Schema, recs []arrow.RecordBatch, own bool) (*dataframe.DataFrame, error) {
	chunks := make([][]arrow.Array, schema.NumFields())
	n := 0
	for _, r := range recs {
		for i := range chunks {
			chunks[i] = append(chunks[i], r.Column(i))
		}
		n += int(r.NumRows())
	}
	return fromChunks(schema, chunks, n, own)
}

// FromTable converts an Arrow table, whose columns may each be split in
// several chunks, to a frame, as FromRecord does.
func FromTable(tbl arrow.Table) (*dataframe.DataFrame, error) {
	return fromChunks(tbl.Schema(), tableChunks(tbl), int(tbl.NumRows()), false)
}

func tableChunks(tbl arrow.Table) [][]arrow.Array {
	chunks := make([][]arrow.Array, tbl.NumCols())
	for i := range chunks {
		chunks[i] = tbl.Column(i).Data().Chunks()
	}
	return chunks
}

// fromChunks converts the columns' chunks to a frame. With own, a column
// held in one chunk keeps that chunk's buffers rather than copying them:
// the caller then gives the buffers up, which only a reader that made them
// with the Go allocator can do (see owns). The columns at the positions
// asString are dictionaries of strings to read as string columns.
func fromChunks(schema *arrow.Schema, chunks [][]arrow.Array, n int, own bool, asString ...int) (*dataframe.DataFrame, error) {
	spec, err := decodePandas(schema)
	if err != nil {
		return nil, err
	}
	fields := schema.Fields()
	series := make([]*dataframe.Series, len(fields))
	errs := make([]error, len(fields))
	convert := func(i int) {
		f := fields[i]
		if f.Type.ID() == arrow.DICTIONARY && slices.Contains(asString, i) {
			series[i] = dictStrings(f.Name, chunks[i], n)
		} else {
			series[i], errs[i] = toSeries(f, chunks[i], n, own)
		}
	}
	// Columns convert independently: a large frame converts them at once.
	if n >= 1<<15 && runtime.GOMAXPROCS(0) > 1 {
		var wg sync.WaitGroup
		for i := range fields {
			wg.Go(func() { convert(i) })
		}
		wg.Wait()
	} else {
		for i := range fields {
			convert(i)
		}
	}
	cols := make([]dataframe.Column, len(fields))
	for i, f := range fields {
		if errs[i] != nil {
			return nil, fmt.Errorf("column %q: %w", f.Name, errs[i])
		}
		cols[i] = dataframe.Col(f.Name, series[i])
	}
	df, _ := dataframe.New(cols...)
	if spec == nil {
		return df, nil
	}
	return applyIndex(df, spec)
}

// toSeries converts the chunks of one column to a series of n values; with
// own, see fromChunks.
func toSeries(f arrow.Field, chunks []arrow.Array, n int, own bool) (*dataframe.Series, error) {
	name := f.Name
	valid := nullBits(chunks, n)
	switch t := f.Type.(type) {
	case *arrow.Float64Type:
		return floatSeries(name, ndarray.Float64, n, valid, chunks, own, ndarray.Data[float64], func(c arrow.Array) []float64 { return c.(*array.Float64).Float64Values() })
	case *arrow.Float32Type:
		return floatSeries(name, ndarray.Float32, n, valid, chunks, own, ndarray.Data[float32], func(c arrow.Array) []float32 { return c.(*array.Float32).Float32Values() })
	case *arrow.Int8Type:
		return fixedSeries(name, ndarray.Int8, n, valid, chunks, own, ndarray.Data[int8], func(c arrow.Array) []int8 { return c.(*array.Int8).Int8Values() })
	case *arrow.Int16Type:
		return fixedSeries(name, ndarray.Int16, n, valid, chunks, own, ndarray.Data[int16], func(c arrow.Array) []int16 { return c.(*array.Int16).Int16Values() })
	case *arrow.Int32Type:
		return fixedSeries(name, ndarray.Int32, n, valid, chunks, own, ndarray.Data[int32], func(c arrow.Array) []int32 { return c.(*array.Int32).Int32Values() })
	case *arrow.Int64Type:
		return fixedSeries(name, ndarray.Int64, n, valid, chunks, own, ndarray.Data[int64], func(c arrow.Array) []int64 { return c.(*array.Int64).Int64Values() })
	case *arrow.Uint8Type:
		return fixedSeries(name, ndarray.Uint8, n, valid, chunks, own, ndarray.Data[uint8], func(c arrow.Array) []uint8 { return c.(*array.Uint8).Uint8Values() })
	case *arrow.Uint16Type:
		return fixedSeries(name, ndarray.Uint16, n, valid, chunks, own, ndarray.Data[uint16], func(c arrow.Array) []uint16 { return c.(*array.Uint16).Uint16Values() })
	case *arrow.Uint32Type:
		return fixedSeries(name, ndarray.Uint32, n, valid, chunks, own, ndarray.Data[uint32], func(c arrow.Array) []uint32 { return c.(*array.Uint32).Uint32Values() })
	case *arrow.Uint64Type:
		return fixedSeries(name, ndarray.Uint64, n, valid, chunks, own, ndarray.Data[uint64], func(c arrow.Array) []uint64 { return c.(*array.Uint64).Uint64Values() })
	case *arrow.BooleanType:
		a, d := alloc(ndarray.Bool, n, ndarray.Data[bool])
		off := 0
		for _, c := range chunks {
			if c.Len() == 0 {
				continue
			}
			unpackBits(d[off:off+c.Len()], c.Data().Buffers()[1].Bytes(), c.Data().Offset())
			off += c.Len()
		}
		return dataframe.NewSeriesBitmap(name, a, valid)
	case *arrow.TimestampType:
		return timestamps(name, t.Unit, n, valid, chunks, own)
	case *arrow.StringType, *arrow.LargeStringType:
		return stringSeries(name, chunks, n, valid, own), nil
	case *arrow.DictionaryType:
		return categories(name, t, n, chunks, own)
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

// adopt returns the values of a column held in one chunk, sharing the
// chunk's buffer, when the caller owns it (own) and the values are aligned
// for T; ok is false otherwise, and the caller copies.
func adopt[T any](own bool, chunks []arrow.Array, values func(arrow.Array) []T) (d []T, ok bool) {
	if !own || len(chunks) != 1 {
		return nil, false
	}
	d = values(chunks[0])
	return d, aligned(d)
}

// aligned reports whether d's first element sits at an address aligned for
// T, as the elements of a Go slice of T must.
func aligned[T any](d []T) bool {
	var zero T
	return uintptr(unsafe.Pointer(unsafe.SliceData(d)))%unsafe.Alignof(zero) == 0
}

// raw builds a series over storage the caller hands over (see rawcol).
func raw(name string, data any, valid []uint64, datetime bool) *dataframe.Series {
	return rawcol.Series(name, data, valid, datetime).(*dataframe.Series)
}

// copyChunks copies each chunk's values after the previous ones.
func copyChunks[T any](d []T, chunks []arrow.Array, values func(arrow.Array) []T) {
	off := 0
	for _, c := range chunks {
		off += copy(d[off:], values(c))
	}
}

// fixedSeries reads an integer column into a series sharing one fresh array,
// or the chunk's own buffer (see adopt).
func fixedSeries[T any](name string, dt ndarray.DType, n int, valid []uint64, chunks []arrow.Array, own bool,
	data func(*ndarray.Array) ([]T, bool), values func(arrow.Array) []T) (*dataframe.Series, error) {
	if d, ok := adopt(own, chunks, values); ok {
		return raw(name, d, valid, false), nil
	}
	a, d := alloc(dt, n, data)
	copyChunks(d, chunks, values)
	return dataframe.NewSeriesBitmap(name, a, valid)
}

// floatSeries reads a float column as fixedSeries does. A float column marks
// its missing values in band, as pandas does: NaN goes in the null slots,
// and the series has no validity bitmap.
func floatSeries[T float32 | float64](name string, dt ndarray.DType, n int, valid []uint64, chunks []arrow.Array, own bool,
	data func(*ndarray.Array) ([]T, bool), values func(arrow.Array) []T) (*dataframe.Series, error) {
	if d, ok := adopt(own, chunks, values); ok {
		nanFill(d, valid)
		return raw(name, d, nil, false), nil
	}
	a, d := alloc(dt, n, data)
	copyChunks(d, chunks, values)
	nanFill(d, valid)
	return dataframe.NewSeries(name, a)
}

// unpackBits expands the Arrow bitmap bm, from bit off on, into d: eight
// bools at a time from a byte-aligned start, through spread.
func unpackBits(d []bool, bm []byte, off int) {
	i := 0
	if off&7 == 0 {
		bm = bm[off>>3:]
		for ; i+8 <= len(d); i += 8 {
			*(*[8]bool)(d[i:]) = spread[bm[i>>3]]
		}
	}
	for ; i < len(d); i++ {
		j := off + i
		d[i] = bm[j>>3]>>(uint(j)&7)&1 == 1
	}
}

// spread maps a byte to the eight bools of its bits, least significant
// first.
var spread = func() (t [256][8]bool) {
	for b := range t {
		for k := range t[b] {
			t[b][k] = b>>k&1 == 1
		}
	}
	return t
}()

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

// nanFill writes NaN in the slots of null floats.
func nanFill[T float32 | float64](d []T, valid []uint64) {
	for w, bits := range valid {
		for bits != math.MaxUint64 {
			b := mbits.TrailingZeros64(^bits)
			bits |= 1 << uint(b)
			if i := w<<6 + b; i < len(d) {
				d[i] = T(math.NaN())
			}
		}
	}
}

// timestamps reads a timestamp column of any unit, converted to
// nanoseconds. Arrow timestamps are UTC instants whatever their zone (a
// naive one is wall time, read as UTC), so the zone itself is dropped.
// With own, the values are converted in the chunk's own buffer (see adopt).
func timestamps(name string, unit arrow.TimeUnit, n int, valid []uint64, chunks []arrow.Array, own bool) (*dataframe.Series, error) {
	values := func(c arrow.Array) []int64 {
		v := c.(*array.Timestamp).TimestampValues()
		return unsafe.Slice((*int64)(unsafe.SliceData(v)), len(v))
	}
	d, adopted := adopt(own, chunks, values)
	var a *ndarray.Array
	if !adopted {
		a, d = alloc(ndarray.Int64, n, ndarray.Data[int64])
		copyChunks(d, chunks, values)
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
	if adopted {
		return raw(name, d, valid, true), nil
	}
	s, _ := dataframe.NewSeriesBitmap(name, a, valid)
	return dataframe.ToDatetime(s)
}

// stringSeries reads utf8 or large_utf8 chunks as a string series. With
// own, a column in one chunk keeps the chunk's bytes and offsets as they
// are (see fromChunks); otherwise the bytes are copied once, end to end,
// into a buffer of the series' own.
func stringSeries(name string, chunks []arrow.Array, n int, valid []uint64, own bool) *dataframe.Series {
	if own && len(chunks) == 1 && chunks[0].Len() > 0 {
		switch c := chunks[0].(type) {
		case *array.String:
			if offs := c.ValueOffsets(); aligned(offs) {
				return rawStrings(name, valueBuffer(c), offs, valid)
			}
		case *array.LargeString:
			if offs := c.ValueOffsets(); aligned(offs) {
				return rawStrings(name, valueBuffer(c), offs, valid)
			}
		}
	}
	total := 0
	for _, c := range chunks {
		if c.Len() > 0 {
			total += len(c.(interface{ ValueBytes() []byte }).ValueBytes())
		}
	}
	buf := make([]byte, 0, total)
	if total <= maxStr32 {
		offs := make([]int32, 1, n+1)
		buf, offs = appendChunks(buf, offs, chunks)
		return rawStrings(name, unsafe.String(unsafe.SliceData(buf), len(buf)), offs, valid)
	}
	offs := make([]int64, 1, n+1)
	buf, offs = appendChunks(buf, offs, chunks)
	return rawStrings(name, unsafe.String(unsafe.SliceData(buf), len(buf)), offs, valid)
}

// maxStr32 is the most bytes a string series is read into with int32
// offsets; a variable so that tests can take the int64 path.
var maxStr32 = math.MaxInt32

// valueBuffer returns the whole values buffer of a string chunk, which its
// offsets index, as a string over the same bytes.
func valueBuffer(c arrow.Array) string {
	b := c.Data().Buffers()[2].Bytes()
	return unsafe.String(unsafe.SliceData(b), len(b))
}

// appendChunks appends the chunks' values to buf and their end offsets to
// offs.
func appendChunks[P int32 | int64](buf []byte, offs []P, chunks []arrow.Array) ([]byte, []P) {
	for _, c := range chunks {
		if c.Len() == 0 {
			continue
		}
		switch c := c.(type) {
		case *array.String:
			buf, offs = appendOffsets(buf, offs, c.ValueOffsets(), c.ValueBytes())
		case *array.LargeString:
			buf, offs = appendOffsets(buf, offs, c.ValueOffsets(), c.ValueBytes())
		}
	}
	return buf, offs
}

func appendOffsets[O, P int32 | int64](buf []byte, offs []P, src []O, b []byte) ([]byte, []P) {
	base := P(len(buf)) - P(src[0])
	for _, o := range src[1:] {
		offs = append(offs, base+P(o))
	}
	return append(buf, b...), offs
}

// rawStrings builds a series over storage the caller hands over.
func rawStrings(name, data string, offs any, valid []uint64) *dataframe.Series {
	return rawcol.StringSeries(name, data, offs, valid).(*dataframe.Series)
}

// stringValues returns the values of a utf8 or large_utf8 array, copied
// so that they do not point into Arrow buffers.
func stringValues(a arrow.Array) []string {
	v := a.(interface{ Value(int) string })
	out := make([]string, a.Len())
	for i := range out {
		out[i] = strings.Clone(v.Value(i))
	}
	return out
}

// dictStrings reads chunks of dictionary<int32, utf8 or large_utf8> as a
// string column: the dictionaries laid end to end, then each value
// gathered from its chunk's dictionary.
func dictStrings(name string, chunks []arrow.Array, n int) *dataframe.Series {
	valid := nullBits(chunks, n)
	idx := make([]int, n)
	var dicts []arrow.Array
	off, base := 0, 0
	for _, c := range chunks {
		d := c.(*array.Dictionary)
		for i, k := range d.Indices().(*array.Int32).Int32Values() {
			idx[off+i] = -1
			if isValid(valid, off+i) {
				idx[off+i] = base + int(k)
			}
		}
		dicts = append(dicts, d.Dictionary())
		base += d.Dictionary().Len()
		off += d.Len()
	}
	all := stringSeries(name, dicts, base, nil, false)
	return rawcol.Take(all, idx).(*dataframe.Series)
}

// categories reads a dictionary column with string values as a category
// column. Chunks with different dictionaries are unified: the categories
// are the first chunk's, then each new value in order of appearance. With
// own, a column in one chunk of int32 indices whose dictionary is already
// the categories in order keeps the indices buffer as its codes.
func categories(name string, t *arrow.DictionaryType, n int, chunks []arrow.Array, own bool) (*dataframe.Series, error) {
	switch t.ValueType.(type) {
	case *arrow.StringType, *arrow.LargeStringType:
	default:
		return nil, fmt.Errorf("%w: dictionary of %v (only string categories)", ErrUnsupported, t.ValueType)
	}
	var levels []string
	pos := map[string]int32{}
	remaps := make([][]int32, len(chunks))
	identity, dictNulls := true, false
	for c, ch := range chunks {
		dict := ch.(*array.Dictionary).Dictionary()
		values := stringValues(dict)
		remap := make([]int32, len(values))
		for k, v := range values {
			if dict.IsNull(k) {
				remap[k] = -1
				dictNulls = true
				continue
			}
			p, ok := pos[v]
			if !ok {
				p = int32(len(levels))
				pos[v] = p
				levels = append(levels, v)
			}
			remap[k] = p
			identity = identity && p == int32(k)
		}
		remaps[c] = remap
	}
	valid := nullBits(chunks, n)
	var codes []int32
	if idx, ok := adopt(own && identity && !dictNulls && t.IndexType.ID() == arrow.INT32, chunks, int32Indices); ok {
		// The indices are the codes already; only their null slots, which
		// may hold anything, need a 0.
		codes = idx
	} else {
		codes = make([]int32, n)
		off := 0
		for c, ch := range chunks {
			d := ch.(*array.Dictionary)
			out := codes[off : off+d.Len()]
			switch idx := d.Indices().(type) {
			case *array.Int8:
				remapCodes(out, idx.Int8Values(), remaps[c])
			case *array.Int16:
				remapCodes(out, idx.Int16Values(), remaps[c])
			case *array.Int32:
				remapCodes(out, idx.Int32Values(), remaps[c])
			case *array.Int64:
				remapCodes(out, idx.Int64Values(), remaps[c])
			case *array.Uint8:
				remapCodes(out, idx.Uint8Values(), remaps[c])
			case *array.Uint16:
				remapCodes(out, idx.Uint16Values(), remaps[c])
			case *array.Uint32:
				remapCodes(out, idx.Uint32Values(), remaps[c])
			case *array.Uint64:
				remapCodes(out, idx.Uint64Values(), remaps[c])
			}
			off += d.Len()
		}
	}
	if dictNulls {
		// A null dictionary entry is a missing value too.
		for i, k := range codes {
			if k < 0 {
				if valid == nil {
					valid = allValid(n)
				}
				valid[i>>6] &^= 1 << (uint(i) & 63)
			}
		}
	}
	zeroMissing(codes, valid)
	return rawcol.Categorical(name, codes, levels, valid).(*dataframe.Series), nil
}

// int32Indices returns the int32 indices of a dictionary chunk.
func int32Indices(c arrow.Array) []int32 {
	return c.(*array.Dictionary).Indices().(*array.Int32).Int32Values()
}

// zeroMissing writes 0 in the slots valid marks missing.
func zeroMissing(codes []int32, valid []uint64) {
	if valid == nil {
		return
	}
	for w, bits := range valid {
		for bits != math.MaxUint64 {
			b := mbits.TrailingZeros64(^bits)
			bits |= 1 << uint(b)
			if i := w<<6 + b; i < len(codes) {
				codes[i] = 0
			}
		}
	}
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
