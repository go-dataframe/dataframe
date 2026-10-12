package parquetio

import (
	"math"
	"math/bits"
	"unsafe"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/bitutil"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/apache/arrow-go/v18/parquet"
)

// The Parquet writer dictionary-encodes every column, as pyarrow's does,
// and gives a column chunk's dictionary up for the plain encoding once the
// dictionary reaches parquet.DefaultDictionaryPageSizeLimit bytes. Two
// things it does per value cost far more in Go than they need to, and the
// writer is given work that spares them, wherever the outcome is known in
// advance. The written data does not change; only how the pages lay it out.
//
//   - A numeric chunk with too many distinct values for the dictionary is
//     hashed into it value by value, until the dictionary overflows and the
//     values are written plain after all. When a prefix of every chunk of a
//     column already holds that many distinct values, the column is written
//     plain from the start.
//   - A string chunk is hashed value by value through an interface
//     conversion, one allocation each. When its distinct values fit the
//     dictionary, the chunk is handed over as an Arrow dictionary array
//     instead, the dictionary built here with a Go map, in order of first
//     appearance as the writer would build it, so that the writer only
//     encodes indices.
//
// Neither applies to uncompressed files, where the writer already gives up
// a dictionary that does not pay as soon as it sees a first batch.

// dictLimit is the dictionary size at which the writer falls back to plain.
const dictLimit = int(parquet.DefaultDictionaryPageSizeLimit)

// probeLen is how many leading values of a chunk the numeric probe looks
// at: enough to find dictLimit bytes of distinct 4-byte values twice over.
const probeLen = 2 * dictLimit / 4

// plainColumns returns the fields of rec whose every row-group chunk
// (rows [k*group, (k+1)*group)) has so many distinct values in its first
// probeLen that the writer would give its dictionary up.
func plainColumns(rec arrow.RecordBatch, group int64) []string {
	var set probeSet
	var names []string
	for i, f := range rec.Schema().Fields() {
		width := 0
		switch f.Type.ID() {
		case arrow.INT64, arrow.UINT64, arrow.FLOAT64, arrow.TIMESTAMP:
			width = 8
		case arrow.INT32, arrow.FLOAT32:
			width = 4
		default:
			continue
		}
		col := rec.Column(i)
		doomed := rec.NumRows() > 0
		for lo := int64(0); lo < rec.NumRows() && doomed; lo += group {
			hi := min(lo+group, rec.NumRows())
			doomed = set.overflows(col, int(lo), int(hi), width)
		}
		if doomed {
			names = append(names, f.Name)
		}
	}
	return names
}

// probeSet is an open-addressing set of 64-bit keys, reused from chunk to
// chunk: a Go map would cost as much as the hashing it is there to spare.
type probeSet struct {
	slots []uint64 // key+1; 0 is an empty slot
	shift uint     // 64 - log2(len(slots))
	zero  bool     // whether the key math.MaxUint64 (stored as 0) is in
	n     int
}

// overflows reports whether the valid values of col in rows [lo, hi) count,
// within their first probeLen, dictLimit/width distinct values or more:
// enough for the writer's dictionary to reach its limit (see floatKey).
func (s *probeSet) overflows(col arrow.Array, lo, hi, width int) bool {
	need := dictLimit / width
	if hi-lo < need {
		return false
	}
	// At most half full when the probe stops.
	if size := 1 << bits.Len(uint(2*need)); len(s.slots) < size {
		s.slots = make([]uint64, size)
		s.shift = uint(64 - bits.Len(uint(size-1)))
	} else {
		clear(s.slots)
	}
	s.zero, s.n = false, 0
	key := keyFunc(col)
	end := min(hi, lo+probeLen)
	// A first look at a short prefix: a column that repeats itself early
	// is not worth probing further, and keeps its dictionary.
	early := min(end, lo+4096)
	for i := lo; i < end; i++ {
		if col.IsValid(i) {
			s.add(key(i))
			if s.n >= need {
				return true
			}
		}
		if i == early && s.n < (early-lo)/2 {
			return false
		}
	}
	return false
}

func (s *probeSet) add(k uint64) {
	k++
	if k == 0 {
		if !s.zero {
			s.zero = true
			s.n++
		}
		return
	}
	mask := uint64(len(s.slots) - 1)
	for h := (k * 0x9E3779B97F4A7C15) >> s.shift; ; h = (h + 1) & mask {
		switch s.slots[h] {
		case k:
			return
		case 0:
			s.slots[h] = k
			s.n++
			return
		}
	}
}

// keyFunc returns the 64-bit key of each value of a numeric array.
func keyFunc(col arrow.Array) func(int) uint64 {
	switch a := col.(type) {
	case *array.Int64:
		v := a.Int64Values()
		return func(i int) uint64 { return uint64(v[i]) }
	case *array.Uint64:
		v := a.Uint64Values()
		return func(i int) uint64 { return v[i] }
	case *array.Timestamp:
		v := a.TimestampValues()
		return func(i int) uint64 { return uint64(v[i]) }
	case *array.Float64:
		v := a.Float64Values()
		return func(i int) uint64 { return floatKey(v[i]) }
	case *array.Int32:
		v := a.Int32Values()
		return func(i int) uint64 { return uint64(uint32(v[i])) }
	}
	v := col.(*array.Float32).Float32Values()
	return func(i int) uint64 { return floatKey(float64(v[i])) }
}

// floatKey is the key of a float: its bits, with -0 counted as 0 and every
// NaN as one, so that two floats the writer may hash apart never count as
// two when it might count them as one.
func floatKey(f float64) uint64 {
	if f != f {
		return math.Float64bits(math.NaN())
	}
	return math.Float64bits(f + 0)
}

// dictType is the type of the dictionary arrays string chunks become.
var dictType = &arrow.DictionaryType{IndexType: arrow.PrimitiveTypes.Int32, ValueType: arrow.BinaryTypes.String}

// stringDict returns rows [lo, hi) of a utf8 array as a dictionary array
// whose dictionary holds each distinct valid value once, in order of first
// appearance, or false when those values reach dictLimit bytes as the
// writer counts them (each value's length plus 4), so that the writer
// would give the dictionary up, or when one of them is empty.
func stringDict(col *array.String, lo, hi int) (arrow.Array, bool) {
	offs, data := col.ValueOffsets(), col.ValueBytes()
	base := offs[0]
	text := unsafe.String(unsafe.SliceData(data), len(data))
	n := hi - lo
	codes := make([]int32, n)
	pos := make(map[string]int32, 256)
	dictOffs := []int32{0}
	var dictData []byte
	size := 0
	for i := lo; i < hi; i++ {
		if col.IsNull(i) {
			continue
		}
		v := text[offs[i]-base : offs[i+1]-base]
		if v == "" {
			// The writer's statistics skip empty strings in a dictionary
			// (arrow-go v18.8.0, ByteArrayStatistics.UpdateFromArrow): a
			// chunk holding one would get a wrong minimum.
			return nil, false
		}
		k, ok := pos[v]
		if !ok {
			size += len(v) + 4
			if size >= dictLimit {
				return nil, false
			}
			k = int32(len(pos))
			pos[v] = k
			dictData = append(dictData, v...)
			dictOffs = append(dictOffs, int32(len(dictData)))
		}
		codes[i-lo] = k
	}
	var valid *memory.Buffer
	nulls := 0
	if col.NullN() > 0 {
		b := make([]byte, bitutil.BytesForBits(int64(n)))
		bitutil.CopyBitmap(col.NullBitmapBytes(), col.Data().Offset()+lo, n, b, 0)
		valid = memory.NewBufferBytes(b)
		nulls = n - bitutil.CountSetBits(b, 0, n)
	}
	idx := array.NewData(arrow.PrimitiveTypes.Int32, n, []*memory.Buffer{valid, memory.NewBufferBytes(int32Bytes(codes))}, nil, nulls, 0)
	defer idx.Release()
	dict := array.NewData(arrow.BinaryTypes.String, len(dictOffs)-1,
		[]*memory.Buffer{nil, memory.NewBufferBytes(int32Bytes(dictOffs)), memory.NewBufferBytes(dictData)}, nil, 0, 0)
	defer dict.Release()
	dictArr := array.MakeFromData(dict)
	defer dictArr.Release()
	idxData := array.NewDataWithDictionary(dictType, n, idx.Buffers(), nulls, 0, dictArr.Data().(*array.Data))
	defer idxData.Release()
	return array.MakeFromData(idxData), true
}

// int32Bytes views int32s as bytes, without copying.
func int32Bytes(d []int32) []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(d))), 4*len(d))
}
