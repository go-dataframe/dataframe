package arrowio

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/bits"
	mbits "math/bits"
	"strings"
	"unsafe"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/go-dataframe/dataframe"
	"github.com/go-dataframe/dataframe/internal/rawcol"
	"github.com/go-ndarray/ndarray"
)

// ToRecord converts a frame to an Arrow record batch: one field per column,
// in order, then (unless WithoutIndex) one field per level of a labelled
// index, and "pandas" schema metadata describing the index and the dtypes,
// so that pandas reads the data back as the same frame. Numeric and datetime
// columns share the frame's storage rather than copying it. The caller owns
// the record and must Release it.
func ToRecord(df *dataframe.DataFrame, opts ...Option) (arrow.RecordBatch, error) {
	return toRecord(df, newConfig(opts))
}

func toRecord(df *dataframe.DataFrame, cfg *config) (arrow.RecordBatch, error) {
	var (
		fields []arrow.Field
		arrs   []arrow.Array
		metas  []pandasColumn
		index  []any
	)
	release := func() {
		for _, a := range arrs {
			a.Release()
		}
	}
	add := func(s *dataframe.Series, field string, name *string) error {
		a, masked, err := toArray(s, cfg)
		if err != nil {
			return fmt.Errorf("column %q: %w", field, err)
		}
		m := pandasType(s, masked, cfg.timeZone)
		m.Name, m.FieldName = name, &field
		fields = append(fields, arrow.Field{Name: field, Type: a.DataType(), Nullable: true})
		arrs = append(arrs, a)
		metas = append(metas, m)
		return nil
	}
	for _, s := range df.Series() {
		name := s.Name()
		if err := add(s, name, &name); err != nil {
			release()
			return nil, err
		}
	}
	if cfg.index {
		ix := df.Index()
		if ix.IsRange() {
			index = []any{rangeOf(ix)}
		} else {
			for k, name := range ix.Names() {
				lv, _ := ix.Level(k)
				field := name
				if field == "" {
					field = fmt.Sprintf("__index_level_%d__", k)
				}
				if err := add(lv, field, nameOrNull(name)); err != nil {
					release()
					return nil, err
				}
				index = append(index, field)
			}
		}
	}
	md := arrow.NewMetadata([]string{pandasKey}, []string{encodePandas(metas, index)})
	rec := array.NewRecordBatch(arrow.NewSchema(fields, &md), arrs, int64(df.Len()))
	release()
	return rec, nil
}

// rangeOf returns the descriptor of a range index.
func rangeOf(ix *dataframe.Index) rangeIndex {
	r := rangeIndex{Kind: "range", Name: nameOrNull(ix.Name()), Step: 1}
	n := int64(ix.Len())
	if n > 0 {
		r.Start = ix.Label(0).(int64)
	}
	if n > 1 {
		r.Step = ix.Label(1).(int64) - r.Start
	}
	r.Stop = r.Start + n*r.Step
	return r
}

// toArray converts one series to an Arrow array, and reports whether it has
// missing values marked in a validity bitmap (rather than in-band NaN).
func toArray(s *dataframe.Series, cfg *config) (arrow.Array, bool, error) {
	n := s.Len()
	valid := s.ValidityBitmap()
	masked := valid != nil
	var a arrow.Array
	switch s.DType() {
	case dataframe.Float64:
		d := data[float64](s)
		a = fixed(arrow.PrimitiveTypes.Float64, d, nanValidity(d, valid), n)
	case dataframe.Float32:
		d := data[float32](s)
		a = fixed(arrow.PrimitiveTypes.Float32, d, nanValidity(d, valid), n)
	case dataframe.Int8:
		a = fixed(arrow.PrimitiveTypes.Int8, data[int8](s), valid, n)
	case dataframe.Int16:
		a = fixed(arrow.PrimitiveTypes.Int16, data[int16](s), valid, n)
	case dataframe.Int32:
		a = fixed(arrow.PrimitiveTypes.Int32, data[int32](s), valid, n)
	case dataframe.Int64:
		a = fixed(arrow.PrimitiveTypes.Int64, data[int64](s), valid, n)
	case dataframe.Uint8:
		a = fixed(arrow.PrimitiveTypes.Uint8, data[uint8](s), valid, n)
	case dataframe.Uint16:
		a = fixed(arrow.PrimitiveTypes.Uint16, data[uint16](s), valid, n)
	case dataframe.Uint32:
		a = fixed(arrow.PrimitiveTypes.Uint32, data[uint32](s), valid, n)
	case dataframe.Uint64:
		a = fixed(arrow.PrimitiveTypes.Uint64, data[uint64](s), valid, n)
	case dataframe.Bool:
		a = boolArray(data[bool](s), valid)
	case dataframe.Datetime:
		t := &arrow.TimestampType{Unit: arrow.Nanosecond, TimeZone: cfg.timeZone}
		a = fixed(t, data[int64](s), valid, n)
	case dataframe.String, dataframe.Period:
		a = textArray(s, valid)
	case dataframe.Category:
		a = dictArray(s)
	default:
		return nil, false, fmt.Errorf("%w: dtype %v has no Arrow mapping", ErrUnsupported, s.DType())
	}
	return a, masked, nil
}

// data returns a numeric series' storage, shared, as a typed slice. T must
// be the series' element type.
func data[T any](s *dataframe.Series) []T {
	a, _ := s.Values()
	var d any
	switch any((*T)(nil)).(type) {
	case *float64:
		d, _ = ndarray.Data[float64](a)
	case *float32:
		d, _ = ndarray.Data[float32](a)
	case *int8:
		d, _ = ndarray.Data[int8](a)
	case *int16:
		d, _ = ndarray.Data[int16](a)
	case *int32:
		d, _ = ndarray.Data[int32](a)
	case *int64:
		d, _ = ndarray.Data[int64](a)
	case *uint8:
		d, _ = ndarray.Data[uint8](a)
	case *uint16:
		d, _ = ndarray.Data[uint16](a)
	case *uint32:
		d, _ = ndarray.Data[uint32](a)
	case *uint64:
		d, _ = ndarray.Data[uint64](a)
	case *bool:
		d, _ = ndarray.Data[bool](a)
	}
	return d.([]T)
}

// bytesOf views a slice of fixed-width values as bytes, without copying.
func bytesOf[T any](d []T) []byte {
	var zero T
	return unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(d))), len(d)*int(unsafe.Sizeof(zero)))
}

// fixed wraps a slice of fixed-width values as an Arrow array, sharing it.
func fixed[T any](dt arrow.DataType, d []T, valid []uint64, n int) arrow.Array {
	vb, nulls := validityBuffer(valid, n)
	ad := array.NewData(dt, n, []*memory.Buffer{vb, memory.NewBufferBytes(bytesOf(d))}, nil, nulls, 0)
	defer ad.Release()
	return array.MakeFromData(ad)
}

// validityBuffer converts a validity bitmap (64-bit words, least significant
// bit first) to an Arrow validity buffer (bytes, least significant bit
// first) and counts the missing values among the first n. Encoding each
// word little-endian makes the byte order right on every machine.
func validityBuffer(valid []uint64, n int) (*memory.Buffer, int) {
	if valid == nil {
		return nil, 0
	}
	b := make([]byte, 8*len(valid))
	present := 0
	for i, w := range valid {
		binary.LittleEndian.PutUint64(b[8*i:], w)
		if r := n - 64*i; r < 64 {
			w &= 1<<uint(r) - 1
		}
		present += bits.OnesCount64(w)
	}
	return memory.NewBufferBytes(b), n - present
}

// allValid returns a bitmap of n set bits.
func allValid(n int) []uint64 {
	v := make([]uint64, (n+63)>>6)
	for i := range v {
		v[i] = math.MaxUint64
	}
	return v
}

// nanValidity clears the validity bit of every NaN: Arrow keeps NaN apart
// from null, but a float column here, as in pandas, has NaN as its missing
// value, and pandas writes it as null.
func nanValidity[T float32 | float64](d []T, valid []uint64) []uint64 {
	for i, v := range d {
		if v != v {
			if valid == nil {
				valid = allValid(len(d))
			}
			valid[i>>6] &^= 1 << (uint(i) & 63)
		}
	}
	return valid
}

// boolArray packs bools into an Arrow boolean array.
func boolArray(d []bool, valid []uint64) arrow.Array {
	n := len(d)
	b := make([]byte, (n+7)>>3)
	for i, v := range d {
		if v {
			b[i>>3] |= 1 << (uint(i) & 7)
		}
	}
	vb, nulls := validityBuffer(valid, n)
	ad := array.NewData(arrow.FixedWidthTypes.Boolean, n, []*memory.Buffer{vb, memory.NewBufferBytes(b)}, nil, nulls, 0)
	defer ad.Release()
	return array.MakeFromData(ad)
}

// largeStrings is the byte size from which strings are written as
// large_utf8 (64-bit offsets) rather than utf8; a variable so that tests can
// take the large path without gigabytes of text.
var largeStrings = math.MaxInt32

// stringArray builds an Arrow utf8 array, or large_utf8 when the bytes
// overflow 32-bit offsets. A missing value is written empty, whatever strs
// holds in its slot.
func stringArray(strs []string, valid []uint64) arrow.Array {
	total := 0
	for i, s := range strs {
		if isValid(valid, i) {
			total += len(s)
		}
	}
	if total > largeStrings {
		return offsetStrings[int64](arrow.BinaryTypes.LargeString, strs, total, valid)
	}
	return offsetStrings[int32](arrow.BinaryTypes.String, strs, total, valid)
}

func offsetStrings[O int32 | int64](dt arrow.DataType, strs []string, total int, valid []uint64) arrow.Array {
	n := len(strs)
	offs := make([]O, n+1)
	buf := make([]byte, 0, total)
	for i, s := range strs {
		if isValid(valid, i) {
			buf = append(buf, s...)
		}
		offs[i+1] = O(len(buf))
	}
	vb, nulls := validityBuffer(valid, n)
	ad := array.NewData(dt, n, []*memory.Buffer{vb, memory.NewBufferBytes(bytesOf(offs)), memory.NewBufferBytes(buf)}, nil, nulls, 0)
	defer ad.Release()
	return array.MakeFromData(ad)
}

// text returns a string series' storage, shared, or a period series'
// values formatted.
// textArray converts a string or period series to an Arrow array, as
// stringArray lays its values out. A string series whose storage already
// is that layout (int32 offsets from 0, an empty value in every missing
// slot, utf8-sized) is shared rather than copied.
func textArray(s *dataframe.Series, valid []uint64) arrow.Array {
	if data, offs, _, ok := rawcol.Strings(s); ok {
		if o, ok := offs.([]int32); ok && o[0] == 0 && int(o[len(o)-1]) <= largeStrings && emptyWhereMissing(o, valid) {
			n := len(o) - 1
			buf := unsafe.Slice(unsafe.StringData(data), int(o[n]))
			vb, nulls := validityBuffer(valid, n)
			ad := array.NewData(arrow.BinaryTypes.String, n, []*memory.Buffer{vb, memory.NewBufferBytes(bytesOf(o)), memory.NewBufferBytes(buf)}, nil, nulls, 0)
			defer ad.Release()
			return array.MakeFromData(ad)
		}
	}
	return stringArray(s.Strings(), valid)
}

// emptyWhereMissing reports whether every value valid marks missing is
// empty.
func emptyWhereMissing(offs []int32, valid []uint64) bool {
	n := len(offs) - 1
	for w, bits := range valid {
		for bits != math.MaxUint64 {
			b := mbits.TrailingZeros64(^bits)
			bits |= 1 << uint(b)
			if i := w<<6 + b; i < n && offs[i] != offs[i+1] {
				return false
			}
		}
	}
	return true
}

// bitsSet counts the set bits of a bitmap.
func bitsSet(v []uint64) int {
	n := 0
	for _, w := range v {
		n += bits.OnesCount64(w)
	}
	return n
}

// dictType is the Arrow type of a category column.
var dictType = &arrow.DictionaryType{IndexType: arrow.PrimitiveTypes.Int32, ValueType: arrow.BinaryTypes.String}

// dictArray converts a category series to dictionary<int32, utf8>. The
// codes are shared when nothing is missing; otherwise they are copied, with
// 0 in the missing slots, and the validity bitmap with its bits past the
// end set.
func dictArray(s *dataframe.Series) arrow.Array {
	codes, valid, cats := rawcol.Codes(s)
	n := len(codes)
	if valid != nil {
		valid = append([]uint64(nil), valid[:(n+63)>>6]...)
		if r := n & 63; r != 0 {
			valid[len(valid)-1] |= math.MaxUint64 << uint(r)
		}
		if bitsSet(valid) == 64*len(valid) {
			valid = nil
		}
	}
	if valid != nil {
		codes = append([]int32(nil), codes...)
		for i := range codes {
			if !isValid(valid, i) {
				codes[i] = 0
			}
		}
	}
	idx := fixed(arrow.PrimitiveTypes.Int32, codes, valid, len(codes))
	defer idx.Release()
	dict := offsetStrings[int32](arrow.BinaryTypes.String, cats, len(strings.Join(cats, "")), nil)
	defer dict.Release()
	return array.NewDictionaryArray(dictType, idx, dict)
}
