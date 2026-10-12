package arrowio

import (
	"bytes"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"unsafe"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/go-dataframe/dataframe"
	"github.com/go-dataframe/dataframe/internal/rawcol"
	"github.com/go-ndarray/ndarray"
)

// identical fails unless got and want hold the same columns, dtypes,
// validity bitmaps, values and index, and the same bits in every numeric
// slot, missing ones included: a frame that keeps the reader's buffers must
// be the frame a copying read makes, not merely an equal-looking one.
func identical(t *testing.T, where string, got, want *dataframe.DataFrame) {
	t.Helper()
	equalFrames(t, where, got, want)
	gs, ws := got.Series(), want.Series()
	for i := range gs {
		g, w := gs[i], ws[i]
		if !slices.Equal(g.ValidityBitmap(), w.ValidityBitmap()) {
			t.Errorf("%s:%s: validity %x, want %x", where, g.Name(), g.ValidityBitmap(), w.ValidityBitmap())
		}
		if !slices.Equal(g.Strings(), w.Strings()) {
			t.Errorf("%s:%s: text differs", where, g.Name())
		}
		if g.DType() == dataframe.Category {
			gc, gv, gl := rawcol.Codes(g)
			wc, wv, wl := rawcol.Codes(w)
			if !slices.Equal(gc, wc) || !slices.Equal(gv, wv) || !slices.Equal(gl, wl) {
				t.Errorf("%s:%s: category storage differs", where, g.Name())
			}
		}
		ga, gerr := g.Values()
		wa, werr := w.Values()
		if (gerr == nil) != (werr == nil) {
			t.Fatalf("%s:%s: Values errors %v, %v", where, g.Name(), gerr, werr)
		}
		if gerr == nil && !bytes.Equal(storageBytes(ga), storageBytes(wa)) {
			t.Errorf("%s:%s: storage bytes differ", where, g.Name())
		}
	}
}

// storageBytes returns the bytes of an array's elements.
func storageBytes(a *ndarray.Array) []byte {
	switch a.DType() {
	case ndarray.Float64:
		return bytesOf(dataOf[float64](a))
	case ndarray.Float32:
		return bytesOf(dataOf[float32](a))
	case ndarray.Int8:
		return bytesOf(dataOf[int8](a))
	case ndarray.Int16:
		return bytesOf(dataOf[int16](a))
	case ndarray.Int32:
		return bytesOf(dataOf[int32](a))
	case ndarray.Int64:
		return bytesOf(dataOf[int64](a))
	case ndarray.Uint8:
		return bytesOf(dataOf[uint8](a))
	case ndarray.Uint16:
		return bytesOf(dataOf[uint16](a))
	case ndarray.Uint32:
		return bytesOf(dataOf[uint32](a))
	case ndarray.Uint64:
		return bytesOf(dataOf[uint64](a))
	}
	return bytesOf(dataOf[bool](a))
}

func dataOf[T float64 | float32 | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | bool](a *ndarray.Array) []T {
	d, _ := ndarray.Data[T](a)
	return d
}

// copying is an allocator that is not the Go allocator, so that a read
// with it copies every buffer, as all reads did before buffers were handed
// over.
func copying() memory.Allocator { return memory.NewCheckedAllocator(memory.NewGoAllocator()) }

// Every fixture, and every round trip, reads the same through the
// zero-copy path (the default Go allocator) as through the copying one.
func TestAdoptMatchesCopy(t *testing.T) {
	files, _ := filepath.Glob(filepath.Join("testdata", "*.feather"))
	streams, _ := filepath.Glob(filepath.Join("testdata", "*.arrows"))
	for _, path := range append(files, streams...) {
		if strings.HasPrefix(filepath.Base(path), "bad_") {
			continue
		}
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		read := func(mem memory.Allocator) (*dataframe.DataFrame, error) {
			if strings.HasSuffix(path, ".arrows") {
				return ReadIPCStream(bytes.NewReader(data), WithAllocator(mem))
			}
			return ReadIPCFile(bytes.NewReader(data), WithAllocator(mem))
		}
		got, gerr := read(memory.DefaultAllocator)
		want, werr := read(copying())
		if gerr != nil || werr != nil {
			t.Fatalf("%s: %v, %v", path, gerr, werr)
		}
		identical(t, path, got, want)
	}
	frames := map[string]*dataframe.DataFrame{
		"all types": allTypes(t),
		"indexed":   dataframe.Must(allTypes(t).SetIndex("i64", "s")),
		"empty":     allTypes(t).Head(0),
		"bench":     benchFrame(1000),
	}
	for name, df := range frames {
		for _, comp := range []Compression{Uncompressed, LZ4, Zstd} {
			var b bytes.Buffer
			if err := WriteIPCFile(&b, df, WithCompression(comp)); err != nil {
				t.Fatal(err)
			}
			got := dataframe.Must(ReadIPCFile(bytes.NewReader(b.Bytes())))
			want := dataframe.Must(ReadIPCFile(bytes.NewReader(b.Bytes()), WithAllocator(copying())))
			identical(t, fmt.Sprintf("%s %v", name, comp), got, want)
			equalFrames(t, fmt.Sprintf("%s %v vs source", name, comp), got, df)
		}
	}
}

// A frame that keeps a reader's buffers is not disturbed by later reads,
// nor by the source bytes changing afterwards.
func TestAdoptedFramesStayPut(t *testing.T) {
	var b bytes.Buffer
	if err := WriteIPCFile(&b, allTypes(t)); err != nil {
		t.Fatal(err)
	}
	data := b.Bytes()
	want := dataframe.Must(ReadIPCFile(bytes.NewReader(data), WithAllocator(copying())))
	first := dataframe.Must(ReadIPCFile(bytes.NewReader(data)))
	_ = dataframe.Must(ReadIPCFile(bytes.NewReader(data)))
	for i := range data {
		data[i] ^= 0x5a
	}
	_, _ = ReadIPCFile(bytes.NewReader(data))
	identical(t, "after", first, want)
}

// A column whose values are not aligned for their type is copied, not
// handed over: a Go slice must be aligned.
func TestAdoptMisaligned(t *testing.T) {
	raw := make([]byte, 8*4+1)
	want := []int64{1, -2, 3, math.MaxInt64}
	for i, v := range want {
		copy(raw[1+8*i:], bytesOf([]int64{v}))
	}
	buf := raw[1:]
	if aligned(unsafe.Slice((*int64)(unsafe.Pointer(&buf[0])), 4)) {
		t.Skip("the allocator handed out an odd buffer; nothing to test")
	}
	ad := array.NewData(arrow.PrimitiveTypes.Int64, 4, []*memory.Buffer{nil, memory.NewBufferBytes(buf)}, nil, 0, 0)
	defer ad.Release()
	arr := array.MakeFromData(ad)
	defer arr.Release()
	s, err := toSeries(arrow.Field{Name: "x", Type: arrow.PrimitiveTypes.Int64}, []arrow.Array{arr}, 4, true)
	if err != nil {
		t.Fatal(err)
	}
	equalItems(t, "misaligned", s.Items(), []any{int64(1), int64(-2), int64(3), int64(math.MaxInt64)})
	if !aligned([]int64{0}) {
		t.Error("a Go slice reads as misaligned")
	}
}

// owns hands buffers over for the Go allocator only.
func TestOwns(t *testing.T) {
	if !owns(memory.NewGoAllocator()) || owns(copying()) {
		t.Error("owns")
	}
}

// A period index is written as its formatted values, read back as strings.
func TestPeriodIndex(t *testing.T) {
	df := dataframe.Must(allTypes(t).SetIndex("ts"))
	df = dataframe.Must(dataframe.Must(df.Head(3).ResetIndex(false).SetIndex("ts")).ToPeriod("M"))
	lv, _ := df.Index().Level(0)
	if lv.DType() != dataframe.Period {
		t.Fatalf("index dtype %v", lv.DType())
	}
	var b bytes.Buffer
	if err := WriteIPCFile(&b, df); err != nil {
		t.Fatal(err)
	}
	got := dataframe.Must(ReadIPCFile(bytes.NewReader(b.Bytes())))
	gl, _ := got.Index().Level(0)
	if !slices.Equal(gl.Strings(), lv.Strings()) {
		t.Errorf("period labels %q, want %q", gl.Strings(), lv.Strings())
	}
}

// A null dictionary entry makes the values coded to it missing, even when
// no index is null.
func TestDictionaryNullEntry(t *testing.T) {
	vb := array.NewStringBuilder(memory.DefaultAllocator)
	defer vb.Release()
	vb.AppendValues([]string{"a", "", "b"}, []bool{true, false, true})
	dict := vb.NewArray()
	defer dict.Release()
	idx := fixed(arrow.PrimitiveTypes.Int32, []int32{2, 1, 0}, nil, 3)
	defer idx.Release()
	dt := &arrow.DictionaryType{IndexType: arrow.PrimitiveTypes.Int32, ValueType: arrow.BinaryTypes.String}
	col := array.NewDictionaryArray(dt, idx, dict)
	defer col.Release()
	s, err := toSeries(arrow.Field{Name: "d", Type: dt}, []arrow.Array{col}, 3, true)
	if err != nil {
		t.Fatal(err)
	}
	equalItems(t, "null entry", s.Items(), []any{"b", nil, "a"})
	if codes, _, _ := rawcol.Codes(s); !slices.Equal(codes, []int32{1, 0, 0}) {
		t.Errorf("codes %v", codes)
	}
}
