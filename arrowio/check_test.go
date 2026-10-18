package arrowio

import (
	"bytes"
	"encoding/binary"
	"errors"
	"math"
	"strings"
	"testing"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/go-dataframe/dataframe/internal/arrowfb"
)

// mk builds an array straight from buffers, as an IPC reader does from
// what a file declares.
func mk(dt arrow.DataType, n, nulls int, bufs ...[]byte) arrow.Array {
	mb := make([]*memory.Buffer, len(bufs))
	for i, b := range bufs {
		if b != nil {
			mb[i] = memory.NewBufferBytes(b)
		}
	}
	d := array.NewData(dt, n, mb, nil, nulls, 0)
	defer d.Release()
	return array.MakeFromData(d)
}

func i32s(v ...int32) []byte { return bytesOf(v) }

// Every way a record can declare more than its buffers hold, or values out
// of order or range, is ErrCorrupt.
func TestCheckArray(t *testing.T) {
	dict := func(idx arrow.Array, values arrow.Array) arrow.Array {
		dt := &arrow.DictionaryType{IndexType: idx.DataType(), ValueType: values.DataType()}
		return array.NewDictionaryArray(dt, idx, values)
	}
	strs := mk(arrow.BinaryTypes.String, 2, 0, nil, i32s(0, 1, 2), []byte("ab"))
	cases := map[string]arrow.Array{
		"negative length":       mk(arrow.Null, -1, 0),
		"short validity":        mk(arrow.PrimitiveTypes.Float64, 10, 1, []byte{0xff}, make([]byte, 80)),
		"nulls without bitmap":  mk(arrow.PrimitiveTypes.Float64, 2, 1, nil, make([]byte, 16)),
		"no values":             mk(arrow.PrimitiveTypes.Float64, 2, 0, nil, nil),
		"short booleans":        mk(arrow.FixedWidthTypes.Boolean, 20, 0, nil, []byte{1}),
		"offsets out of order":  mk(arrow.BinaryTypes.String, 3, 0, nil, i32s(0, 3, 1, 4), []byte("abcd")),
		"negative offset":       mk(arrow.BinaryTypes.String, 1, 0, nil, i32s(-1, 2), []byte("ab")),
		"large out of order":    mk(arrow.BinaryTypes.LargeString, 2, 0, nil, bytesOf([]int64{0, 2, 1}), []byte("ab")),
		"bad dictionary values": dict(mk(arrow.PrimitiveTypes.Int32, 1, 0, nil, i32s(0)), mk(arrow.BinaryTypes.String, 2, 0, nil, i32s(0, 2, 1), []byte("ab"))),
	}
	for name, idx := range map[string]arrow.Array{
		"int8":   mk(arrow.PrimitiveTypes.Int8, 1, 0, nil, []byte{5}),
		"int16":  mk(arrow.PrimitiveTypes.Int16, 1, 0, nil, bytesOf([]int16{-1})),
		"int32":  mk(arrow.PrimitiveTypes.Int32, 1, 0, nil, i32s(2)),
		"int64":  mk(arrow.PrimitiveTypes.Int64, 1, 0, nil, bytesOf([]int64{9})),
		"uint8":  mk(arrow.PrimitiveTypes.Uint8, 1, 0, nil, []byte{2}),
		"uint16": mk(arrow.PrimitiveTypes.Uint16, 1, 0, nil, bytesOf([]uint16{2})),
		"uint32": mk(arrow.PrimitiveTypes.Uint32, 1, 0, nil, bytesOf([]uint32{2})),
		"uint64": mk(arrow.PrimitiveTypes.Uint64, 1, 0, nil, bytesOf([]uint64{2})),
	} {
		cases["index "+name] = dict(idx, strs)
	}
	for name, a := range cases {
		if err := checkArray(a); !errors.Is(err, ErrCorrupt) {
			t.Errorf("%s: %v", name, err)
		}
	}
	for name, a := range map[string]arrow.Array{
		"empty":   mk(arrow.PrimitiveTypes.Float64, 0, 0, nil, nil),
		"strings": strs,
		"dict":    dict(mk(arrow.PrimitiveTypes.Int8, 2, 0, nil, []byte{1, 0}), strs),
	} {
		if err := checkArray(a); err != nil {
			t.Errorf("%s: %v", name, err)
		}
	}
}

func TestCheckColumns(t *testing.T) {
	f := []arrow.Field{{Name: "n", Type: arrow.Null}}
	half := math.MaxInt/2 + 1
	for name, c := range map[string]struct {
		chunks []arrow.Array
		n      int
	}{
		"bad chunk":     {[]arrow.Array{mk(arrow.Null, -1, 0)}, 0},
		"overflow":      {[]arrow.Array{mk(arrow.Null, half, 0), mk(arrow.Null, half, 0)}, 0},
		"wrong length":  {[]arrow.Array{mk(arrow.Null, 3, 0)}, 4},
		"too many null": {[]arrow.Array{mk(arrow.Null, maxNullRows+1, 0)}, maxNullRows + 1},
	} {
		if err := checkColumns(f, [][]arrow.Array{c.chunks}, c.n); !errors.Is(err, ErrCorrupt) {
			t.Errorf("%s: %v", name, err)
		}
	}
	// FromRecord checks too, before converting anything.
	schema := arrow.NewSchema([]arrow.Field{{Name: "s", Type: arrow.BinaryTypes.String}}, nil)
	rec := array.NewRecordBatch(schema, []arrow.Array{mk(arrow.BinaryTypes.String, 3, 0, nil, i32s(0, 3, 1, 4), []byte("abcd"))}, 3)
	if _, err := FromRecord(rec); !errors.Is(err, ErrCorrupt) {
		t.Errorf("FromRecord: %v", err)
	}
}

// frame prefixes metadata with the continuation marker and its length.
func frame(meta []byte) []byte {
	out := binary.LittleEndian.AppendUint32(nil, 0xFFFFFFFF)
	out = binary.LittleEndian.AppendUint32(out, uint32(len(meta)))
	return append(out, meta...)
}

// A stream whose framing, metadata or body is not what it declares fails
// with ErrCorrupt, whatever part fails.
func TestStreamMessages(t *testing.T) {
	var ok bytes.Buffer
	if err := WriteIPCStream(&ok, allTypes(t)); err != nil {
		t.Fatal(err)
	}
	good := ok.Bytes()
	metaLen := int(binary.LittleEndian.Uint32(good[4:]))
	for name, b := range map[string][]byte{
		"continuation, then nothing": {0xFF, 0xFF, 0xFF, 0xFF},
		"a 3-byte metadata":          frame([]byte{1, 2, 3}),
		"metadata past the end":      append(binary.LittleEndian.AppendUint32(nil, 100), 1, 2, 3, 4),
		"invalid flatbuffer":         frame([]byte{0xff, 0xff, 0xff, 0x7f, 0, 0, 0, 0}),
		"one byte":                   {1},
		"metadata of 1 GiB":          binary.LittleEndian.AppendUint32([]byte{0xFF, 0xFF, 0xFF, 0xFF}, 1<<30),
		"body cut short":             good[:len(good)/2],
	} {
		if _, err := ReadIPCStream(bytes.NewReader(b)); !errors.Is(err, ErrCorrupt) {
			t.Errorf("%s: %v", name, err)
		}
	}
	// A record batch message declaring a body beyond the limit.
	rest := good[8+metaLen:]
	bLen := int(binary.LittleEndian.Uint32(rest[4:]))
	batch := bytes.Clone(rest[8 : 8+bLen])
	n, _ := arrowfb.CheckMessage(batch)
	patched := bytes.Replace(batch, binary.LittleEndian.AppendUint64(nil, uint64(n)), binary.LittleEndian.AppendUint64(nil, maxBody+1), 1)
	b := append(append(bytes.Clone(good[:8+metaLen]), frame(patched)...), rest[8+bLen:]...)
	if _, err := ReadIPCStream(bytes.NewReader(b)); err == nil || !strings.Contains(err.Error(), "body") {
		t.Errorf("a body beyond the limit: %v", err)
	}
	m := newCheckedMessages(bytes.NewReader(good))
	m.Retain()
	m.Release()
	m.Release()
}

// failingFile is a ReadAtSeeker that fails at a chosen call.
type failingFile struct {
	*bytes.Reader
	seeks, reads int
	failSeek     int
	failRead     int
}

var errIO = errors.New("i/o failed")

func (f *failingFile) Seek(off int64, whence int) (int64, error) {
	if f.seeks++; f.seeks == f.failSeek {
		return 0, errIO
	}
	return f.Reader.Seek(off, whence)
}

func (f *failingFile) ReadAt(p []byte, off int64) (int, error) {
	if f.reads++; f.reads == f.failRead {
		return 0, errIO
	}
	return f.Reader.ReadAt(p, off)
}

// An IPC file whose footer, blocks or messages are not what they declare
// fails before arrow-go reads them; so does a file that cannot be read.
func TestFileChecks(t *testing.T) {
	var ok bytes.Buffer
	if err := WriteIPCFile(&ok, allTypes(t)); err != nil {
		t.Fatal(err)
	}
	good := ok.Bytes()
	for _, f := range []*failingFile{{failSeek: 1}, {failSeek: 2}, {failRead: 1}, {failRead: 2}, {failRead: 3}} {
		f.Reader = bytes.NewReader(good)
		if _, err := ReadIPCFile(f); err == nil {
			t.Errorf("%+v: no error", *f)
		}
	}
	n := len(good)
	flen := int(binary.LittleEndian.Uint32(good[n-10:]))
	setLen := func(v uint32) []byte {
		b := bytes.Clone(good)
		binary.LittleEndian.PutUint32(b[n-10:], v)
		return b
	}
	footerAt := n - 10 - flen
	for name, b := range map[string][]byte{
		"footer longer than the file": setLen(uint32(n)),
		"footer of zero bytes":        setLen(0),
		"footer not a flatbuffer":     append(append(bytes.Clone(good[:footerAt]), bytes.Repeat([]byte{0xff}, flen)...), good[n-10:]...),
	} {
		if _, err := ReadIPCFile(bytes.NewReader(b)); !errors.Is(err, ErrCorrupt) {
			t.Errorf("%s: %v", name, err)
		}
	}
	// The block of the record batch: its offset moved off the end, then
	// onto bytes that are not a message.
	footer := good[footerAt : n-10]
	blocks := arrowfb.FooterBlocks(footer)
	blk := blocks[len(blocks)-1]
	for name, off := range map[string]int64{"past the end": int64(n), "onto garbage": int64(footerAt)} {
		b := bytes.Clone(good)
		i := bytes.Index(b[footerAt:n-10], binary.LittleEndian.AppendUint64(nil, uint64(blk.Offset)))
		binary.LittleEndian.PutUint64(b[footerAt+i:], uint64(off))
		if _, err := ReadIPCFile(bytes.NewReader(b)); !errors.Is(err, ErrCorrupt) {
			t.Errorf("block %s: %v", name, err)
		}
	}
	// Bytes too short to be a file, or without the magic, are arrow-go's
	// to refuse.
	for _, b := range [][]byte{[]byte("short"), append(bytes.Repeat([]byte{0}, 32), "NOTARW"...)} {
		if _, err := ReadIPCFile(bytes.NewReader(b)); err == nil {
			t.Errorf("%q read", b)
		}
	}
	// A panic of the decoders is an error.
	if _, err := ReadIPCFile(panicky{}); !errors.Is(err, ErrCorrupt) {
		t.Errorf("a panicking reader: %v", err)
	}
}

type panicky struct{}

func (panicky) Read([]byte) (int, error)          { panic("read") }
func (panicky) ReadAt([]byte, int64) (int, error) { panic("read at") }
func (panicky) Seek(int64, int) (int64, error)    { return 1 << 20, nil }
