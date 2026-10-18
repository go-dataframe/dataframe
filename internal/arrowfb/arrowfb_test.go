package arrowfb

import (
	"bytes"
	"encoding/binary"
	"errors"
	"strings"
	"testing"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/flight"
	"github.com/apache/arrow-go/v18/arrow/ipc"
	"github.com/apache/arrow-go/v18/arrow/memory"
	flatbuffers "github.com/google/flatbuffers/go"
)

// richSchema has every part of a schema the checker walks: field names,
// field and schema metadata, a timestamp zone, union type ids, a
// dictionary encoding, nested children.
func richSchema() *arrow.Schema {
	md := arrow.NewMetadata([]string{"k"}, []string{"v"})
	return arrow.NewSchema([]arrow.Field{
		{Name: "a", Type: arrow.PrimitiveTypes.Int64, Metadata: md},
		{Name: "t", Type: &arrow.TimestampType{Unit: arrow.Nanosecond, TimeZone: "Europe/Paris"}},
		{Name: "d", Type: &arrow.DictionaryType{IndexType: arrow.PrimitiveTypes.Int32, ValueType: arrow.BinaryTypes.String}},
		{Name: "s", Type: arrow.StructOf(arrow.Field{Name: "x", Type: arrow.ListOf(arrow.PrimitiveTypes.Int8)})},
		{Name: "u", Type: arrow.DenseUnionOf([]arrow.Field{{Name: "i", Type: arrow.PrimitiveTypes.Int32}}, []arrow.UnionTypeCode{5})},
	}, &md)
}

// unframe strips a serialized message's continuation marker and length.
func unframe(t *testing.T, b []byte) []byte {
	t.Helper()
	if binary.LittleEndian.Uint32(b) == 0xFFFFFFFF {
		b = b[4:]
	}
	n := binary.LittleEndian.Uint32(b)
	return b[4 : 4+n]
}

// fileParts writes an LZ4-compressed IPC file of one batch, an int64 and a
// dictionary column, with schema metadata, and returns its footer and the
// metadata of its dictionary batch and record batch.
func fileParts(t *testing.T) (footer, dict, batch []byte) {
	t.Helper()
	md := arrow.NewMetadata([]string{"k"}, []string{"v"})
	dt := &arrow.DictionaryType{IndexType: arrow.PrimitiveTypes.Int32, ValueType: arrow.BinaryTypes.String}
	schema := arrow.NewSchema([]arrow.Field{{Name: "a", Type: arrow.PrimitiveTypes.Int64}, {Name: "d", Type: dt}}, &md)
	b := array.NewInt64Builder(memory.DefaultAllocator)
	b.AppendValues([]int64{1, 2, 3}, nil)
	col := b.NewArray()
	db := array.NewDictionaryBuilder(memory.DefaultAllocator, dt).(*array.BinaryDictionaryBuilder)
	for _, s := range []string{"x", "y", "x"} {
		_ = db.AppendString(s)
	}
	dcol := db.NewArray()
	rec := array.NewRecordBatch(schema, []arrow.Array{col, dcol}, 3)
	var buf bytes.Buffer
	w, err := ipc.NewFileWriter(&buf, ipc.WithSchema(schema), ipc.WithLZ4())
	if err != nil {
		t.Fatal(err)
	}
	if err := w.Write(rec); err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	data := buf.Bytes()
	n := int(binary.LittleEndian.Uint32(data[len(data)-10:]))
	footer = data[len(data)-10-n : len(data)-10]
	if err := CheckFooter(footer); err != nil {
		t.Fatalf("a valid footer: %v", err)
	}
	blocks := FooterBlocks(footer)
	if len(blocks) != 2 {
		t.Fatalf("%d blocks", len(blocks))
	}
	msg := func(blk Block) []byte { return unframe(t, data[blk.Offset:blk.Offset+int64(blk.MetaDataLength)]) }
	return footer, msg(blocks[0]), msg(blocks[1])
}

// Overwriting any 4 bytes of valid metadata with a large or negative
// number gives a buffer the checker passes or refuses with ErrInvalid,
// never a panic.
func TestOverwrites(t *testing.T) {
	schema := unframe(t, flight.SerializeSchema(richSchema(), memory.DefaultAllocator))
	footer, dict, batch := fileParts(t)
	msg := func(b []byte) error { _, err := CheckMessage(b); return err }
	for _, c := range []struct {
		b     []byte
		check func([]byte) error
	}{{schema, msg}, {dict, msg}, {batch, msg}, {footer, CheckFooter}} {
		if err := c.check(c.b); err != nil {
			t.Fatalf("valid metadata refused: %v", err)
		}
		for i := 0; i+4 <= len(c.b); i++ {
			for _, v := range []uint32{0x7fffffff, 0x80000000, 0xfffffff0, 0x00ff0000} {
				b := bytes.Clone(c.b)
				binary.LittleEndian.PutUint32(b[i:], v)
				if err := c.check(b); err != nil && !errors.Is(err, ErrInvalid) {
					t.Fatalf("%v at %d: %v is not ErrInvalid", v, i, err)
				}
			}
		}
	}
}

// Valid metadata passes; any truncation of it fails with ErrInvalid, and
// the truncations reach every kind of out-of-bounds the checker finds.
func TestTruncations(t *testing.T) {
	schema := unframe(t, flight.SerializeSchema(richSchema(), memory.DefaultAllocator))
	footer, _, batch := fileParts(t)
	if _, err := CheckMessage(schema); err != nil {
		t.Fatalf("a valid schema message: %v", err)
	}
	if n, err := CheckMessage(batch); err != nil || n <= 0 {
		t.Fatalf("a valid batch message: %d, %v", n, err)
	}
	seen := map[string]bool{}
	for _, c := range []struct {
		b     []byte
		check func([]byte) error
	}{
		{schema, func(b []byte) error { _, err := CheckMessage(b); return err }},
		{batch, func(b []byte) error { _, err := CheckMessage(b); return err }},
		{footer, CheckFooter},
	} {
		for n := range len(c.b) {
			err := c.check(bytes.Clone(c.b[:n]))
			if err == nil {
				continue
			}
			if !errors.Is(err, ErrInvalid) {
				t.Fatalf("truncated to %d: %v is not ErrInvalid", n, err)
			}
			msg := err.Error()
			for _, kind := range []string{"no root", "table at", "vtable", "offset at", "vector at", "element", "string at"} {
				if strings.Contains(msg, kind) {
					seen[kind] = true
				}
			}
		}
	}
	for _, kind := range []string{"no root", "table at", "vtable", "offset at", "vector at", "string at"} {
		if !seen[kind] {
			t.Errorf("no truncation found %q", kind)
		}
	}
}

// A vector of tables whose elements point out of the buffer, fields nested
// too deep, and more tables than the walk visits are each refused.
func TestHostile(t *testing.T) {
	// One field table, then a vector of offsets to it.
	build := func(n int, bad bool) []byte {
		b := flatbuffers.NewBuilder(0)
		name := b.CreateString("x")
		b.StartObject(7)
		b.PrependUOffsetTSlot(0, name, 0)
		field := b.EndObject()
		b.StartVector(4, n, 4)
		for range n {
			b.PrependUOffsetT(field)
		}
		vec := b.EndVector(n)
		b.StartObject(4)
		b.PrependUOffsetTSlot(1, vec, 0)
		schema := b.EndObject()
		b.StartObject(5)
		b.PrependByteSlot(1, 1, 0)
		b.PrependUOffsetTSlot(2, schema, 0)
		b.Finish(b.EndObject())
		out := b.FinishedBytes()
		if bad {
			// Point the vector's first element past the end.
			c := &checker{b: out}
			root, _ := c.root()
			sch, _, _ := c.indirect(root, 2)
			v, _ := c.vector(sch, 1, 4)
			binary.LittleEndian.PutUint32(out[v+4:], 1<<30)
		}
		return out
	}
	if _, err := CheckMessage(build(3, false)); err != nil {
		t.Fatalf("a valid hand-built message: %v", err)
	}
	if _, err := CheckMessage(build(1, true)); err == nil || !strings.Contains(err.Error(), "element") {
		t.Errorf("an element out of bounds: %v", err)
	}
	if _, err := CheckMessage(build(maxTables+1, false)); err == nil || !strings.Contains(err.Error(), "tables") {
		t.Errorf("too many tables: %v", err)
	}
	var deep arrow.DataType = arrow.PrimitiveTypes.Int8
	for range maxDepth + 2 {
		deep = arrow.StructOf(arrow.Field{Name: "c", Type: deep})
	}
	msg := unframe(t, flight.SerializeSchema(arrow.NewSchema([]arrow.Field{{Name: "deep", Type: deep}}, nil), memory.DefaultAllocator))
	if _, err := CheckMessage(msg); err == nil || !strings.Contains(err.Error(), "deeper") {
		t.Errorf("nesting too deep: %v", err)
	}
}

// A footer with no dictionaries vector lists its record batches alone.
func TestFooterWithoutDictionaries(t *testing.T) {
	b := flatbuffers.NewBuilder(0)
	b.StartVector(24, 1, 8)
	b.PrependInt64(99) // bodyLength
	b.PrependInt32(0)  // padding
	b.PrependInt32(64) // metaDataLength
	b.PrependInt64(8)  // offset
	batches := b.EndVector(1)
	b.StartObject(5)
	b.PrependUOffsetTSlot(3, batches, 0)
	b.Finish(b.EndObject())
	footer := b.FinishedBytes()
	if err := CheckFooter(footer); err != nil {
		t.Fatal(err)
	}
	got := FooterBlocks(footer)
	if len(got) != 1 || got[0] != (Block{Offset: 8, MetaDataLength: 64, BodyLength: 99}) {
		t.Errorf("blocks %+v", got)
	}
}

func TestCheckFramed(t *testing.T) {
	framed := flight.SerializeSchema(richSchema(), memory.DefaultAllocator)
	if err := CheckFramed(framed); err != nil {
		t.Fatal(err)
	}
	if err := CheckFramed(framed[4:]); err != nil {
		t.Errorf("without the continuation marker: %v", err)
	}
	for _, b := range [][]byte{{1, 2}, {0xFF, 0xFF, 0xFF, 0xFF, 1}, {200, 0, 0, 0, 1, 2, 3, 4}, {2, 0, 0, 0, 1, 2}} {
		if err := CheckFramed(b); !errors.Is(err, ErrInvalid) {
			t.Errorf("%v: %v", b, err)
		}
	}
}
