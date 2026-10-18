// Package arrowfb checks the flatbuffers of Arrow IPC metadata (a Message,
// or a file's Footer) before arrow-go reads them. arrow-go trusts what a
// flatbuffer declares: it allocates a slice as long as a vector says it is
// (make([]arrow.Field, n) with n read from the input, for instance), so a
// few bytes can make it allocate gigabytes. Check walks every table,
// vector and string the Arrow schema defines, and fails unless each lies
// within the buffer: a vector of n elements needs n times its element
// size of bytes after it, so nothing the buffer does not hold can be
// declared.
package arrowfb

import (
	"encoding/binary"
	"errors"
	"fmt"
)

// ErrInvalid reports a flatbuffer that does not lie within its bytes.
var ErrInvalid = errors.New("arrowfb: invalid flatbuffer")

// maxDepth bounds the nesting of fields (children of children ...);
// maxTables bounds the tables visited, as flatbuffers may share subtables
// and so make a walk exponential.
const (
	maxDepth  = 64
	maxTables = 1 << 20
)

type checker struct {
	b      []byte
	tables int
}

// CheckMessage checks a Message flatbuffer (the bytes of an IPC message's
// metadata, without the length prefix) and returns its bodyLength.
func CheckMessage(b []byte) (bodyLength int64, err error) {
	c := &checker{b: b}
	root, err := c.root()
	if err != nil {
		return 0, err
	}
	// Message: version, header_type, header, bodyLength, custom_metadata.
	switch c.u8(root, 1) {
	case 1: // Schema
		err = c.table(root, 2, c.schema)
	case 2: // DictionaryBatch: id, data, isDelta
		err = c.table(root, 2, func(t uint32) error { return c.table(t, 1, c.recordBatch) })
	case 3: // RecordBatch
		err = c.table(root, 2, c.recordBatch)
	}
	if err == nil {
		err = c.tables_(root, 4, c.keyValue)
	}
	if err != nil {
		return 0, err
	}
	return c.i64(root, 3), nil
}

// CheckFooter checks a file's Footer flatbuffer.
func CheckFooter(b []byte) error {
	c := &checker{b: b}
	root, err := c.root()
	if err != nil {
		return err
	}
	// Footer: version, schema, dictionaries [Block], recordBatches [Block],
	// custom_metadata. A Block is a 24-byte struct.
	if err := c.table(root, 1, c.schema); err != nil {
		return err
	}
	if _, err := c.vector(root, 2, 24); err != nil {
		return err
	}
	if _, err := c.vector(root, 3, 24); err != nil {
		return err
	}
	return c.tables_(root, 4, c.keyValue)
}

func (c *checker) fail(format string, a ...any) error {
	return fmt.Errorf("%w: "+format, append([]any{ErrInvalid}, a...)...)
}

func (c *checker) u32at(p uint64) (uint32, bool) {
	if p+4 > uint64(len(c.b)) {
		return 0, false
	}
	return binary.LittleEndian.Uint32(c.b[p:]), true
}

// root returns the position of the root table.
func (c *checker) root() (uint32, error) {
	off, ok := c.u32at(0)
	if !ok {
		return 0, c.fail("%d bytes, no root", len(c.b))
	}
	if err := c.checkTable(off); err != nil {
		return 0, err
	}
	return off, nil
}

// checkTable checks the table at t: its vtable offset, the vtable, and that
// every field the vtable lists lies within the table and the buffer.
func (c *checker) checkTable(t uint32) error {
	if c.tables++; c.tables > maxTables {
		return c.fail("more than %d tables", maxTables)
	}
	so, ok := c.u32at(uint64(t))
	if !ok {
		return c.fail("table at %d past the end", t)
	}
	vt := int64(t) - int64(int32(so))
	if vt < 0 || vt+4 > int64(len(c.b)) {
		return c.fail("vtable of the table at %d at %d", t, vt)
	}
	vsize := int64(binary.LittleEndian.Uint16(c.b[vt:]))
	tsize := int64(binary.LittleEndian.Uint16(c.b[vt+2:]))
	if vsize < 4 || vsize%2 != 0 || vt+vsize > int64(len(c.b)) || int64(t)+tsize > int64(len(c.b)) {
		return c.fail("vtable of the table at %d", t)
	}
	return nil
}

// field returns the position of field slot of the table at t, or 0 when
// absent.
func (c *checker) field(t uint32, slot int) uint64 {
	so := binary.LittleEndian.Uint32(c.b[t:])
	vt := int64(t) - int64(int32(so))
	vsize := int64(binary.LittleEndian.Uint16(c.b[vt:]))
	e := 4 + 2*int64(slot)
	if e+2 > vsize {
		return 0
	}
	o := binary.LittleEndian.Uint16(c.b[vt+e:])
	if o == 0 {
		return 0
	}
	return uint64(t) + uint64(o)
}

func (c *checker) u8(t uint32, slot int) byte {
	p := c.field(t, slot)
	if p == 0 || p >= uint64(len(c.b)) {
		return 0
	}
	return c.b[p]
}

func (c *checker) i64(t uint32, slot int) int64 {
	p := c.field(t, slot)
	if p == 0 || p+8 > uint64(len(c.b)) {
		return 0
	}
	return int64(binary.LittleEndian.Uint64(c.b[p:]))
}

// indirect follows the offset stored at field slot of the table at t; ok
// is false when the field is absent.
func (c *checker) indirect(t uint32, slot int) (uint32, bool, error) {
	p := c.field(t, slot)
	if p == 0 {
		return 0, false, nil
	}
	o, ok := c.u32at(p)
	if !ok || p+uint64(o) > uint64(len(c.b)) || p+uint64(o) > 1<<32-1 {
		return 0, false, c.fail("offset at %d out of bounds", p)
	}
	return uint32(p + uint64(o)), true, nil
}

// table checks the subtable at field slot, if present, with f.
func (c *checker) table(t uint32, slot int, f func(uint32) error) error {
	sub, ok, err := c.indirect(t, slot)
	if err != nil || !ok {
		return err
	}
	if err := c.checkTable(sub); err != nil {
		return err
	}
	return f(sub)
}

// vector checks the vector at field slot, of elements of size bytes each,
// and returns its position and length (0 when absent).
func (c *checker) vector(t uint32, slot int, size uint64) (uint32, error) {
	v, ok, err := c.indirect(t, slot)
	if err != nil || !ok {
		return 0, err
	}
	n, ok := c.u32at(uint64(v))
	if !ok || uint64(v)+4+uint64(n)*size > uint64(len(c.b)) {
		return 0, c.fail("vector at %d of %d elements past the end", v, n)
	}
	return v, nil
}

// tables_ checks a vector of tables at field slot, each with f.
func (c *checker) tables_(t uint32, slot int, f func(uint32) error) error {
	v, err := c.vector(t, slot, 4)
	if err != nil || v == 0 {
		return err
	}
	n, _ := c.u32at(uint64(v))
	for i := range uint64(n) {
		e := uint64(v) + 4 + 4*i
		o, _ := c.u32at(e)
		if e+uint64(o) > uint64(len(c.b)) || e+uint64(o) > 1<<32-1 {
			return c.fail("element %d of the vector at %d out of bounds", i, v)
		}
		sub := uint32(e + uint64(o))
		if err := c.checkTable(sub); err != nil {
			return err
		}
		if err := f(sub); err != nil {
			return err
		}
	}
	return nil
}

// str checks the string at field slot: its bytes and their terminator.
func (c *checker) str(t uint32, slot int) error {
	v, err := c.vector(t, slot, 1)
	if err != nil || v == 0 {
		return err
	}
	n, _ := c.u32at(uint64(v))
	if uint64(v)+4+uint64(n)+1 > uint64(len(c.b)) {
		return c.fail("string at %d past the end", v)
	}
	return nil
}

func (c *checker) keyValue(t uint32) error {
	if err := c.str(t, 0); err != nil {
		return err
	}
	return c.str(t, 1)
}

// schema: endianness, fields, custom_metadata, features [long].
func (c *checker) schema(t uint32) error {
	if err := c.tables_(t, 1, func(f uint32) error { return c.fieldTable(f, 0) }); err != nil {
		return err
	}
	if err := c.tables_(t, 2, c.keyValue); err != nil {
		return err
	}
	_, err := c.vector(t, 3, 8)
	return err
}

// fieldTable: name, nullable, type_type, type, dictionary, children,
// custom_metadata.
func (c *checker) fieldTable(t uint32, depth int) error {
	if depth > maxDepth {
		return c.fail("fields nested deeper than %d", maxDepth)
	}
	if err := c.str(t, 0); err != nil {
		return err
	}
	typ := c.u8(t, 2)
	if err := c.table(t, 3, func(u uint32) error {
		switch typ {
		case 10: // Timestamp: unit, timezone
			return c.str(u, 1)
		case 14: // Union: mode, typeIds [int]
			_, err := c.vector(u, 1, 4)
			return err
		}
		return nil
	}); err != nil {
		return err
	}
	// DictionaryEncoding: id, indexType (Int: scalars), isOrdered, kind.
	if err := c.table(t, 4, func(d uint32) error { return c.table(d, 1, func(uint32) error { return nil }) }); err != nil {
		return err
	}
	if err := c.tables_(t, 5, func(f uint32) error { return c.fieldTable(f, depth+1) }); err != nil {
		return err
	}
	return c.tables_(t, 6, c.keyValue)
}

// recordBatch: length, nodes [FieldNode 16 bytes], buffers [Buffer 16
// bytes], compression (scalars), variadicBufferCounts [long].
func (c *checker) recordBatch(t uint32) error {
	for _, v := range []struct {
		slot int
		size uint64
	}{{1, 16}, {2, 16}, {4, 8}} {
		if _, err := c.vector(t, v.slot, v.size); err != nil {
			return err
		}
	}
	return c.table(t, 3, func(uint32) error { return nil })
}

// Block is a message's place in an IPC file, as the footer lists it.
type Block struct {
	Offset         int64
	MetaDataLength int32
	BodyLength     int64
}

// FooterBlocks returns the dictionary and record batch blocks of a footer
// that CheckFooter accepted.
func FooterBlocks(b []byte) []Block {
	c := &checker{b: b}
	root, _ := c.u32at(0)
	var out []Block
	for _, slot := range []int{2, 3} {
		v, err := c.vector(root, slot, 24)
		if err != nil || v == 0 {
			continue
		}
		n, _ := c.u32at(uint64(v))
		for i := range uint64(n) {
			p := uint64(v) + 4 + 24*i
			out = append(out, Block{
				Offset:         int64(binary.LittleEndian.Uint64(b[p:])),
				MetaDataLength: int32(binary.LittleEndian.Uint32(b[p+8:])),
				BodyLength:     int64(binary.LittleEndian.Uint64(b[p+16:])),
			})
		}
	}
	return out
}

// CheckFramed checks a serialized IPC message, framing included (the
// continuation marker, if any, and the length), as an Arrow schema is
// stored in a Parquet file's ARROW:schema metadata.
func CheckFramed(b []byte) error {
	if len(b) >= 4 && binary.LittleEndian.Uint32(b) == 0xFFFFFFFF {
		b = b[4:]
	}
	if len(b) < 4 {
		return fmt.Errorf("%w: %d bytes of framing", ErrInvalid, len(b))
	}
	n := int64(int32(binary.LittleEndian.Uint32(b)))
	if n < 4 || n > int64(len(b)-4) {
		return fmt.Errorf("%w: message of %d bytes in %d", ErrInvalid, n, len(b)-4)
	}
	_, err := CheckMessage(b[4 : 4+n])
	return err
}
