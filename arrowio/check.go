package arrowio

import (
	"fmt"
	"math"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/memory"
)

// The Arrow readers check little of what a record declares beyond what
// building its arrays needs (arrow-go's array constructors panic on a
// values buffer too short for a numeric array, or on string offsets past
// their bytes, and the readers turn that panic into ErrCorrupt, see
// decoded); the conversion indexes and reinterprets the rest, so it checks
// it first. A file read from anywhere may declare anything.

// maxNullRows bounds the rows of a frame whose every column is of the null
// type: no buffer backs such a column, so nothing else bounds what a few
// bytes of metadata can declare, and the frame allocates per row.
const maxNullRows = 1 << 26

// checkColumns checks every chunk of every column (see checkArray), and
// that each column holds n values.
func checkColumns(fields []arrow.Field, chunks [][]arrow.Array, n int) error {
	allNull := len(fields) > 0
	for i, f := range fields {
		total := 0
		for _, c := range chunks[i] {
			err := checkArray(c)
			if total += c.Len(); err == nil && total < 0 {
				err = fmt.Errorf("%w: lengths overflow", ErrCorrupt)
			}
			if err != nil {
				return fmt.Errorf("column %q: %w", f.Name, err)
			}
		}
		if total != n {
			return fmt.Errorf("%w: column %q holds %d values for %d rows", ErrCorrupt, f.Name, total, n)
		}
		allNull = allNull && f.Type.ID() == arrow.NULL
	}
	if allNull && n > maxNullRows {
		return fmt.Errorf("%w: %d rows of nothing but nulls", ErrCorrupt, n)
	}
	return nil
}

// checkArray checks what the conversion relies on and arrow-go does not:
// a length and offset that are not negative, a validity bitmap as long as
// the values (and one whenever there are nulls), a values buffer whenever
// there are values, a boolean bitmap as long as the values, string offsets
// not negative and in order, and a dictionary whose values pass too and
// whose every present index is in range. Other types are refused later,
// but their validity bitmap is read first, so it is checked too.
func checkArray(a arrow.Array) error {
	d := a.Data()
	off, n := d.Offset(), d.Len()
	if off < 0 || n < 0 || off > math.MaxInt-n-1 {
		return fmt.Errorf("%w: offset %d, length %d", ErrCorrupt, off, n)
	}
	if a.DataType().ID() == arrow.NULL || n == 0 {
		return nil
	}
	end := off + n
	bufs := make([]*memory.Buffer, 2)
	copy(bufs, d.Buffers())
	switch v := bufs[0]; {
	case v != nil && v.Len() < (end+7)/8:
		return fmt.Errorf("%w: validity bitmap of %d bytes for %d values", ErrCorrupt, v.Len(), end)
	case v == nil && a.NullN() > 0:
		return fmt.Errorf("%w: %d nulls and no validity bitmap", ErrCorrupt, a.NullN())
	case bufs[1] == nil:
		return fmt.Errorf("%w: %d values and no values buffer", ErrCorrupt, n)
	}
	switch c := a.(type) {
	case *array.Boolean:
		if bufs[1].Len() < (end+7)/8 {
			return fmt.Errorf("%w: boolean values too short for %d values", ErrCorrupt, end)
		}
	case *array.String:
		return ordered(c.ValueOffsets())
	case *array.LargeString:
		return ordered(c.ValueOffsets())
	case *array.Dictionary:
		if err := checkArray(c.Dictionary()); err != nil {
			return fmt.Errorf("dictionary: %w", err)
		}
		return checkIndices(c)
	}
	return nil
}

// ordered checks string offsets: none negative, in order. arrow-go has
// checked that the last is within the bytes.
func ordered[O int32 | int64](o []O) error {
	prev := O(0)
	for i, x := range o {
		if x < prev {
			return fmt.Errorf("%w: string offset %d at %d after %d", ErrCorrupt, x, i, prev)
		}
		prev = x
	}
	return nil
}

// checkIndices checks that every present slot of a dictionary column
// indexes its dictionary.
func checkIndices(d *array.Dictionary) error {
	m := d.Dictionary().Len()
	switch idx := d.Indices().(type) {
	case *array.Int8:
		return inRange(d, idx.Int8Values(), m)
	case *array.Int16:
		return inRange(d, idx.Int16Values(), m)
	case *array.Int32:
		return inRange(d, idx.Int32Values(), m)
	case *array.Int64:
		return inRange(d, idx.Int64Values(), m)
	case *array.Uint8:
		return inRange(d, idx.Uint8Values(), m)
	case *array.Uint16:
		return inRange(d, idx.Uint16Values(), m)
	case *array.Uint32:
		return inRange(d, idx.Uint32Values(), m)
	}
	return inRange(d, d.Indices().(*array.Uint64).Uint64Values(), m)
}

func inRange[T int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64](d *array.Dictionary, idx []T, m int) error {
	for i, k := range idx {
		if (k < 0 || uint64(k) >= uint64(m)) && d.IsValid(i) {
			return fmt.Errorf("%w: dictionary index %d at %d for %d values", ErrCorrupt, k, i, m)
		}
	}
	return nil
}
