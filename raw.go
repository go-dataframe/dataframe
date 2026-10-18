package dataframe

import (
	"github.com/go-dataframe/dataframe/internal/rawcol"
)

// The storage hooks of internal/rawcol, which the I/O subpackages use to
// read into and write from a column's storage without copying it.
func init() {
	rawcol.Series = rawSeries
	rawcol.Categorical = rawCategorical
	rawcol.Strings = rawStrings
	rawcol.StringSeries = rawStringSeries
	rawcol.Take = func(s any, idx []int) any {
		p := s.(*Series)
		return &Series{name: p.name, index: NewRangeIndex(len(idx)), c: p.c.take(idx)}
	}
	rawcol.Codes = rawCodes
}

// rawSeries is NewSeriesBitmap without the copies, for numeric storage
// (strings go through rawStringSeries): the column shares data
// and valid. A bitmap with nothing cleared is dropped, as there.
func rawSeries(name string, data any, valid []uint64, datetime bool) any {
	var c column
	switch d := data.(type) {
	case []float64:
		c = rawCol(Float64, d)
	case []float32:
		c = rawCol(Float32, d)
	case []int8:
		c = rawCol(Int8, d)
	case []int16:
		c = rawCol(Int16, d)
	case []int32:
		c = rawCol(Int32, d)
	case []int64:
		if datetime {
			c = rawCol(Datetime, d)
		} else {
			c = rawCol(Int64, d)
		}
	case []uint8:
		c = rawCol(Uint8, d)
	case []uint16:
		c = rawCol(Uint16, d)
	case []uint32:
		c = rawCol(Uint32, d)
	default:
		c = rawCol(Uint64, data.([]uint64))
	}
	n := c.len()
	if valid != nil && bitmap(valid).count(n) < n {
		c = c.withValidity(bitmap(valid[:(n+63)>>6]))
	}
	return &Series{name: name, index: NewRangeIndex(n), c: c}
}

// rawCol wraps data as a column. It has no ndarray yet: one is made, with a
// copy, the first time the column is asked for it.
func rawCol[T comparable](dt DType, data []T) *col[T] {
	return &col[T]{dt: dt, data: data, o: opsOf[T](dt)}
}

// rawCategorical is NewCategorical sharing codes, which are known to be in
// range, with 0 in the missing slots, which valid marks (nil: none).
func rawCategorical(name string, codes []int32, levels []string, valid []uint64) any {
	c := rawCol(Category, codes)
	c.levels = levels
	c.o = catOps(levels)
	c.valid = valid
	return &Series{name: name, index: NewRangeIndex(len(codes)), c: c}
}

func rawStrings(s any) (string, any, []uint64, bool) {
	c, ok := s.(*Series).c.(stringCol)
	if !ok {
		return "", nil, nil, false
	}
	data, offs := c.raw()
	return data, offs, c.validity(), true
}

// rawStringSeries is a string series over data and offs ([]int32 or
// []int64, value i data[offs[i]:offs[i+1]]), sharing both, and valid,
// dropped when nothing is cleared.
func rawStringSeries(name, data string, offs any, valid []uint64) any {
	var c column
	if o, ok := offs.([]int32); ok {
		c = &strCol[int32]{data: data, offs: o}
	} else {
		c = &strCol[int64]{data: data, offs: offs.([]int64)}
	}
	n := c.len()
	if valid != nil && bitmap(valid).count(n) < n {
		c = c.withValidity(bitmap(valid[:(n+63)>>6]))
	}
	return &Series{name: name, index: NewRangeIndex(n), c: c}
}

func rawCodes(s any) ([]int32, []uint64, []string) {
	c := s.(*Series).c.(*col[int32])
	return c.data, c.valid, c.levels
}
