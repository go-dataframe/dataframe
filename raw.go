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
	rawcol.Codes = rawCodes
}

// rawSeries is NewSeriesBitmap without the copies: the column shares data
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
	case []uint64:
		c = rawCol(Uint64, d)
	default:
		c = rawCol(String, data.([]string))
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

func rawStrings(s any) ([]string, []uint64, bool) {
	c, ok := s.(*Series).c.(*col[string])
	if !ok {
		return nil, nil, false
	}
	return c.data, c.valid, true
}

func rawCodes(s any) ([]int32, []uint64, []string) {
	c := s.(*Series).c.(*col[int32])
	return c.data, c.valid, c.levels
}
