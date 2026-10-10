package dataframe

import (
	"fmt"
	"math"
	"time"

	"github.com/go-ndarray/ndarray"
)

// fromAny builds a column from Go data. It accepts slices of every Go number
// type, []bool, []string, []time.Time, []any (nil is missing; the dtype is
// inferred), a one-dimensional *ndarray.Array of any dtype (shared, not
// copied, when it is contiguous) and a *Series (its column).
func fromAny(data any) (column, error) {
	switch d := data.(type) {
	case []float64:
		return newCol(Float64, d, nil), nil
	case []float32:
		return newCol(Float32, d, nil), nil
	case []int:
		c := makeCol[int64](Int64, len(d))
		for i, v := range d {
			c.data[i] = int64(v)
		}
		return c, nil
	case []int8:
		return newCol(Int8, d, nil), nil
	case []int16:
		return newCol(Int16, d, nil), nil
	case []int32:
		return newCol(Int32, d, nil), nil
	case []int64:
		return newCol(Int64, d, nil), nil
	case []uint8:
		return newCol(Uint8, d, nil), nil
	case []uint16:
		return newCol(Uint16, d, nil), nil
	case []uint32:
		return newCol(Uint32, d, nil), nil
	case []uint64:
		return newCol(Uint64, d, nil), nil
	case []complex64:
		return newCol(Complex64, d, nil), nil
	case []complex128:
		return newCol(Complex128, d, nil), nil
	case []bool:
		return newCol(Bool, d, nil), nil
	case []string:
		return newCol(String, d, nil), nil
	case []time.Time:
		c := makeCol[int64](Datetime, len(d))
		for i, t := range d {
			c.data[i] = t.UnixNano()
		}
		return c, nil
	case *ndarray.Array:
		return fromArray(d)
	case *Series:
		return d.c, nil
	case []any:
		return fromValues(d)
	}
	return nil, fmt.Errorf("%w: cannot make a column from %T", ErrArgument, data)
}

// fromArray wraps a one-dimensional array as a column, sharing its storage
// when it is contiguous.
func fromArray(a *ndarray.Array) (column, error) {
	if a.Ndim() != 1 {
		return nil, fmt.Errorf("%w: a column needs a 1-D array, got shape %v", ErrArgument, a.Shape())
	}
	switch a.DType() {
	case ndarray.Float64:
		return wrapArray[float64](a), nil
	case ndarray.Bool:
		return wrapArray[bool](a), nil
	case ndarray.Int8:
		return wrapArray[int8](a), nil
	case ndarray.Int16:
		return wrapArray[int16](a), nil
	case ndarray.Int32:
		return wrapArray[int32](a), nil
	case ndarray.Int64:
		return wrapArray[int64](a), nil
	case ndarray.Uint8:
		return wrapArray[uint8](a), nil
	case ndarray.Uint16:
		return wrapArray[uint16](a), nil
	case ndarray.Uint32:
		return wrapArray[uint32](a), nil
	case ndarray.Uint64:
		return wrapArray[uint64](a), nil
	case ndarray.Float32:
		return wrapArray[float32](a), nil
	case ndarray.Complex64:
		return wrapArray[complex64](a), nil
	}
	return wrapArray[complex128](a), nil
}

func wrapArray[T comparable](a *ndarray.Array) *col[T] {
	dt := DType(a.DType())
	d := arrayData[T](a)
	c := &col[T]{dt: dt, data: d, o: opsOf[T](dt)}
	// Data copies a strided view into fresh storage; only an array whose
	// storage d is can be handed back out by the column.
	if sameStorage(a, d) {
		c.arr = a
	}
	return c
}

// sameStorage reports whether d is a's own storage (Data did not copy).
func sameStorage[T comparable](a *ndarray.Array, d []T) bool {
	if len(d) == 0 {
		return true
	}
	again := arrayData[T](a)
	return &again[0] == &d[0]
}

// fromValues infers a column from boxed values: nil is missing; the dtype is
// bool, int64, float64, complex128, string or datetime, the widest one the
// values need. Mixing strings with other kinds is an error.
func fromValues(vs []any) (column, error) {
	kind := -1 // 0 bool, 1 int, 2 float, 3 complex, 4 string, 5 time
	for _, v := range vs {
		k := -1
		switch v.(type) {
		case nil:
			continue
		case bool:
			k = 0
		case int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64:
			k = 1
		case float32, float64:
			k = 2
		case complex64, complex128:
			k = 3
		case string:
			k = 4
		case time.Time:
			k = 5
		default:
			return nil, fmt.Errorf("%w: cannot hold %T in a column", ErrArgument, v)
		}
		switch {
		case kind < 0:
			kind = k
		case kind == k:
		case kind <= 3 && k <= 3 && (kind > 0 && k > 0):
			kind = max(kind, k)
		default:
			return nil, fmt.Errorf("%w: mixed %s and %T values", ErrArgument, kindName[kind], v)
		}
	}
	n := len(vs)
	var c column
	switch kind {
	case -1:
		c = makeCol[float64](Float64, n)
	case 0:
		c = makeCol[bool](Bool, n)
	case 1:
		c = makeCol[int64](Int64, n)
	case 2:
		c = makeCol[float64](Float64, n)
	case 3:
		c = makeCol[complex128](Complex128, n)
	case 4:
		c = makeCol[string](String, n)
	default:
		c = makeCol[int64](Datetime, n)
	}
	valid := allValid(n)
	missing := false
	for i, v := range vs {
		if v == nil {
			valid.clear(i)
			missing = true
			continue
		}
		setBoxed(c, i, v)
	}
	if missing {
		c = c.withValidity(valid)
	}
	return c, nil
}

var kindName = []string{"bool", "integer", "float", "complex", "string", "time"}

// setBoxed stores a boxed value, already known to fit, at row i.
func setBoxed(c column, i int, v any) {
	switch c := c.(type) {
	case *col[bool]:
		c.data[i] = v.(bool)
	case *col[int64]:
		c.data[i], _ = c.o.unbox(v)
	case *col[float64]:
		c.data[i], _ = c.o.unbox(v)
	case *col[complex128]:
		c.data[i], _ = c.o.unbox(v)
	case *col[string]:
		c.data[i] = v.(string)
	}
}

// catCol builds a Category column from codes (-1 missing) into levels.
func catCol(codes []int32, levels []string) *col[int32] {
	c := makeCol[int32](Category, len(codes))
	c.levels = levels
	c.o = catOps(levels)
	var valid bitmap
	for i, k := range codes {
		if k < 0 {
			if valid == nil {
				valid = allValid(len(codes))
			}
			valid.clear(i)
			k = 0
		}
		c.data[i] = k
	}
	c.valid = valid
	return c
}

// fullCol returns a column of n copies of a scalar Go value.
func fullCol(v any, n int) (column, error) {
	one, err := fromValues([]any{v})
	if err != nil {
		return nil, err
	}
	idx := make([]int, n)
	return one.take(idx), nil
}

// floatCol returns a float64 column, NaN marking missing values.
func floatCol(d []float64) *col[float64] {
	return &col[float64]{dt: Float64, data: d, o: opsFloat64}
}

// floatsOf returns a real or bool column as float64s, missing as NaN.
func floatsOf(c column) ([]float64, bool) {
	out := make([]float64, c.len())
	switch c := c.(type) {
	case *col[float64]:
		copy(out, c.data)
	case *col[int64]:
		if c.dt != Int64 {
			return nil, false
		}
		for i, v := range c.data {
			out[i] = float64(v)
		}
	default:
		if !c.dtype().IsNumeric() || c.dtype() == Complex64 || c.dtype() == Complex128 {
			return nil, false
		}
		a, _ := c.ndarray()
		f, _ := ndarray.Data[float64](a.AsType(ndarray.Float64))
		copy(out, f)
	}
	v := c.validity()
	if v != nil {
		for i := range out {
			if !v.get(i) {
				out[i] = math.NaN()
			}
		}
	}
	return out, true
}
