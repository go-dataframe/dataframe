package dataframe

import (
	"cmp"
	"math"
	"strconv"
	"strings"
	"time"
)

type realT interface {
	int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | float32 | float64
}

// The ops tables of the dtypes that need no per-column state.
var (
	opsFloat64    = floatOps[float64](64)
	opsFloat32    = floatOps[float32](32)
	opsInt8       = intOps[int8]()
	opsInt16      = intOps[int16]()
	opsInt32      = intOps[int32]()
	opsInt64      = intOps[int64]()
	opsUint8      = uintOps[uint8]()
	opsUint16     = uintOps[uint16]()
	opsUint32     = uintOps[uint32]()
	opsUint64     = uintOps[uint64]()
	opsComplex64  = complexOps[complex64](64)
	opsComplex128 = complexOps[complex128](128)
	opsBool       = &ops[bool]{
		cmp: func(a, b bool) int {
			switch {
			case a == b:
				return 0
			case b:
				return -1
			}
			return 1
		},
		str: func(b bool) string {
			if b {
				return "True"
			}
			return "False"
		},
		box: func(b bool) any { return b },
		unbox: func(v any) (bool, bool) {
			b, ok := v.(bool)
			return b, ok
		},
	}
	opsDatetime = &ops[int64]{
		cmp:   cmp.Compare[int64],
		str:   formatTime,
		box:   func(ns int64) any { return time.Unix(0, ns).UTC() },
		unbox: toNanos,
	}
)

func floatOps[T float32 | float64](bits int) *ops[T] {
	return &ops[T]{
		cmp:   cmp.Compare[T],
		nan:   func(x T) bool { return x != x },
		str:   func(x T) string { return reprFloat(float64(x), bits) },
		box:   func(x T) any { return x },
		unbox: func(v any) (T, bool) { f, ok := toFloat(v); return T(f), ok },
	}
}

func intOps[T int8 | int16 | int32 | int64]() *ops[T] {
	return &ops[T]{
		cmp: cmp.Compare[T],
		str: func(x T) string { return strconv.FormatInt(int64(x), 10) },
		box: func(x T) any { return x },
		unbox: func(v any) (T, bool) {
			i, ok := toInt(v)
			return T(i), ok && int64(T(i)) == i
		},
	}
}

func uintOps[T uint8 | uint16 | uint32 | uint64]() *ops[T] {
	return &ops[T]{
		cmp: cmp.Compare[T],
		str: func(x T) string { return strconv.FormatUint(uint64(x), 10) },
		box: func(x T) any { return x },
		unbox: func(v any) (T, bool) {
			if u, ok := v.(uint64); ok {
				return T(u), uint64(T(u)) == u
			}
			i, ok := toInt(v)
			return T(i), ok && i >= 0 && int64(T(i)) == i
		},
	}
}

func complexOps[T complex64 | complex128](bits int) *ops[T] {
	return &ops[T]{
		// Complex numbers order lexicographically, real part first, as
		// NumPy sorts them.
		cmp: func(a, b T) int {
			x, y := complex128(a), complex128(b)
			if c := cmp.Compare(real(x), real(y)); c != 0 {
				return c
			}
			return cmp.Compare(imag(x), imag(y))
		},
		nan: func(v T) bool { x := complex128(v); return real(x) != real(x) || imag(x) != imag(x) },
		str: func(x T) string { return strconv.FormatComplex(complex128(x), 'g', -1, bits) },
		box: func(x T) any { return x },
		unbox: func(v any) (T, bool) {
			switch z := v.(type) {
			case complex128:
				return T(z), true
			case complex64:
				return T(z), true
			}
			f, ok := toFloat(v)
			return T(complex(f, 0)), ok
		},
	}
}

// catOps returns the ops of a Category column over levels. Categories order
// by their position in levels, as pandas orders an ordered categorical.
func catOps(levels []string) *ops[int32] {
	pos := make(map[string]int32, len(levels))
	for i, l := range levels {
		pos[l] = int32(i)
	}
	return &ops[int32]{
		cmp: cmp.Compare[int32],
		str: func(c int32) string { return levels[c] },
		box: func(c int32) any { return levels[c] },
		unbox: func(v any) (int32, bool) {
			s, ok := v.(string)
			c, found := pos[s]
			return c, ok && found
		},
	}
}

// toFloat converts a Go number to float64.
func toFloat(v any) (float64, bool) {
	switch x := v.(type) {
	case float64:
		return x, true
	case float32:
		return float64(x), true
	case uint64:
		return float64(x), true
	}
	i, ok := toInt(v)
	return float64(i), ok
}

// toInt converts a Go integer, or an integral float, to int64.
func toInt(v any) (int64, bool) {
	switch x := v.(type) {
	case int:
		return int64(x), true
	case int8:
		return int64(x), true
	case int16:
		return int64(x), true
	case int32:
		return int64(x), true
	case int64:
		return x, true
	case uint:
		return int64(x), uint64(x) <= math.MaxInt64
	case uint8:
		return int64(x), true
	case uint16:
		return int64(x), true
	case uint32:
		return int64(x), true
	case uint64:
		return int64(x), uint64(x) <= math.MaxInt64
	case float64:
		return int64(x), x == math.Trunc(x) && math.Abs(x) < 1<<63
	case float32:
		return int64(x), float64(x) == math.Trunc(float64(x)) && math.Abs(float64(x)) < 1<<63
	}
	return 0, false
}

// toNanos converts a time.Time, or a date string, to nanoseconds since the
// epoch.
func toNanos(v any) (int64, bool) {
	switch x := v.(type) {
	case time.Time:
		return x.UnixNano(), true
	case string:
		ns, err := parseTime(x, "")
		return ns, err == nil
	}
	return 0, false
}

// formatTime formats nanoseconds since the epoch as pandas prints a
// Timestamp: date, time, and as many fractional digits as are needed.
func formatTime(ns int64) string {
	t := time.Unix(0, ns).UTC()
	switch {
	case t.Nanosecond() == 0:
		return t.Format("2006-01-02 15:04:05")
	case t.Nanosecond()%1000 == 0:
		return t.Format("2006-01-02 15:04:05.000000")
	}
	return t.Format("2006-01-02 15:04:05.000000000")
}

// reprFloat formats a float the way Python's repr does: the shortest digits
// that read back, in fixed notation (with at least one decimal) for
// exponents from -4 to 15 and in scientific notation otherwise.
func reprFloat(f float64, bits int) string {
	switch {
	case math.IsNaN(f):
		return "nan"
	case math.IsInf(f, 1):
		return "inf"
	case math.IsInf(f, -1):
		return "-inf"
	}
	e := strconv.FormatFloat(f, 'e', -1, bits)
	exp, _ := strconv.Atoi(e[strings.IndexByte(e, 'e')+1:])
	if exp < -4 || exp >= 16 {
		return e
	}
	s := strconv.FormatFloat(f, 'f', -1, bits)
	if !strings.ContainsAny(s, ".") {
		s += ".0"
	}
	return s
}
