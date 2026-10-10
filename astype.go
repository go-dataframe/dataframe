package dataframe

import (
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"

	"github.com/go-ndarray/ndarray"
)

// resultType is NumPy's promotion of two numeric dtypes.
func resultType(a, b DType) DType {
	return DType(ndarray.ResultType(ndarray.DType(a), ndarray.DType(b)))
}

// cast converts a column to another dtype, keeping missing values missing.
// Numbers convert as NumPy's astype converts them (a float NaN going to an
// integer type becomes a missing value rather than garbage); anything
// formats to string; strings parse to numbers, bools and instants, and an
// unparsable string is an error.
func cast(c column, dt DType) (column, error) {
	src := c.dtype()
	switch {
	case src == dt && dt != Category:
		return c, nil
	case dt == String:
		out := makeCol[string](String, c.len())
		for i := range out.data {
			if !c.isNA(i) {
				out.data[i] = c.str(i)
			}
		}
		out.valid = naBitmap(c)
		return out, nil
	case dt == Category:
		return toCategory(c)
	case src.IsNumeric() && dt.IsNumeric():
		valid := naBitmap(c)
		if src.isFloat() && !dt.isFloat() {
			// NaN has no integer form: keep it missing.
			c = mustFill(c, 0)
		}
		a, _ := c.ndarray()
		out, _ := fromArray(a.AsType(ndarray.DType(dt)))
		return out.withValidity(valid), nil
	case src == Datetime && dt == Int64:
		p := c.(*col[int64])
		return &col[int64]{dt: Int64, data: p.data, arr: p.arr, valid: p.valid, o: opsInt64}, nil
	case src == Int64 && dt == Datetime:
		p := c.(*col[int64])
		return &col[int64]{dt: Datetime, data: p.data, arr: p.arr, valid: p.valid, o: opsDatetime}, nil
	case src == String && (dt.IsNumeric() || dt == Datetime):
		return parseStrings(c.(*col[string]), dt)
	case src == Category:
		s, _ := cast(c, String)
		return cast(s, dt)
	}
	return nil, fmt.Errorf("%w: cannot convert %v to %v", ErrDType, src, dt)
}

// mustFill fills missing values with a value known to fit.
func mustFill(c column, v any) column {
	out, _ := c.fill(v)
	return out
}

// naBitmap returns the column's missing values as a bitmap (NaN included),
// nil when there are none.
func naBitmap(c column) bitmap {
	if c.naCount() == 0 {
		return nil
	}
	b := allValid(c.len())
	for i := range c.len() {
		if c.isNA(i) {
			b.clear(i)
		}
	}
	return b
}

// parseStrings parses a string column into a numeric or datetime column.
func parseStrings(c *col[string], dt DType) (column, error) {
	parsed := make([]any, len(c.data))
	for i, s := range c.data {
		if c.isNA(i) {
			continue
		}
		v, err := parseScalar(strings.TrimSpace(s), dt)
		if err != nil {
			return nil, err
		}
		parsed[i] = v
	}
	if dt == Uint64 {
		out := makeCol[uint64](Uint64, len(parsed))
		out.valid = c.valid
		for i, v := range parsed {
			if v != nil {
				out.data[i] = v.(uint64)
			}
		}
		return out, nil
	}
	f, _ := fromValues(parsed)
	return cast(f, dt)
}

// parseScalar parses one string as a value of a numeric or datetime dtype.
func parseScalar(s string, dt DType) (any, error) {
	switch {
	case dt == Datetime:
		ns, err := parseTime(s, "")
		if err != nil {
			return nil, err
		}
		return timeOf(ns), nil
	case dt == Bool:
		switch strings.ToLower(s) {
		case "true":
			return true, nil
		case "false":
			return false, nil
		}
	case dt == Uint64:
		if u, err := strconv.ParseUint(s, 10, 64); err == nil {
			return u, nil
		}
	case dt.isInt():
		if i, err := strconv.ParseInt(s, 10, 64); err == nil {
			return i, nil
		}
	case dt == Complex64 || dt == Complex128:
		if z, err := strconv.ParseComplex(s, 128); err == nil {
			return z, nil
		}
	default:
		if f, err := strconv.ParseFloat(s, 64); err == nil {
			return f, nil
		}
	}
	return nil, fmt.Errorf("%w: %q is not a %v", ErrParse, s, dt)
}

// toCategory converts a column to a Category column whose categories are its
// distinct values, formatted, in sorted order (as pandas infers them).
func toCategory(c column) (column, error) {
	codes, uniq := c.factorize()
	order := make([]int, uniq.len())
	for i := range order {
		order[i] = i
	}
	slices.SortStableFunc(order, uniq.compare)
	levels := make([]string, len(order))
	rank := make([]int32, len(order))
	for r, u := range order {
		levels[r] = uniq.str(u)
		rank[u] = int32(r)
	}
	for i, k := range codes {
		if k >= 0 {
			codes[i] = rank[k]
		}
	}
	return catCol(codes, levels), nil
}

// AsType returns the series converted to another dtype.
func (s *Series) AsType(dt DType) (*Series, error) {
	c, err := cast(s.c, dt)
	if err != nil {
		return nil, err
	}
	return s.with(c), nil
}

// AsType returns the frame with the named columns converted, as
// Series.AsType converts.
func (df *DataFrame) AsType(dtypes map[string]DType) (*DataFrame, error) {
	cols := slices.Clone(df.cols)
	for name, dt := range dtypes {
		i, err := df.pos(name)
		if err != nil {
			return nil, err
		}
		if cols[i], err = cast(cols[i], dt); err != nil {
			return nil, fmt.Errorf("column %q: %w", name, err)
		}
	}
	return df.derive(cols), nil
}

// AsTypeAll returns the frame with every column converted to one dtype.
func (df *DataFrame) AsTypeAll(dt DType) (*DataFrame, error) {
	m := make(map[string]DType, len(df.names))
	for _, n := range df.names {
		m[n] = dt
	}
	return df.AsType(m)
}

// Round returns the series rounded to the given number of decimals (half to
// even, as NumPy rounds). Integer and non-numeric series are unchanged.
func (s *Series) Round(decimals int) *Series { return s.with(roundCol(s.c, decimals)) }

// Round rounds every float column, as Series.Round.
func (df *DataFrame) Round(decimals int) *DataFrame {
	cols := make([]column, len(df.cols))
	for i, c := range df.cols {
		cols[i] = roundCol(c, decimals)
	}
	return df.derive(cols)
}

func roundCol(c column, decimals int) column {
	switch p := c.(type) {
	case *col[float64]:
		out := p.blank(len(p.data))
		for i, v := range p.data {
			out.data[i] = roundHalfEven(v, decimals)
		}
		out.valid = p.valid
		return out
	case *col[float32]:
		out := p.blank(len(p.data))
		for i, v := range p.data {
			out.data[i] = float32(roundHalfEven(float64(v), decimals))
		}
		out.valid = p.valid
		return out
	}
	return c
}

// roundHalfEven rounds as numpy.round does: scale by 10**decimals, round half
// to even, scale back.
func roundHalfEven(v float64, decimals int) float64 {
	if decimals == 0 {
		return math.RoundToEven(v)
	}
	if decimals > 0 {
		p := math.Pow(10, float64(decimals))
		return math.RoundToEven(v*p) / p
	}
	p := math.Pow(10, float64(-decimals))
	return math.RoundToEven(v/p) * p
}
