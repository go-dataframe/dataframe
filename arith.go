package dataframe

import (
	"fmt"
	"math"

	"github.com/go-ndarray/ndarray"
)

type arithOp uint8

const (
	opAdd arithOp = iota
	opSub
	opMul
	opDiv
	opFloorDiv
	opMod
	opPow
)

var arithNames = [...]string{"+", "-", "*", "/", "//", "%", "**"}

// operands aligns the second operand of a binary operation with the series:
// another series is aligned by label (the union of labels, missing where one
// side lacks a label); a scalar becomes a one-element column, broadcast.
func (s *Series) operands(other any) (*Index, column, column, error) {
	if o, ok := other.(*Series); ok {
		return alignSeries(s, o)
	}
	b, err := scalarFor(s.c, other)
	if err != nil {
		return nil, nil, nil, err
	}
	return s.index, s.c, b, nil
}

// scalarFor makes a one-element column of a scalar, in the dtype NumPy 2
// gives a Python scalar next to an array of c's dtype: an int takes an int
// or float column's dtype, a float takes a float column's dtype.
func scalarFor(c column, v any) (column, error) {
	one, err := fromValues([]any{v})
	if err != nil {
		return nil, err
	}
	dt, odt := c.dtype(), one.dtype()
	if (odt == Int64 && dt.isReal()) || (odt == Float64 && dt.isFloat()) {
		if conv, ok := c.convert(v); ok {
			return conv, nil
		}
	}
	return one, nil
}

func (s *Series) arith(op arithOp, other any) (*Series, error) {
	ix, a, b, err := s.operands(other)
	if err != nil {
		return nil, err
	}
	c, err := arithCols(op, a, b)
	if err != nil {
		return nil, err
	}
	return &Series{name: s.name, index: ix, c: c}, nil
}

// arithCols applies an operator elementwise; b has a's length or one element.
func arithCols(op arithOp, a, b column) (column, error) {
	da, db := a.dtype(), b.dtype()
	if da == String && db == String && op == opAdd {
		return concatStrings(a, b), nil
	}
	if !da.IsNumeric() || !db.IsNumeric() {
		return nil, fmt.Errorf("%w: %v %s %v", ErrDType, da, arithNames[op], db)
	}
	n := a.len()
	valid := broadcastValidity(a, b, n)
	aa, _ := a.ndarray()
	ba, _ := b.ndarray()
	var r *ndarray.Array
	var err error
	switch op {
	case opAdd:
		r, err = aa.Add(ba)
	case opSub:
		r, err = aa.Sub(ba)
	case opMul:
		r, err = aa.Mul(ba)
	case opDiv:
		r, err = aa.Div(ba)
	default:
		return floorOps(op, a, b, valid)
	}
	if err != nil {
		return nil, err
	}
	out, _ := fromArray(r)
	return out.withValidity(valid), nil
}

// broadcastValidity combines the validity of an n-element column and an n-
// or one-element column.
func broadcastValidity(a, b column, n int) bitmap {
	va := naBitmap(a)
	if b.len() == n {
		return va.and(naBitmap(b), n)
	}
	if b.isNA(0) {
		return newBitmap(n)
	}
	return va
}

// floorOps computes floor division, modulo and power, in int64 when both
// sides are integers (a zero divisor gives a missing value), in float64
// otherwise, with Python's sign conventions.
func floorOps(op arithOp, a, b column, valid bitmap) (column, error) {
	n := a.len()
	at := func(c column, i int) int {
		if c.len() == 1 {
			return 0
		}
		return i
	}
	if (a.dtype().isInt() || a.dtype() == Bool) && (b.dtype().isInt() || b.dtype() == Bool) {
		x, _ := cast(a, Int64)
		y, _ := cast(b, Int64)
		xs, ys := x.(*col[int64]).data, y.(*col[int64]).data
		out := makeCol[int64](Int64, n)
		for i := range n {
			p, q := xs[i], ys[at(b, i)]
			switch {
			case op == opPow && q >= 0:
				out.data[i] = ipow(p, q)
			case op == opPow:
				return floorOps(op, mustCast(a, Float64), b, valid)
			case q == 0:
				if valid == nil {
					valid = allValid(n)
				}
				valid.clear(i)
			case op == opFloorDiv:
				out.data[i] = floorDiv(p, q)
			default:
				out.data[i] = p - floorDiv(p, q)*q
			}
		}
		out.valid = valid
		return out, nil
	}
	xs, ok1 := floatsOf(a)
	ys, ok2 := floatsOf(b)
	if !ok1 || !ok2 {
		return nil, fmt.Errorf("%w: %v %s %v", ErrDType, a.dtype(), arithNames[op], b.dtype())
	}
	out := makeCol[float64](Float64, n)
	for i := range n {
		p, q := xs[i], ys[at(b, i)]
		switch op {
		case opPow:
			out.data[i] = math.Pow(p, q)
		case opFloorDiv:
			out.data[i] = math.Floor(p / q)
		default:
			m := math.Mod(p, q)
			if m != 0 && (m < 0) != (q < 0) {
				m += q
			}
			out.data[i] = m
		}
	}
	out.valid = valid
	return out, nil
}

func mustCast(c column, dt DType) column {
	out, _ := cast(c, dt)
	return out
}

func floorDiv(p, q int64) int64 {
	d := p / q
	if (p%q != 0) && ((p < 0) != (q < 0)) {
		d--
	}
	return d
}

func ipow(b, e int64) int64 {
	r := int64(1)
	for e > 0 {
		if e&1 == 1 {
			r *= b
		}
		b *= b
		e >>= 1
	}
	return r
}

func concatStrings(a, b column) column {
	n := a.len()
	out := newStrBuilder(n, 0)
	for i := range n {
		j := i
		if b.len() == 1 {
			j = 0
		}
		if !a.isNA(i) && !b.isNA(j) {
			out.buf = append(append(out.buf, a.str(i)...), b.str(j)...)
		}
		out.offs = append(out.offs, int64(len(out.buf)))
	}
	return out.column(broadcastValidity(a, b, n))
}

// Add returns s + other: other is a scalar or a series (aligned by label).
// Types promote as NumPy promotes them; strings concatenate.
func (s *Series) Add(other any) (*Series, error) { return s.arith(opAdd, other) }

// Sub returns s - other.
func (s *Series) Sub(other any) (*Series, error) { return s.arith(opSub, other) }

// Mul returns s * other.
func (s *Series) Mul(other any) (*Series, error) { return s.arith(opMul, other) }

// Div returns s / other, true division (integers give float64).
func (s *Series) Div(other any) (*Series, error) { return s.arith(opDiv, other) }

// FloorDiv returns s // other (Python's floor division).
func (s *Series) FloorDiv(other any) (*Series, error) { return s.arith(opFloorDiv, other) }

// Mod returns s % other, with the sign of the divisor, as in Python.
func (s *Series) Mod(other any) (*Series, error) { return s.arith(opMod, other) }

// Pow returns s ** other.
func (s *Series) Pow(other any) (*Series, error) { return s.arith(opPow, other) }

// Neg returns -s.
func (s *Series) Neg() (*Series, error) {
	if !s.DType().IsNumeric() || s.DType() == Bool {
		return nil, fmt.Errorf("%w: -%v", ErrDType, s.DType())
	}
	a, _ := s.c.ndarray()
	c, _ := fromArray(a.Neg())
	return s.with(c.withValidity(s.c.validity())), nil
}

// Abs returns |s|.
func (s *Series) Abs() (*Series, error) {
	if !s.DType().IsNumeric() {
		return nil, fmt.Errorf("%w: abs of %v", ErrDType, s.DType())
	}
	a, _ := s.c.ndarray()
	c, _ := fromArray(a.Abs())
	return s.with(c.withValidity(s.c.validity())), nil
}

// MapFloat applies f to every present value of a real series, giving a
// float64 series; missing values stay missing.
func (s *Series) MapFloat(f func(float64) float64) (*Series, error) {
	x, err := s.Floats()
	if err != nil {
		return nil, err
	}
	out := makeCol[float64](Float64, len(x))
	for i, v := range x {
		if s.c.isNA(i) {
			out.data[i] = math.NaN()
		} else {
			out.data[i] = f(v)
		}
	}
	return s.with(out), nil
}

// Map applies f to every value (nil for a missing one) and builds a series of
// the results, whose dtype is inferred as for NewSeries([]any); a nil result
// is missing.
func (s *Series) Map(f func(any) any) (*Series, error) {
	vals := make([]any, s.Len())
	for i := range vals {
		vals[i] = f(s.c.value(i))
	}
	c, err := fromValues(vals)
	if err != nil {
		return nil, err
	}
	return s.with(c), nil
}

// Row is one row of a frame, as Apply hands it to a function.
type Row struct {
	df *DataFrame
	i  int
}

// Get returns the value in the named column (nil when missing or when there
// is no such column).
func (r Row) Get(column string) any {
	k, err := r.df.pos(column)
	if err != nil {
		return nil
	}
	return r.df.cols[k].value(r.i)
}

// Float returns the value in the named column as a float64 (NaN when it is
// missing or not a number).
func (r Row) Float(column string) float64 { return asFloat(r.Get(column)) }

// Label returns the row's index label.
func (r Row) Label() any { return r.df.index.Label(r.i) }

// Position returns the row's position.
func (r Row) Position() int { return r.i }

// Apply calls f on every row and returns a series of the results, indexed
// like the frame, with an inferred dtype (pandas' df.apply(f, axis=1)).
func (df *DataFrame) Apply(name string, f func(Row) any) (*Series, error) {
	vals := make([]any, df.Len())
	for i := range vals {
		vals[i] = f(Row{df, i})
	}
	c, err := fromValues(vals)
	if err != nil {
		return nil, err
	}
	return &Series{name: name, index: df.index, c: c}, nil
}

// ApplyColumns calls f on every column and returns the frame of the results,
// which must keep the frame's length.
func (df *DataFrame) ApplyColumns(f func(*Series) (*Series, error)) (*DataFrame, error) {
	cols := make([]column, len(df.cols))
	for k, s := range df.Series() {
		r, err := f(s)
		if err != nil {
			return nil, fmt.Errorf("column %q: %w", df.names[k], err)
		}
		if r.Len() != df.Len() {
			return nil, fmt.Errorf("%w: column %q became %d long", ErrLength, df.names[k], r.Len())
		}
		cols[k] = r.c
	}
	return df.derive(cols), nil
}

// arith applies an operator between every column and a scalar, or between
// the like-named columns of two frames (aligned by label; a column on one
// side only gives an all-missing column, as pandas does).
func (df *DataFrame) arith(op arithOp, other any) (*DataFrame, error) {
	if o, ok := other.(*DataFrame); ok {
		a, b := df, o
		if !df.index.equal(o.index) {
			ix, err := unionIndex([]*Series{{index: df.index}, {index: o.index}})
			if err != nil {
				return nil, err
			}
			a, _ = df.Reindex(ix)
			b, _ = o.Reindex(ix)
		}
		names := append([]string(nil), a.names...)
		for _, n := range b.names {
			if _, err := a.pos(n); err != nil {
				names = append(names, n)
			}
		}
		out := &DataFrame{index: a.index, names: names}
		for _, n := range names {
			i, errA := a.pos(n)
			j, errB := b.pos(n)
			if errA != nil || errB != nil {
				out.cols = append(out.cols, makeCol[float64](Float64, 0).nulls(a.Len()))
				continue
			}
			c, err := arithCols(op, a.cols[i], b.cols[j])
			if err != nil {
				return nil, fmt.Errorf("column %q: %w", n, err)
			}
			out.cols = append(out.cols, c)
		}
		return out, nil
	}
	cols := make([]column, len(df.cols))
	for k, s := range df.Series() {
		r, err := s.arith(op, other)
		if err != nil {
			return nil, fmt.Errorf("column %q: %w", df.names[k], err)
		}
		cols[k] = r.c
	}
	return df.derive(cols), nil
}

// Add returns df + other, other a scalar or a frame (aligned on labels and
// column names).
func (df *DataFrame) Add(other any) (*DataFrame, error) { return df.arith(opAdd, other) }

// Sub returns df - other.
func (df *DataFrame) Sub(other any) (*DataFrame, error) { return df.arith(opSub, other) }

// Mul returns df * other.
func (df *DataFrame) Mul(other any) (*DataFrame, error) { return df.arith(opMul, other) }

// Div returns df / other.
func (df *DataFrame) Div(other any) (*DataFrame, error) { return df.arith(opDiv, other) }
