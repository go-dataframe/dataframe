package dataframe

import (
	"fmt"
)

type cmpOp uint8

const (
	cmpEq cmpOp = iota
	cmpNe
	cmpLt
	cmpLe
	cmpGt
	cmpGe
)

func (op cmpOp) holds(r int) bool {
	switch op {
	case cmpEq:
		return r == 0
	case cmpNe:
		return r != 0
	case cmpLt:
		return r < 0
	case cmpLe:
		return r <= 0
	case cmpGt:
		return r > 0
	}
	return r >= 0
}

// compareWith compares the series elementwise with a scalar or a series
// (aligned by label). A comparison with a missing value is false, except !=,
// which is true, as for NaN in pandas.
func (s *Series) compareWith(op cmpOp, other any) (*Series, error) {
	ix, a, b, err := s.operands(other)
	if err != nil {
		return nil, err
	}
	out, err := compareCols(op, a, b)
	if err != nil {
		return nil, err
	}
	return &Series{name: s.name, index: ix, c: out}, nil
}

// compareCols compares a with b (of a's length or one element).
func compareCols(op cmpOp, a, b column) (*col[bool], error) {
	n := a.len()
	out := makeCol[bool](Bool, n)
	if fastCompare(op, a, b, out.data) {
		return out, nil
	}
	dt, err := commonType(a.dtype(), b.dtype())
	if (a.dtype() == Datetime && b.dtype() == String) || (a.dtype() == String && b.dtype() == Datetime) {
		// Instants compare with date strings, parsed.
		dt, err = Datetime, nil
	}
	if err != nil {
		return nil, err
	}
	if dt == Complex64 || dt == Complex128 {
		if op != cmpEq && op != cmpNe {
			return nil, fmt.Errorf("%w: complex numbers have no order", ErrDType)
		}
	}
	if a, err = cast(a, dt); err != nil {
		return nil, err
	}
	if b, err = cast(b, dt); err != nil {
		return nil, err
	}
	both := a.appendCol(b)
	for i := range n {
		j := i
		if b.len() == 1 {
			j = 0
		}
		if a.isNA(i) || b.isNA(j) {
			out.data[i] = op == cmpNe
			continue
		}
		out.data[i] = op.holds(both.compare(i, n+j))
	}
	return out, nil
}

// fastCompare handles a float64 or int64 column against a float64 or int64
// scalar, and a string column against a string, with tight loops, the
// common filters.
func fastCompare(op cmpOp, a, b column, out []bool) bool {
	if b.len() != 1 || b.isNA(0) {
		return false
	}
	var x float64
	switch p := b.(type) {
	case stringCol:
		s, ok := a.(stringCol)
		if !ok {
			return false
		}
		s.cmpScalar(op, p.at(0), out)
		return true
	case *col[float64]:
		x = p.data[0]
	case *col[int64]:
		if p.dt != Int64 {
			return false
		}
		x = float64(p.data[0])
	default:
		return false
	}
	switch p := a.(type) {
	case *col[float64]:
		if p.valid != nil {
			return false
		}
		parallel(len(out), func(lo, hi int) { cmpLoop(op, p.data[lo:hi], x, out[lo:hi]) })
		return true
	case *col[int64]:
		if p.valid != nil || p.dt != Int64 || float64(int64(x)) != x {
			return false
		}
		parallel(len(out), func(lo, hi int) { cmpLoop(op, p.data[lo:hi], int64(x), out[lo:hi]) })
		return true
	}
	return false
}

func cmpLoop[T float64 | int64](op cmpOp, d []T, x T, out []bool) {
	switch op {
	case cmpEq:
		for i, v := range d {
			out[i] = v == x
		}
	case cmpNe:
		for i, v := range d {
			out[i] = v != x
		}
	case cmpLt:
		for i, v := range d {
			out[i] = v < x
		}
	case cmpLe:
		for i, v := range d {
			out[i] = v <= x
		}
	case cmpGt:
		for i, v := range d {
			out[i] = v > x
		}
	default:
		for i, v := range d {
			out[i] = v >= x
		}
	}
}

// Eq returns s == other, elementwise; other is a scalar or a series.
func (s *Series) Eq(other any) (*Series, error) { return s.compareWith(cmpEq, other) }

// Ne returns s != other.
func (s *Series) Ne(other any) (*Series, error) { return s.compareWith(cmpNe, other) }

// Lt returns s < other.
func (s *Series) Lt(other any) (*Series, error) { return s.compareWith(cmpLt, other) }

// Le returns s <= other.
func (s *Series) Le(other any) (*Series, error) { return s.compareWith(cmpLe, other) }

// Gt returns s > other.
func (s *Series) Gt(other any) (*Series, error) { return s.compareWith(cmpGt, other) }

// Ge returns s >= other.
func (s *Series) Ge(other any) (*Series, error) { return s.compareWith(cmpGe, other) }

// Between returns lo <= s <= hi.
func (s *Series) Between(lo, hi any) (*Series, error) {
	a, err := s.Ge(lo)
	if err != nil {
		return nil, err
	}
	b, err := s.Le(hi)
	if err != nil {
		return nil, err
	}
	return a.And(b)
}

// logical combines two bool series (aligned by label; missing is false).
func (s *Series) logical(o *Series, f func(a, b bool) bool) (*Series, error) {
	ix, a, b, err := alignSeries(s, o)
	if err != nil {
		return nil, err
	}
	x, err := (&Series{c: a}).bools()
	if err != nil {
		return nil, err
	}
	y, err := (&Series{c: b}).bools()
	if err != nil {
		return nil, err
	}
	out := makeCol[bool](Bool, len(x))
	for i := range x {
		out.data[i] = f(x[i], y[i])
	}
	return &Series{name: s.name, index: ix, c: out}, nil
}

// And returns s & o for bool series.
func (s *Series) And(o *Series) (*Series, error) {
	return s.logical(o, func(a, b bool) bool { return a && b })
}

// Or returns s | o for bool series.
func (s *Series) Or(o *Series) (*Series, error) {
	return s.logical(o, func(a, b bool) bool { return a || b })
}

// Xor returns s ^ o for bool series.
func (s *Series) Xor(o *Series) (*Series, error) {
	return s.logical(o, func(a, b bool) bool { return a != b })
}

// Not returns ~s for a bool series (missing gives true, as ~False).
func (s *Series) Not() (*Series, error) {
	x, err := s.bools()
	if err != nil {
		return nil, err
	}
	out := makeCol[bool](Bool, len(x))
	for i, b := range x {
		out.data[i] = !b
	}
	return s.with(out), nil
}

// IsIn returns whether each value is one of values (pandas' isin). A missing
// value is never in.
func (s *Series) IsIn(values ...any) (*Series, error) {
	out := makeCol[bool](Bool, s.Len())
	if len(values) == 0 {
		return s.with(out), nil
	}
	v, err := fromValues(values)
	if err != nil {
		return nil, err
	}
	if c, ok := s.c.(stringCol); ok {
		if set, ok := v.(stringCol); ok {
			// Strings against strings: no codes needed, a lookup in the
			// set of the present values does.
			m := make(map[string]struct{}, set.len())
			for i := range set.len() {
				if !set.isNA(i) {
					m[set.at(i)] = struct{}{}
				}
			}
			c.isInSet(m, out.data)
			return s.with(out), nil
		}
	}
	a, b, n, err := jointCodes([]column{s.c}, []column{v})
	if err != nil {
		return nil, err
	}
	in := make([]bool, n)
	for _, c := range b {
		if c >= 0 {
			in[c] = true
		}
	}
	for i, c := range a {
		out.data[i] = c >= 0 && in[c]
	}
	return s.with(out), nil
}
