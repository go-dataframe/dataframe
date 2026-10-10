package dataframe

import (
	"errors"
	"math"
	"testing"
	"time"

	"github.com/go-ndarray/ndarray"
)

func fails(t *testing.T, err error, want error) {
	t.Helper()
	if err == nil {
		t.Fatalf("no error, want %v", want)
	}
	if want != nil && !errors.Is(err, want) {
		t.Fatalf("error %v, want %v", err, want)
	}
}

func TestDTypeNames(t *testing.T) {
	for d := DType(0); d < numDTypes; d++ {
		p, err := ParseDType(d.String())
		if err != nil || p != d {
			t.Errorf("round trip %v: %v %v", d, p, err)
		}
	}
	if DType(99).String() != "DType(99)" {
		t.Error(DType(99).String())
	}
	for name, want := range map[string]DType{"int": Int64, "float": Float64, "str": String,
		"object": String, "datetime64[ns]": Datetime, " Datetime64 ": Datetime} {
		if got := Must(ParseDType(name)); got != want {
			t.Errorf("%q: %v", name, got)
		}
	}
	_, err := ParseDType("nope")
	fails(t, err, ErrDType)
	if !Int8.IsNumeric() || String.IsNumeric() {
		t.Error("IsNumeric")
	}
	if _, ok := String.storage(); ok {
		t.Error("string storage")
	}
	if d, _ := Category.storage(); d != ndarray.Int32 {
		t.Error("category storage")
	}
}

func TestMust(t *testing.T) {
	if Must(1, nil) != 1 {
		t.Fatal()
	}
	defer func() {
		if recover() == nil {
			t.Error("no panic")
		}
	}()
	Must(0, errors.New("x"))
}

func TestBitmap(t *testing.T) {
	b := allValid(130)
	b.clear(3)
	b.clear(129)
	if b.count(130) != 128 || b.count(64) != 63 || bitmap(nil).count(5) != 5 {
		t.Error("count", b.count(130))
	}
	if s := b.slice(2, 5); s.get(1) || !s.get(0) || !s.get(2) {
		t.Error("slice")
	}
	if bitmap(nil).slice(0, 3) != nil {
		t.Error("nil slice")
	}
	if b.take([]int{0, 1, 2}) != nil {
		t.Error("take all valid")
	}
	if tk := b.take([]int{3, -1, 0}); tk.get(0) || tk.get(1) || !tk.get(2) {
		t.Error("take")
	}
	if a := bitmap(nil).and(b, 130); a.get(3) || !a.get(4) {
		t.Error("and nil")
	}
	if a := b.and(nil, 130); a.get(3) {
		t.Error("and other nil")
	}
	c := allValid(130)
	c.clear(4)
	if a := b.and(c, 130); a.get(3) || a.get(4) || !a.get(5) {
		t.Error("and")
	}
}

// everyDType returns a three-element column of every dtype, the middle
// element missing.
func everyDType(t *testing.T) []column {
	t.Helper()
	datas := []any{
		[]float64{1, 2, 3}, []float32{1, 2, 3}, []int{1, 2, 3}, []int8{1, 2, 3}, []int16{1, 2, 3},
		[]int32{1, 2, 3}, []int64{1, 2, 3}, []uint8{1, 2, 3}, []uint16{1, 2, 3}, []uint32{1, 2, 3},
		[]uint64{1, 2, 3}, []complex64{1, 2, 3}, []complex128{1, 2, 3}, []bool{true, false, true},
		[]string{"a", "b", "c"},
		[]time.Time{time.Unix(0, 0).UTC(), time.Unix(86400, 0).UTC(), time.Unix(172800, 0).UTC()},
	}
	var out []column
	for _, d := range datas {
		s := Must(NewSeriesMasked("v", d, []bool{true, false, true}))
		out = append(out, s.c)
	}
	cat := Must(Must(NewSeries("c", []string{"x", "y", "x"})).AsType(Category))
	out = append(out, cat.c.withValidity(b3()))
	p := Must(Must(NewSeries("p", []time.Time{time.Unix(0, 0), time.Unix(4e7, 0), time.Unix(8e7, 0)})).ToPeriod("M"))
	out = append(out, p.c.withValidity(b3()))
	return out
}

func b3() bitmap {
	b := allValid(3)
	b.clear(1)
	return b
}

func TestColumnsOfEveryDType(t *testing.T) {
	for _, c := range everyDType(t) {
		dt := c.dtype()
		if c.len() != 3 || c.naCount() != 1 || !c.isNA(1) || c.isNA(0) {
			t.Errorf("%v: missing", dt)
		}
		if c.value(1) != nil || c.value(0) == nil {
			t.Errorf("%v: value", dt)
		}
		if c.compare(0, 2) > 0 {
			t.Errorf("%v: compare", dt)
		}
		if !c.equal(0, c, 0) {
			t.Errorf("%v: equal", dt)
		}
		tk := c.take([]int{2, -1, 0})
		if tk.isNA(0) || !tk.isNA(1) || c.str(0) != tk.str(2) {
			t.Errorf("%v: take", dt)
		}
		if sl := c.slice(1, 3); sl.len() != 2 || !sl.isNA(0) {
			t.Errorf("%v: slice", dt)
		}
		if full := c.slice(0, 3); full.len() != 3 {
			t.Errorf("%v: full slice", dt)
		}
		codes, u := c.factorize()
		if codes[1] != -1 || u.len() > 2 {
			t.Errorf("%v: factorize %v", dt, codes)
		}
		a, b, u2 := c.factorizeWith(c.slice(0, 1))
		if b[0] != a[0] || u2.len() != u.len() {
			t.Errorf("%v: factorizeWith", dt)
		}
		if ap := c.appendCol(c); ap.len() != 6 || !ap.isNA(4) {
			t.Errorf("%v: append", dt)
		}
		if n := c.nulls(2); n.naCount() != 2 {
			t.Errorf("%v: nulls", dt)
		}
		one, ok := c.convert(c.value(0))
		if !ok || !one.equal(0, c, 0) {
			t.Errorf("%v: convert", dt)
		}
		if _, ok := c.convert(struct{}{}); ok {
			t.Errorf("%v: convert junk", dt)
		}
		f, err := c.fill(c.value(2))
		if err != nil || f.naCount() != 0 {
			t.Errorf("%v: fill %v", dt, err)
		}
		if _, err := c.fill(struct{}{}); err == nil {
			t.Errorf("%v: fill junk", dt)
		}
		find := c.lookup()
		if p := find(c.value(2)); len(p) == 0 || p[len(p)-1] != 2 {
			t.Errorf("%v: lookup %v", dt, p)
		}
		if find(struct{}{}) != nil {
			t.Errorf("%v: lookup junk", dt)
		}
		if arr, ok := c.ndarray(); ok != (dt < String || dt == Datetime || dt == Period) || (ok && arr.Size() != 3) {
			t.Errorf("%v: ndarray", dt)
		}
		if w := c.withValidity(nil); w.naCount() != 0 {
			t.Errorf("%v: withValidity", dt)
		}
		if s, err := cast(c, String); err != nil || s.value(0) != c.str(0) {
			t.Errorf("%v: to string %v", dt, err)
		}
	}
}

func TestEqualAcrossTypes(t *testing.T) {
	a := newCol(Int64, []int64{1}, nil)
	b := newCol(Float64, []float64{1}, nil)
	if a.equal(0, b, 0) {
		t.Error("int equals float column")
	}
	cat := Must(toCategory(newCol(String, []string{"u", "v"}, nil)))
	cat2 := Must(toCategory(newCol(String, []string{"v"}, nil)))
	if !cat.equal(1, cat2, 0) || cat.equal(0, cat2, 0) {
		t.Error("category equal by label")
	}
}

func TestFromAny(t *testing.T) {
	_, err := fromAny(42)
	fails(t, err, ErrArgument)
	m, _ := ndarray.New(2, 2)
	_, err = fromAny(m)
	fails(t, err, ErrArgument)
	// A strided view is copied, a contiguous array shared.
	a, _ := ndarray.FromData([]float64{1, 2, 3, 4}, 4)
	c := Must(fromAny(a))
	if got, _ := c.ndarray(); got != a {
		t.Error("contiguous array not shared")
	}
	v, _ := a.Slice(ndarray.Step(2))
	c = Must(fromAny(v))
	if c.len() != 2 || c.value(1) != 3.0 {
		t.Error("strided view", c.value(1))
	}
	if got, _ := c.ndarray(); got == v {
		t.Error("strided view shared")
	}
	s := Must(NewSeries("s", []int{1}))
	if c := Must(fromAny(s)); c != s.c {
		t.Error("series column")
	}
	// Every ndarray dtype wraps.
	for d := ndarray.Float64; d <= ndarray.Complex128; d++ {
		x, _ := ndarray.ZerosOf(d, 2)
		c := Must(fromArray(x))
		if c.dtype() != DType(d) {
			t.Errorf("wrap %v", d)
		}
	}
	z, _ := ndarray.ZerosOf(ndarray.Int32, 0)
	if c := Must(fromArray(z)); c.len() != 0 {
		t.Error("empty array")
	}
}

func TestFromValues(t *testing.T) {
	cases := []struct {
		in   []any
		want DType
	}{
		{[]any{nil, nil}, Float64},
		{[]any{true, nil}, Bool},
		{[]any{1, int8(2), uint64(3)}, Int64},
		{[]any{1, 2.5}, Float64},
		{[]any{float32(1), 2}, Float64},
		{[]any{1, complex(1, 2)}, Complex128},
		{[]any{"a", nil}, String},
		{[]any{time.Unix(0, 0)}, Datetime},
	}
	for _, c := range cases {
		col := Must(fromValues(c.in))
		if col.dtype() != c.want {
			t.Errorf("%v: %v", c.in, col.dtype())
		}
	}
	for _, bad := range [][]any{{struct{}{}}, {"a", 1}, {true, 1}, {time.Now(), "x"}} {
		_, err := fromValues(bad)
		fails(t, err, ErrArgument)
	}
	if _, err := fullCol(struct{}{}, 2); err == nil {
		t.Error("fullCol junk")
	}
}

func TestOpsConversions(t *testing.T) {
	for _, v := range []any{int(1), int8(1), int16(1), int32(1), int64(1), uint(1), uint8(1),
		uint16(1), uint32(1), uint64(1), float64(1), float32(1)} {
		if i, ok := toInt(v); !ok || i != 1 {
			t.Errorf("toInt %T", v)
		}
		if f, ok := toFloat(v); !ok || f != 1 {
			t.Errorf("toFloat %T", v)
		}
	}
	for _, v := range []any{1.5, float32(1.5), uint64(math.MaxUint64), uint64(1 << 63), math.Inf(1), "x"} {
		if _, ok := toInt(v); ok {
			t.Errorf("toInt %v", v)
		}
	}
	if _, ok := toFloat("x"); ok {
		t.Error("toFloat string")
	}
	if _, ok := opsInt8.unbox(300); ok {
		t.Error("int8 overflow")
	}
	if v, ok := opsUint64.unbox(uint64(math.MaxUint64)); !ok || v != math.MaxUint64 {
		t.Error("uint64 max")
	}
	if _, ok := opsUint8.unbox(uint64(300)); ok {
		t.Error("uint8 overflow")
	}
	if _, ok := opsUint8.unbox(-1); ok {
		t.Error("uint8 negative")
	}
	if v, ok := opsComplex64.unbox(complex64(1 + 2i)); !ok || v != 1+2i {
		t.Error("complex64")
	}
	if v, ok := opsComplex128.unbox(2); !ok || v != 2 {
		t.Error("complex from int")
	}
	if opsComplex128.cmp(1+1i, 1+2i) >= 0 || opsComplex128.cmp(2, 1+5i) <= 0 {
		t.Error("complex order")
	}
	if !opsComplex128.nan(complex(math.NaN(), 0)) || opsComplex64.nan(1) {
		t.Error("complex nan")
	}
	if opsBool.cmp(true, true) != 0 || opsBool.cmp(false, true) != -1 || opsBool.cmp(true, false) != 1 {
		t.Error("bool order")
	}
	if ns, ok := toNanos("2020-01-02"); !ok || ns != 1577923200e9 {
		t.Error("toNanos string")
	}
	if _, ok := toNanos(3); ok {
		t.Error("toNanos int")
	}
	for f, want := range map[float64]string{1: "1.0", 0.1: "0.1", 1e-5: "1e-05", 1e16: "1e+16",
		123456789012345.6: "123456789012345.6", math.NaN(): "nan", math.Inf(1): "inf", math.Inf(-1): "-inf", -2.5: "-2.5"} {
		if got := reprFloat(f, 64); got != want {
			t.Errorf("reprFloat(%v) = %q, want %q", f, got, want)
		}
	}
	for ns, want := range map[int64]string{0: "1970-01-01 00:00:00", 1500: "1970-01-01 00:00:00.000001500",
		2000: "1970-01-01 00:00:00.000002"} {
		if got := formatTime(ns); got != want {
			t.Errorf("formatTime %d: %q", ns, got)
		}
	}
}
