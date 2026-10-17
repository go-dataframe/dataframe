package dataframe

import (
	"fmt"
	"slices"
	"sync"

	"github.com/go-ndarray/ndarray"
)

// column is one typed, immutable vector of values with a validity bitmap. All
// the dtypes share one generic implementation, col[T]; what differs between
// dtypes (ordering, formatting, boxing to a Go value, missing NaNs) lives in
// an ops table, so most code is written once and is the same machine code
// path for every dtype of a given storage type.
type column interface {
	dtype() DType
	len() int
	isNA(i int) bool
	naCount() int
	validity() bitmap
	// take gathers the rows at idx; a negative position gives a missing row.
	take(idx []int) column
	slice(i, j int) column
	// value returns the element as a Go value, nil when missing.
	value(i int) any
	// str formats a present element.
	str(i int) string
	// compare orders two present elements.
	compare(i, j int) int
	// equal reports whether two present elements are equal.
	equal(i int, o column, j int) bool
	// ndarray returns the column's data as an array (missing elements keep
	// whatever value the storage holds), and false for a dtype an array
	// cannot hold.
	ndarray() (*ndarray.Array, bool)
	// withValidity returns the column sharing data, with another bitmap.
	withValidity(b bitmap) column
	// factorize maps each element to a dense code, -1 for missing, and
	// returns the distinct values in order of first appearance.
	factorize() ([]int32, column)
	// factorizeWith codes this column and o (of the same dtype) in one code
	// space: the codes of o reuse this column's codes for equal values.
	factorizeWith(o column) (a, b []int32, uniq column)
	// appendCol returns this column followed by o, which has the same dtype.
	appendCol(o column) column
	// nulls returns a column of n missing elements of this dtype.
	nulls(n int) column
	// fill replaces missing elements by v, which must convert to the dtype.
	fill(v any) (column, error)
	// convert converts a Go value to a one-element column of this dtype.
	convert(v any) (column, bool)
	// lookup returns a function giving the rows equal to a value.
	lookup() func(any) []int
}

// ops is what one dtype contributes to col[T].
type ops[T comparable] struct {
	cmp   func(a, b T) int
	nan   func(T) bool // nil when the dtype has no in-band missing value
	str   func(T) string
	box   func(T) any
	unbox func(any) (T, bool)
}

// col is the one implementation of column.
type col[T comparable] struct {
	dt    DType
	data  []T
	arr   *ndarray.Array // the array whose storage is exactly data, or nil
	valid bitmap
	o     *ops[T]
	// levels are a Category column's categories; freq is a Period column's
	// frequency. Both are carried unchanged through take, slice and so on.
	levels []string
	freq   string
}

// alloc returns n zero elements of the dtype, born inside an ndarray, so
// that the column hands the array out without copying. Every dtype col[T]
// holds has array storage (strings live in strCol).
func alloc[T comparable](dt DType, n int) (*ndarray.Array, []T) {
	nd, _ := dt.storage()
	a, _ := ndarray.ZerosOf(nd, n)
	return a, arrayData[T](a)
}

// arrayData returns a contiguous array's storage as []T.
func arrayData[T comparable](a *ndarray.Array) []T {
	var s any
	switch a.DType() {
	case ndarray.Float64:
		s, _ = ndarray.Data[float64](a)
	case ndarray.Bool:
		s, _ = ndarray.Data[bool](a)
	case ndarray.Int8:
		s, _ = ndarray.Data[int8](a)
	case ndarray.Int16:
		s, _ = ndarray.Data[int16](a)
	case ndarray.Int32:
		s, _ = ndarray.Data[int32](a)
	case ndarray.Int64:
		s, _ = ndarray.Data[int64](a)
	case ndarray.Uint8:
		s, _ = ndarray.Data[uint8](a)
	case ndarray.Uint16:
		s, _ = ndarray.Data[uint16](a)
	case ndarray.Uint32:
		s, _ = ndarray.Data[uint32](a)
	case ndarray.Uint64:
		s, _ = ndarray.Data[uint64](a)
	case ndarray.Float32:
		s, _ = ndarray.Data[float32](a)
	case ndarray.Complex64:
		s, _ = ndarray.Data[complex64](a)
	default:
		s, _ = ndarray.Data[complex128](a)
	}
	return s.([]T)
}

// makeCol returns a column of n zero elements to be filled in place.
func makeCol[T comparable](dt DType, n int) *col[T] {
	a, d := alloc[T](dt, n)
	return &col[T]{dt: dt, data: d, arr: a, o: opsOf[T](dt)}
}

// newCol copies data into a new column.
func newCol[T comparable](dt DType, data []T, valid bitmap) *col[T] {
	c := makeCol[T](dt, len(data))
	copy(c.data, data)
	c.valid = valid
	return c
}

// derive returns a column like c (dtype, levels, freq) over other data.
func (c *col[T]) derive(arr *ndarray.Array, data []T, valid bitmap) *col[T] {
	return &col[T]{dt: c.dt, data: data, arr: arr, valid: valid, o: c.o,
		levels: c.levels, freq: c.freq}
}

// blank returns an all-zero column like c of length n.
func (c *col[T]) blank(n int) *col[T] {
	a, d := alloc[T](c.dt, n)
	return c.derive(a, d, nil)
}

func (c *col[T]) dtype() DType     { return c.dt }
func (c *col[T]) len() int         { return len(c.data) }
func (c *col[T]) validity() bitmap { return c.valid }

func (c *col[T]) isNA(i int) bool {
	return !c.valid.get(i) || (c.o.nan != nil && c.o.nan(c.data[i]))
}

func (c *col[T]) naCount() int {
	if c.valid == nil && c.o.nan == nil {
		return 0
	}
	n := 0
	for i := range c.data {
		if c.isNA(i) {
			n++
		}
	}
	return n
}

func (c *col[T]) take(idx []int) column {
	out := c.blank(len(idx))
	parallel(len(idx), func(lo, hi int) {
		d, src := out.data[lo:hi], c.data
		for k, i := range idx[lo:hi] {
			if i >= 0 {
				d[k] = src[i]
			}
		}
	})
	out.valid = c.valid.take(idx)
	return out
}

func (c *col[T]) slice(i, j int) column {
	out := c.derive(nil, c.data[i:j:j], c.valid.slice(i, j))
	if i == 0 && j == len(c.data) {
		out.arr = c.arr
	}
	return out
}

func (c *col[T]) value(i int) any {
	if c.isNA(i) {
		return nil
	}
	return c.o.box(c.data[i])
}

func (c *col[T]) str(i int) string     { return c.o.str(c.data[i]) }
func (c *col[T]) compare(i, j int) int { return c.o.cmp(c.data[i], c.data[j]) }

func (c *col[T]) equal(i int, o column, j int) bool {
	oc, ok := o.(*col[T])
	if !ok || oc.dt != c.dt {
		return false
	}
	if c.dt == Category {
		return c.str(i) == oc.str(j)
	}
	return c.data[i] == oc.data[j]
}

func (c *col[T]) ndarray() (*ndarray.Array, bool) {
	if _, ok := c.dt.storage(); !ok || c.dt == Category {
		return nil, false
	}
	if c.arr == nil {
		a, d := alloc[T](c.dt, len(c.data))
		copy(d, c.data)
		c.arr = a
		c.data = d
	}
	return c.arr, true
}

func (c *col[T]) withValidity(b bitmap) column {
	out := *c
	out.valid = b
	return &out
}

func (c *col[T]) factorize() ([]int32, column) {
	codes := make([]int32, len(c.data))
	m := make(map[T]int32)
	var uniq []T
	for i, v := range c.data {
		if c.isNA(i) {
			codes[i] = -1
			continue
		}
		k, ok := m[v]
		if !ok {
			k = int32(len(uniq))
			m[v] = k
			uniq = append(uniq, v)
		}
		codes[i] = k
	}
	return codes, c.withData(uniq)
}

// withData returns a column like c holding a copy of data, all present.
func (c *col[T]) withData(data []T) *col[T] {
	out := c.blank(len(data))
	copy(out.data, data)
	return out
}

func (c *col[T]) factorizeWith(o column) ([]int32, []int32, column) {
	oc := o.(*col[T])
	d := &dict[T]{m: make(map[T]int32)}
	a := d.code(c)
	b := d.code(oc)
	return a, b, c.withData(d.uniq)
}

// dict assigns dense codes to values in order of first appearance.
type dict[T comparable] struct {
	m    map[T]int32
	uniq []T
}

func (d *dict[T]) add(v T) int32 {
	k, ok := d.m[v]
	if !ok {
		k = int32(len(d.uniq))
		d.m[v] = k
		d.uniq = append(d.uniq, v)
	}
	return k
}

// codeRange codes rows [lo, hi) of src into codes (-1 for missing).
func (d *dict[T]) codeRange(src *col[T], codes []int32, lo, hi int) {
	for i := lo; i < hi; i++ {
		if src.isNA(i) {
			codes[i] = -1
			continue
		}
		codes[i] = d.add(src.data[i])
	}
}

// code codes every row of src. A large column is coded in parallel: each
// chunk against a dictionary of its own, then the chunk dictionaries are
// merged in chunk order, which keeps first-appearance order overall, and
// the codes translated.
func (d *dict[T]) code(src *col[T]) []int32 {
	n := len(src.data)
	codes := make([]int32, n)
	if n < 4*parallelMin {
		d.codeRange(src, codes, 0, n)
		return codes
	}
	type part struct {
		lo, hi int
		local  *dict[T]
	}
	var parts []*part
	var mu sync.Mutex
	parallel(n, func(lo, hi int) {
		p := &part{lo: lo, hi: hi, local: &dict[T]{m: make(map[T]int32)}}
		p.local.codeRange(src, codes, lo, hi)
		mu.Lock()
		parts = append(parts, p)
		mu.Unlock()
	})
	slices.SortFunc(parts, func(a, b *part) int { return a.lo - b.lo })
	trans := make([][]int32, len(parts))
	for k, p := range parts {
		trans[k] = make([]int32, len(p.local.uniq))
		for j, v := range p.local.uniq {
			trans[k][j] = d.add(v)
		}
	}
	// parallel splits n the same way again: one chunk per part.
	parallel(n, func(lo, hi int) {
		k, _ := slices.BinarySearchFunc(parts, lo, func(p *part, lo int) int { return p.lo - lo })
		t := trans[k]
		for i := lo; i < hi; i++ {
			if codes[i] >= 0 {
				codes[i] = t[codes[i]]
			}
		}
	})
	return codes
}

func (c *col[T]) appendCol(o column) column {
	oc := o.(*col[T])
	n, m := len(c.data), len(oc.data)
	out := c.blank(n + m)
	copy(out.data, c.data)
	copy(out.data[n:], oc.data)
	if c.valid != nil || oc.valid != nil {
		out.valid = allValid(n + m)
		for i := range n {
			if !c.valid.get(i) {
				out.valid.clear(i)
			}
		}
		for i := range m {
			if !oc.valid.get(i) {
				out.valid.clear(n + i)
			}
		}
	}
	return out
}

func (c *col[T]) nulls(n int) column {
	out := c.blank(n)
	out.valid = newBitmap(n)
	return out
}

func (c *col[T]) convert(v any) (column, bool) {
	x, ok := c.o.unbox(v)
	if !ok {
		return nil, false
	}
	return c.withData([]T{x}), true
}

func (c *col[T]) fill(v any) (column, error) {
	x, ok := c.o.unbox(v)
	if !ok {
		return nil, fmt.Errorf("%w: cannot fill a %v column with %T", ErrDType, c.dt, v)
	}
	out := c.withData(c.data)
	for i := range c.data {
		if c.isNA(i) {
			out.data[i] = x
		}
	}
	return out, nil
}

func (c *col[T]) lookup() func(any) []int {
	m := make(map[T][]int, len(c.data))
	for i, v := range c.data {
		if !c.isNA(i) {
			m[v] = append(m[v], i)
		}
	}
	return func(key any) []int {
		x, ok := c.o.unbox(key)
		if !ok {
			return nil
		}
		return m[x]
	}
}

// opsOf returns the ops of a dtype whose storage element is T.
func opsOf[T comparable](dt DType) *ops[T] {
	var o any
	switch dt {
	case Float64:
		o = opsFloat64
	case Float32:
		o = opsFloat32
	case Bool:
		o = opsBool
	case Int8:
		o = opsInt8
	case Int16:
		o = opsInt16
	case Int32:
		o = opsInt32
	case Int64:
		o = opsInt64
	case Uint8:
		o = opsUint8
	case Uint16:
		o = opsUint16
	case Uint32:
		o = opsUint32
	case Uint64:
		o = opsUint64
	case Complex64:
		o = opsComplex64
	case Complex128:
		o = opsComplex128
	case Datetime:
		o = opsDatetime
	default:
		// Category and Period get a per-column table; see catCol and
		// periodCol. Until then, the plain storage ops.
		if dt == Category {
			o = opsInt32
		} else {
			o = opsInt64
		}
	}
	return o.(*ops[T])
}
