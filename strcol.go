package dataframe

import (
	"encoding/binary"
	"fmt"
	"math"
	mbits "math/bits"
	"runtime"
	"strings"
	"sync"
	"unsafe"

	"github.com/go-ndarray/ndarray"
)

// strCol is a string column laid out as Arrow lays one out: the bytes of
// every value end to end in one buffer, and offsets, value i being
// data[offs[i]:offs[i+1]]. Nothing in it is a pointer the garbage collector
// has to follow value by value, which a []string of a million values would
// make it do a million times at every collection. O is int32 while the
// bytes fit (2 GiB), int64 beyond.
//
// The buffer is immutable once the column exists; slices of a column share
// it, and offs[0] need not be 0. A missing value may hold any bytes.
type strCol[O int32 | int64] struct {
	data  string
	offs  []O
	valid bitmap
}

// stringCol is either strCol, for the code that needs a string column's
// values without caring how wide its offsets are.
type stringCol interface {
	column
	at(i int) string
	// raw returns the buffer and the offsets ([]int32 or []int64).
	raw() (data string, offs any)
	cmpScalar(op cmpOp, x string, out []bool)
	isInSet(m map[string]struct{}, out []bool)
	codeRange(d *dict[string], codes []int32, lo, hi int)
}

func (c *strCol[O]) at(i int) string { return c.data[c.offs[i]:c.offs[i+1]] }

func (c *strCol[O]) raw() (string, any) { return c.data, c.offs }

func (c *strCol[O]) dtype() DType     { return String }
func (c *strCol[O]) len() int         { return len(c.offs) - 1 }
func (c *strCol[O]) validity() bitmap { return c.valid }
func (c *strCol[O]) isNA(i int) bool  { return !c.valid.get(i) }

func (c *strCol[O]) naCount() int { return c.len() - c.valid.count(c.len()) }

// strBuilder lays strings out one after the other into a string column.
type strBuilder struct {
	buf  []byte
	offs []int64
}

func newStrBuilder(n, bytes int) *strBuilder {
	b := &strBuilder{buf: make([]byte, 0, bytes), offs: make([]int64, 1, n+1)}
	return b
}

func (b *strBuilder) add(s string) {
	b.buf = append(b.buf, s...)
	b.offs = append(b.offs, int64(len(b.buf)))
}

// column returns the strings added, with validity valid.
func (b *strBuilder) column(valid bitmap) column {
	return strColumn(unsafe.String(unsafe.SliceData(b.buf), len(b.buf)), b.offs, valid)
}

// maxStr32 is the most bytes a string column holds with int32 offsets; a
// variable so that tests can take the int64 paths without gigabytes.
var maxStr32 int64 = math.MaxInt32

// strColumn returns a column over data with int64 offsets, narrowed to
// int32 offsets when the bytes allow.
func strColumn(data string, offs []int64, valid bitmap) column {
	if offs[len(offs)-1] <= maxStr32 {
		o32 := make([]int32, len(offs))
		for i, o := range offs {
			o32[i] = int32(o)
		}
		return &strCol[int32]{data: data, offs: o32, valid: valid}
	}
	return &strCol[int64]{data: data, offs: offs, valid: valid}
}

// newStrCol copies strs into a string column with validity valid, span by
// span (see gather).
func newStrCol(strs []string, valid bitmap) column {
	n := len(strs)
	sp := spans(n)
	sizes := make([]int64, len(sp)+1)
	eachSpan(sp, func(k, lo, hi int) {
		t := int64(0)
		for _, s := range strs[lo:hi] {
			t += int64(len(s))
		}
		sizes[k+1] = t
	})
	for k := range sp {
		sizes[k+1] += sizes[k]
	}
	buf := make([]byte, sizes[len(sp)])
	data := unsafe.String(unsafe.SliceData(buf), len(buf))
	if int64(len(buf)) <= maxStr32 {
		offs := make([]int32, n+1)
		eachSpan(sp, func(k, lo, hi int) { layOut(strs, lo, hi, buf, offs, sizes[k]) })
		return &strCol[int32]{data: data, offs: offs, valid: valid}
	}
	offs := make([]int64, n+1)
	eachSpan(sp, func(k, lo, hi int) { layOut(strs, lo, hi, buf, offs, sizes[k]) })
	return &strCol[int64]{data: data, offs: offs, valid: valid}
}

// layOut copies strs[lo:hi] into buf from byte at, writing offs[lo+1:hi+1].
func layOut[P int32 | int64](strs []string, lo, hi int, buf []byte, offs []P, at int64) {
	p := at
	for k := lo; k < hi; k++ {
		p += int64(copy(buf[p:], strs[k]))
		offs[k+1] = P(p)
	}
}

// emptyStrings returns n empty strings, all present or all missing.
func emptyStrings(n int, missing bool) column {
	c := &strCol[int32]{offs: make([]int32, n+1)}
	if missing {
		c.valid = newBitmap(n)
	}
	return c
}

// spans returns up to GOMAXPROCS contiguous ranges covering [0, n), one
// when n is small.
func spans(n int) [][2]int {
	w := runtime.GOMAXPROCS(0)
	if n < parallelMin || w < 2 {
		return [][2]int{{0, n}}
	}
	w = min(w, n/(parallelMin/4))
	chunk := (n + w - 1) / w
	var out [][2]int
	for lo := 0; lo < n; lo += chunk {
		out = append(out, [2]int{lo, min(lo+chunk, n)})
	}
	return out
}

// eachSpan runs f on each span, concurrently when there are several.
func eachSpan(sp [][2]int, f func(k, lo, hi int)) {
	if len(sp) == 1 {
		f(0, sp[0][0], sp[0][1])
		return
	}
	var wg sync.WaitGroup
	for k, s := range sp {
		wg.Go(func() { f(k, s[0], s[1]) })
	}
	wg.Wait()
}

// gather lays out the values at idx (a negative position gives an empty,
// missing value) as a new column: the bytes counted, then copied, span by
// span.
func (c *strCol[O]) gather(idx []int) column {
	n := len(idx)
	sp := spans(n)
	sizes := make([]int64, len(sp))
	eachSpan(sp, func(k, lo, hi int) {
		t := int64(0)
		for _, i := range idx[lo:hi] {
			if i >= 0 {
				t += int64(c.offs[i+1] - c.offs[i])
			}
		}
		sizes[k] = t
	})
	starts := make([]int64, len(sp)+1)
	for k, s := range sizes {
		starts[k+1] = starts[k] + s
	}
	total := starts[len(sp)]
	buf := make([]byte, total)
	if total <= maxStr32 {
		offs := make([]int32, n+1)
		eachSpan(sp, func(k, lo, hi int) { gatherInto(c, idx, lo, hi, buf, offs, starts[k]) })
		return &strCol[int32]{data: unsafe.String(unsafe.SliceData(buf), len(buf)), offs: offs}
	}
	offs := make([]int64, n+1)
	eachSpan(sp, func(k, lo, hi int) { gatherInto(c, idx, lo, hi, buf, offs, starts[k]) })
	return &strCol[int64]{data: unsafe.String(unsafe.SliceData(buf), len(buf)), offs: offs}
}

// gatherInto copies the values at idx[lo:hi] into buf from byte at,
// writing offs[lo+1 : hi+1].
func gatherInto[O, P int32 | int64](c *strCol[O], idx []int, lo, hi int, buf []byte, offs []P, at int64) {
	p := at
	for k := lo; k < hi; k++ {
		if i := idx[k]; i >= 0 {
			p += int64(copy(buf[p:], c.data[c.offs[i]:c.offs[i+1]]))
		}
		offs[k+1] = P(p)
	}
}

func (c *strCol[O]) take(idx []int) column {
	out := c.gather(idx)
	return out.withValidity(c.valid.take(idx))
}

func (c *strCol[O]) takeNonNeg(idx []int) column {
	out := c.gather(idx)
	if c.valid == nil {
		return out
	}
	return out.withValidity(c.valid.take(idx))
}

func (c *strCol[O]) slice(i, j int) column {
	return &strCol[O]{data: c.data, offs: c.offs[i : j+1 : j+1], valid: c.valid.slice(i, j)}
}

func (c *strCol[O]) value(i int) any {
	if c.isNA(i) {
		return nil
	}
	return c.at(i)
}

func (c *strCol[O]) str(i int) string     { return c.at(i) }
func (c *strCol[O]) compare(i, j int) int { return strings.Compare(c.at(i), c.at(j)) }

func (c *strCol[O]) equal(i int, o column, j int) bool {
	oc, ok := o.(stringCol)
	return ok && c.at(i) == oc.at(j)
}

func (c *strCol[O]) ndarray() (*ndarray.Array, bool) { return nil, false }

func (c *strCol[O]) withValidity(b bitmap) column {
	out := *c
	out.valid = b
	return &out
}

func (c *strCol[O]) factorize() ([]int32, column) {
	d := &dict[string]{m: make(map[string]int32)}
	codes := make([]int32, c.len())
	c.codeRange(d, codes, 0, c.len())
	return codes, newStrCol(d.uniq, nil)
}

func (c *strCol[O]) factorizeWith(o column) ([]int32, []int32, column) {
	d := &dict[string]{m: make(map[string]int32)}
	a := codeStringCol(d, c)
	b := codeStringCol(d, o.(stringCol))
	return a, b, newStrCol(d.uniq, nil)
}

func (c *strCol[O]) appendCol(o column) column {
	oc := o.(stringCol)
	n, m := c.len(), oc.len()
	b := newStrBuilder(n+m, 0)
	for i := range n {
		b.add(c.at(i))
	}
	for i := range m {
		b.add(oc.at(i))
	}
	var valid bitmap
	if c.valid != nil || oc.validity() != nil {
		valid = allValid(n + m)
		for i := range n {
			if c.isNA(i) {
				valid.clear(i)
			}
		}
		for i := range m {
			if oc.isNA(i) {
				valid.clear(n + i)
			}
		}
	}
	return b.column(valid)
}

func (c *strCol[O]) nulls(n int) column { return emptyStrings(n, true) }

func (c *strCol[O]) convert(v any) (column, bool) {
	x, ok := v.(string)
	if !ok {
		return nil, false
	}
	return newStrCol([]string{x}, nil), true
}

func (c *strCol[O]) fill(v any) (column, error) {
	x, ok := v.(string)
	if !ok {
		return nil, fmt.Errorf("%w: cannot fill a %v column with %T", ErrDType, String, v)
	}
	n := c.len()
	b := newStrBuilder(n, len(c.data))
	for i := range n {
		if c.isNA(i) {
			b.add(x)
		} else {
			b.add(c.at(i))
		}
	}
	return b.column(nil), nil
}

func (c *strCol[O]) lookup() func(any) []int {
	m := make(map[string][]int, c.len())
	for i := range c.len() {
		if !c.isNA(i) {
			v := c.at(i)
			m[v] = append(m[v], i)
		}
	}
	return func(key any) []int {
		x, ok := key.(string)
		if !ok {
			return nil
		}
		return m[x]
	}
}

// cmpScalar compares every value with x, a missing one comparing false
// except with !=, as compareCols has it.
func (c *strCol[O]) cmpScalar(op cmpOp, x string, out []bool) {
	parallel(len(out), func(lo, hi int) {
		d, offs := c.data, c.offs
		switch op {
		case cmpEq, cmpNe:
			// Lengths first: most values differ from x in length already.
			eq, nx := op == cmpEq, O(len(x))
			for i := lo; i < hi; i++ {
				a, b := offs[i], offs[i+1]
				out[i] = (b-a == nx && d[a:b] == x) == eq
			}
		default:
			for i := lo; i < hi; i++ {
				out[i] = op.holds(strings.Compare(d[offs[i]:offs[i+1]], x))
			}
		}
		if c.valid != nil {
			for i := lo; i < hi; i++ {
				if !c.valid.get(i) {
					out[i] = op == cmpNe
				}
			}
		}
	})
}

// isInSet sets out[i] when value i is present and in m. A short value can
// only equal a short member: it is looked up by its packed key, in a slice
// when there are few such members.
func (c *strCol[O]) isInSet(m map[string]struct{}, out []bool) {
	var keys []uint64
	var short shortCodes
	for v := range m {
		if k, ok := shortKey(v); ok {
			keys = append(keys, k)
			short.put(k, 0)
		}
	}
	few := len(keys) <= 8
	// The few keys, padded with one no string packs to (length 8).
	var set [8]uint64
	for j := range set {
		set[j] = 8 << 56
	}
	copy(set[:], keys)
	parallel(len(out), func(lo, hi int) {
		d, offs := c.data, c.offs
		for i := lo; i < hi; i++ {
			if !c.valid.get(i) {
				continue
			}
			a, b := int(offs[i]), int(offs[i+1])
			if b-a > 7 {
				_, out[i] = m[d[a:b]]
				continue
			}
			var k uint64
			if a+8 <= len(d) {
				k = binary.LittleEndian.Uint64(unsafe.Slice(unsafe.StringData(d[a:]), 8))&(1<<(8*uint(b-a))-1) | uint64(b-a)<<56
			} else {
				k, _ = shortKey(d[a:b])
			}
			if few {
				out[i] = k == set[0] || k == set[1] || k == set[2] || k == set[3] ||
					k == set[4] || k == set[5] || k == set[6] || k == set[7]
			} else {
				_, out[i] = short.get(k)
			}
		}
	})
}

// shortKey packs a string of at most 7 bytes, with its length, in a uint64
// that no other string packs to; false for a longer one.
func shortKey(s string) (uint64, bool) {
	if len(s) > 7 {
		return 0, false
	}
	k := uint64(len(s)) << 56
	for i := range len(s) {
		k |= uint64(s[i]) << (8 * i)
	}
	return k, true
}

// codeRange codes rows [lo, hi) into codes (-1 for missing) in d. A value
// of up to 7 bytes is coded through a table keyed by its packed bytes,
// which costs no hashing of a string; a longer one through d's map.
func (c *strCol[O]) codeRange(d *dict[string], codes []int32, lo, hi int) {
	var short shortCodes
	for i := lo; i < hi; i++ {
		if !c.valid.get(i) {
			codes[i] = -1
			continue
		}
		a, b := int(c.offs[i]), int(c.offs[i+1])
		if b-a > 7 {
			codes[i] = d.add(c.data[a:b])
			continue
		}
		var k uint64
		if a+8 <= len(c.data) {
			k = binary.LittleEndian.Uint64(unsafe.Slice(unsafe.StringData(c.data[a:]), 8))&(1<<(8*uint(b-a))-1) | uint64(b-a)<<56
		} else {
			k, _ = shortKey(c.data[a:b])
		}
		if code, found := short.get(k); found {
			codes[i] = code
			continue
		}
		code := d.add(c.at(i))
		short.put(k, code)
		codes[i] = code
	}
}

// shortCodes is an open-addressing table from packed short strings (see
// shortKey; never 0, as the length byte of a non-empty one is not 0, and
// the empty string is stored apart) to codes.
type shortCodes struct {
	keys  []uint64
	codes []int32
	n     int
	shift uint  // 64 - log2(len(keys))
	empty int32 // the empty string's code + 1, 0 when not seen
}

func (t *shortCodes) slot(k uint64) int { return int((k * 0x9E3779B97F4A7C15) >> t.shift) }

func (t *shortCodes) get(k uint64) (int32, bool) {
	if k == 0 {
		return t.empty - 1, t.empty != 0
	}
	if t.keys == nil {
		return 0, false
	}
	mask := len(t.keys) - 1
	for h := t.slot(k); ; h = (h + 1) & mask {
		switch t.keys[h] {
		case k:
			return t.codes[h], true
		case 0:
			return 0, false
		}
	}
}

func (t *shortCodes) put(k uint64, code int32) {
	if k == 0 {
		t.empty = code + 1
		return
	}
	if 2*(t.n+1) > len(t.keys) {
		t.grow()
	}
	mask := len(t.keys) - 1
	h := t.slot(k)
	for t.keys[h] != 0 {
		h = (h + 1) & mask
	}
	t.keys[h], t.codes[h] = k, code
	t.n++
}

func (t *shortCodes) grow() {
	old, oldCodes := t.keys, t.codes
	t.keys = make([]uint64, max(64, 2*len(old)))
	t.codes = make([]int32, len(t.keys))
	t.shift = uint(64 - mbits.Len(uint(len(t.keys)-1)))
	t.n = 0
	for i, k := range old {
		if k != 0 {
			t.put(k, oldCodes[i])
		}
	}
}

// codeStringCol codes every row of src in d, in parallel chunks for a
// long column, as dict.code does for other columns.
func codeStringCol(d *dict[string], src stringCol) []int32 {
	n := src.len()
	codes := make([]int32, n)
	if n < 4*parallelMin {
		src.codeRange(d, codes, 0, n)
		return codes
	}
	sp := spans(n)
	locals := make([]*dict[string], len(sp))
	eachSpan(sp, func(k, lo, hi int) {
		locals[k] = &dict[string]{m: make(map[string]int32)}
		src.codeRange(locals[k], codes, lo, hi)
	})
	trans := make([][]int32, len(sp))
	for k, l := range locals {
		trans[k] = make([]int32, len(l.uniq))
		for j, v := range l.uniq {
			trans[k][j] = d.add(v)
		}
	}
	eachSpan(sp, func(k, lo, hi int) {
		t := trans[k]
		for i := lo; i < hi; i++ {
			if codes[i] >= 0 {
				codes[i] = t[codes[i]]
			}
		}
	})
	return codes
}
