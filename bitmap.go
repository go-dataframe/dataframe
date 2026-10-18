package dataframe

import (
	"math/bits"

	"github.com/go-simd/bitset"
)

// bitmap is a validity bitmap: bit i set means element i is present. The nil
// bitmap means every element is present, which is the common case and costs
// nothing.
type bitmap []uint64

func newBitmap(n int) bitmap { return make(bitmap, (n+63)>>6) }

// allValid returns a bitmap of n set bits.
func allValid(n int) bitmap {
	b := newBitmap(n)
	for i := range b {
		b[i] = ^uint64(0)
	}
	return b
}

func (b bitmap) get(i int) bool { return b == nil || b[i>>6]>>(uint(i)&63)&1 == 1 }

func (b bitmap) set(i int) { b[i>>6] |= 1 << (uint(i) & 63) }

func (b bitmap) clear(i int) { b[i>>6] &^= 1 << (uint(i) & 63) }

// count returns the number of set bits among the first n.
func (b bitmap) count(n int) int {
	if b == nil {
		return n
	}
	full := n >> 6
	c := bitset.Count(b[:full]) // go-simd's SIMD popcount
	if r := uint(n) & 63; r != 0 {
		c += bits.OnesCount64(b[full] & (1<<r - 1))
	}
	return c
}

// slice returns the bits [i, j) as a new bitmap, or nil when b is nil.
func (b bitmap) slice(i, j int) bitmap {
	if b == nil {
		return nil
	}
	out := newBitmap(j - i)
	for k := i; k < j; k++ {
		if b.get(k) {
			out.set(k - i)
		}
	}
	return out
}

// take gathers bits at the given positions; a negative position is a
// missing element. The result is nil when every gathered bit is set.
func (b bitmap) take(idx []int) bitmap {
	if b == nil && !hasNegative(idx) {
		return nil
	}
	var out bitmap
	for k, i := range idx {
		if i < 0 || !b.get(i) {
			if out == nil {
				out = allValid(len(idx))
			}
			out.clear(k)
		}
	}
	return out
}

// and returns the intersection of two validity bitmaps of length n.
func (b bitmap) and(o bitmap, n int) bitmap {
	if b == nil {
		return o.clone()
	}
	if o == nil {
		return b.clone()
	}
	out := newBitmap(n)
	for i := range out {
		out[i] = b[i] & o[i]
	}
	return out
}

func (b bitmap) clone() bitmap {
	if b == nil {
		return nil
	}
	return append(bitmap(nil), b...)
}

func hasNegative(idx []int) bool {
	for _, i := range idx {
		if i < 0 {
			return true
		}
	}
	return false
}
