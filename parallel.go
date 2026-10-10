package dataframe

import (
	"runtime"
	"sync"
	"sync/atomic"
)

// parallelMin is the length below which splitting work across goroutines
// costs more than it saves.
const parallelMin = 1 << 16

// parallel runs f over [0, n) in contiguous chunks, one per CPU when n is
// large enough to pay for it, and waits for them all.
func parallel(n int, f func(lo, hi int)) {
	w := runtime.GOMAXPROCS(0)
	if n < parallelMin || w < 2 {
		f(0, n)
		return
	}
	w = min(w, n/(parallelMin/4))
	var wg sync.WaitGroup
	chunk := (n + w - 1) / w
	for lo := 0; lo < n; lo += chunk {
		wg.Go(func() { f(lo, min(lo+chunk, n)) })
	}
	wg.Wait()
}

// eachColumn runs f on every column index concurrently when the frame is
// large, sequentially otherwise.
func eachColumn(rows, ncols int, f func(k int)) {
	if rows*ncols < parallelMin || ncols < 2 {
		for k := range ncols {
			f(k)
		}
		return
	}
	var wg sync.WaitGroup
	for k := range ncols {
		wg.Go(func() { f(k) })
	}
	wg.Wait()
}

// parallelEach runs f(0), ..., f(n-1) on up to GOMAXPROCS goroutines.
func parallelEach(n int, f func(i int)) {
	w := min(runtime.GOMAXPROCS(0), n)
	if w <= 1 {
		for i := range n {
			f(i)
		}
		return
	}
	var next atomic.Int64
	var wg sync.WaitGroup
	for range w {
		wg.Go(func() {
			for {
				i := int(next.Add(1) - 1)
				if i >= n {
					return
				}
				f(i)
			}
		})
	}
	wg.Wait()
}
