//go:build cgo

package main

/*
#cgo CFLAGS: -I/opt/homebrew/include -I/usr/local/include
#cgo LDFLAGS: -L/opt/homebrew/lib -L/usr/local/lib -lsnappy
#include <stdlib.h>
#include <snappy-c.h>
*/
import "C"

import (
	"bytes"
	"fmt"
	"math"
	"time"
	"unsafe"

	"github.com/apache/arrow-go/v18/parquet/file"
	"github.com/go-dataframe/dataframe"
	"github.com/go-dataframe/dataframe/parquetio"
	gosnappy "github.com/golang/snappy"
	"github.com/klauspost/compress/s2"
	"github.com/klauspost/compress/snappy"
)

// snappyCompare compresses the pages of the Parquet bench frame (written
// uncompressed, so that each page's bytes are what a codec would get) with
// the C++ snappy library pyarrow uses (linked here through its C API) and
// with klauspost/compress's encoders, and prints each one's total size
// and time.
func snappyCompare(rounds int) {
	var buf bytes.Buffer
	if err := parquetio.WriteParquet(&buf, benchFrame(1_000_000), parquetio.WithCompression(parquetio.Uncompressed)); err != nil {
		fail("%v", err)
	}
	r, err := file.NewParquetReader(bytes.NewReader(buf.Bytes()))
	if err != nil {
		fail("%v", err)
	}
	var pages [][]byte
	for g := range r.NumRowGroups() {
		rg := r.RowGroup(g)
		for c := range rg.NumColumns() {
			pr, err := rg.GetColumnPageReader(c)
			if err != nil {
				fail("%v", err)
			}
			for pr.Next() {
				pages = append(pages, bytes.Clone(pr.Page().Data()))
			}
		}
	}
	total := 0
	for _, p := range pages {
		total += len(p)
	}
	encoders := []struct {
		name string
		f    func(dst, src []byte) []byte
	}{
		{"C++ snappy (libsnappy)", cxxSnappy},
		{"s2.EncodeSnappy", s2.EncodeSnappy},
		{"s2.EncodeSnappyBetter (arrow-go's)", snappy.Encode},
		{"s2.EncodeSnappyBest", s2.EncodeSnappyBest},
		{"golang/snappy (Go port of C++)", gosnappy.Encode},
	}
	fmt.Printf("%d pages, %d bytes uncompressed\n\n", len(pages), total)
	fmt.Println("| encoder | bytes | ratio to C++ | ms per pass |")
	fmt.Println("|---|---:|---:|---:|")
	var cxx int
	for _, e := range encoders {
		dst := make([]byte, s2.MaxEncodedLen(1<<24))
		best, size := math.Inf(1), 0
		for range rounds {
			t := time.Now()
			size = 0
			for _, p := range pages {
				out := e.f(dst, p)
				size += len(out)
				if d, err := s2.Decode(nil, out); err != nil || !bytes.Equal(d, p) {
					fail("%s: does not decode back", e.name)
				}
			}
			best = min(best, float64(time.Since(t).Microseconds())/1e3)
		}
		if cxx == 0 {
			cxx = size
		}
		fmt.Printf("| %s | %d | %.3f | %.1f |\n", e.name, size, float64(size)/float64(cxx), best)
	}
}

func cxxSnappy(dst, src []byte) []byte {
	n := C.size_t(len(dst))
	if C.snappy_compress((*C.char)(unsafe.Pointer(&src[0])), C.size_t(len(src)), (*C.char)(unsafe.Pointer(&dst[0])), &n) != C.SNAPPY_OK {
		fail("snappy_compress failed")
	}
	return dst[:n]
}

// benchFrame is the 10^6-row frame of parquetio's benchmarks.
func benchFrame(n int) *dataframe.DataFrame {
	f := make([]float64, n)
	k := make([]int64, n)
	s := make([]string, n)
	sv := make([]bool, n)
	codes := make([]int32, n)
	b := make([]bool, n)
	ts := make([]int64, n)
	for i := range n {
		f[i] = float64(i) * 0.5
		if i%100 == 0 {
			f[i] = math.NaN()
		}
		k[i] = int64(i*7) % 1000003
		s[i] = fmt.Sprintf("s%d", i%1000)
		sv[i] = i%100 != 1
		codes[i] = int32(i % 10)
		b[i] = i%3 == 0
		ts[i] = 1_600_000_000e9 + int64(i)*1e9
	}
	cats := []string{"c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9"}
	return dataframe.Must(dataframe.FromSeries(
		dataframe.Must(dataframe.NewSeries("f", f)),
		dataframe.Must(dataframe.NewSeries("k", k)),
		dataframe.Must(dataframe.NewSeriesMasked("s", s, sv)),
		dataframe.Must(dataframe.NewCategorical("c", codes, cats)),
		dataframe.Must(dataframe.NewSeries("b", b)),
		dataframe.Must(dataframe.ToDatetime(dataframe.Must(dataframe.NewSeries("t", ts)))),
	))
}
