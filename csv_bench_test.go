package dataframe

import (
	"bytes"
	"fmt"
	"math/rand/v2"
	"os"
	"runtime"
	"strconv"
	"testing"
	"time"
)

// benchCSV returns n rows of the mixed table the cross-tool benchmark uses
// (bench/csv/bench.py writes the same shape): an int id, two floats, a short
// category string and a date.
func benchCSV(n int) []byte {
	r := rand.New(rand.NewPCG(1, 2))
	var b bytes.Buffer
	b.WriteString("id,x,y,cat,date\n")
	for i := range n {
		b.WriteString(strconv.Itoa(i))
		b.WriteByte(',')
		b.WriteString(strconv.FormatFloat(r.NormFloat64(), 'g', -1, 64))
		b.WriteByte(',')
		b.WriteString(strconv.FormatFloat(float64(r.IntN(1000000))/100, 'f', 2, 64))
		fmt.Fprintf(&b, ",cat%d,2020-%02d-%02d\n", r.IntN(50), 1+r.IntN(12), 1+r.IntN(28))
	}
	return b.Bytes()
}

func BenchmarkReadCSV(b *testing.B) {
	for _, n := range []int{1e4, 1e5, 1e6} {
		data := benchCSV(n)
		b.Run(strconv.Itoa(n), func(b *testing.B) {
			b.SetBytes(int64(len(data)))
			for b.Loop() {
				if _, err := readCSV(bytes.Clone(data), nil); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkWriteCSV(b *testing.B) {
	df, err := readCSV(benchCSV(1e5), []CSVOption{CSVParseDates("date")})
	if err != nil {
		b.Fatal(err)
	}
	var buf bytes.Buffer
	for b.Loop() {
		buf.Reset()
		if err := df.WriteCSV(&buf); err != nil {
			b.Fatal(err)
		}
	}
	b.SetBytes(int64(buf.Len()))
}

// TestCSVBenchFile is the Go side of bench/csv/bench.py: with
// CSV_BENCH_FILE set, it reads that file once (CSV_BENCH_DATES=1 parses the
// date column) and prints the time and the Go heap the frame holds.
func TestCSVBenchFile(t *testing.T) {
	path := os.Getenv("CSV_BENCH_FILE")
	if path == "" {
		t.Skip("CSV_BENCH_FILE not set")
	}
	var opts []CSVOption
	if os.Getenv("CSV_BENCH_DATES") == "1" {
		opts = append(opts, CSVParseDates("date"))
	}
	runtime.GC()
	start := time.Now()
	df, err := ReadCSVFile(path, opts...)
	elapsed := time.Since(start)
	if err != nil {
		t.Fatal(err)
	}
	runtime.GC()
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	fmt.Printf("csvbench seconds=%.4f rows=%d heap_live=%d total_alloc=%d\n",
		elapsed.Seconds(), df.Len(), m.HeapAlloc, m.TotalAlloc)
	runtime.KeepAlive(df)
}
