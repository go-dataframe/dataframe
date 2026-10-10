package parquetio

import (
	"bytes"
	"fmt"
	"math"
	"testing"

	"github.com/go-dataframe/dataframe"
)

// benchFrame is the mixed frame of docs/perf.md (Arrow and Parquet section), a copy of the one in
// arrowio's benchmarks, built as ../arrowio/testdata/bench_pandas.py builds
// its pandas twin: float64 with 1% NaN, int64, string with 1% missing, a
// 10-level category, bool, and datetime.
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

func BenchmarkWriteParquet1e6(b *testing.B) {
	df := benchFrame(1_000_000)
	var buf bytes.Buffer
	b.ResetTimer()
	for range b.N {
		buf.Reset()
		if err := WriteParquet(&buf, df); err != nil {
			b.Fatal(err)
		}
	}
	b.ReportMetric(float64(buf.Len()), "bytes")
}

func BenchmarkReadParquet1e6(b *testing.B) {
	var buf bytes.Buffer
	if err := WriteParquet(&buf, benchFrame(1_000_000)); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for range b.N {
		if _, err := ReadParquet(bytes.NewReader(buf.Bytes())); err != nil {
			b.Fatal(err)
		}
	}
}
