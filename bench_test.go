package dataframe

import (
	"bytes"
	"os"
	"runtime"
	"strconv"
	"testing"
	"time"
)

// The benchmarks below run on the same synthetic table as
// bench/bench_pandas.py builds, value for value, so that the two can be
// compared. Set DATAFRAME_BENCH_N (default 1,000,000 rows).

func benchN() int {
	if n, err := strconv.Atoi(os.Getenv("DATAFRAME_BENCH_N")); err == nil {
		return n
	}
	return 1_000_000
}

// benchFrame builds: key int64 in [0, 1000), x and y float64 in [0, 1),
// cat a string among 100.
func benchFrame(n int) *DataFrame {
	key := makeCol[int64](Int64, n)
	x := makeCol[float64](Float64, n)
	y := makeCol[float64](Float64, n)
	cats := make([]string, 100)
	for i := range cats {
		cats[i] = "c" + strconv.Itoa(i)
	}
	cat := makeCol[string](String, n)
	for i := range n {
		key.data[i] = int64(i*7919) % 1000
		x.data[i] = float64(uint64(i)*2654435761%1000003) / 1000003
		y.data[i] = float64(uint64(i)*40503%999983) / 999983
		cat.data[i] = cats[(i*31)%100]
	}
	return &DataFrame{index: NewRangeIndex(n), names: []string{"key", "x", "y", "cat"},
		cols: []column{key, x, y, cat}}
}

func benchRight() *DataFrame {
	key := makeCol[int64](Int64, 1000)
	v := makeCol[float64](Float64, 1000)
	for i := range 1000 {
		key.data[i] = int64(i)
		v.data[i] = float64(i) / 10
	}
	return &DataFrame{index: NewRangeIndex(1000), names: []string{"key", "v"}, cols: []column{key, v}}
}

func BenchmarkGroupBySum(b *testing.B) {
	df := benchFrame(benchN())
	b.ResetTimer()
	for b.Loop() {
		g, _ := df.GroupBy("key")
		g.Sum()
	}
}

func BenchmarkGroupBySumString(b *testing.B) {
	df := benchFrame(benchN())
	b.ResetTimer()
	for b.Loop() {
		g, _ := df.GroupBy("cat")
		g.Sum()
	}
}

func BenchmarkSortValues(b *testing.B) {
	df := benchFrame(benchN())
	b.ResetTimer()
	for b.Loop() {
		df.SortValues([]string{"x"})
	}
}

func BenchmarkFilter(b *testing.B) {
	df := benchFrame(benchN())
	x, _ := df.Column("x")
	b.ResetTimer()
	for b.Loop() {
		m, _ := x.Gt(0.5)
		df.Filter(m)
	}
}

func BenchmarkMerge(b *testing.B) {
	df := benchFrame(benchN())
	r := benchRight()
	b.ResetTimer()
	for b.Loop() {
		Merge(df, r, MergeOptions{On: []string{"key"}})
	}
}

// benchDates returns n instants of 2020 as "YYYY-MM-DD HH:MM:SS" text, as
// bench_pandas.py builds them: 1577836800 + (i*7919 mod 31536000) seconds.
func benchDates(n int) *Series {
	d := make([]string, n)
	for i := range n {
		sec := 1577836800 + int64(i)*7919%31536000
		d[i] = time.Unix(sec, 0).UTC().Format(time.DateTime)
	}
	s, _ := NewSeries("d", d)
	return s
}

func BenchmarkToDatetime(b *testing.B) {
	s := benchDates(benchN())
	b.ResetTimer()
	for b.Loop() {
		if _, err := ToDatetime(s); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkToDatetimeFormat(b *testing.B) {
	s := benchDates(benchN())
	b.ResetTimer()
	for b.Loop() {
		if _, err := ToDatetime(s, "%Y-%m-%d %H:%M:%S"); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkWriteCSVTable writes the operations table as CSV, index
// included, as pandas' df.to_csv() does.
func BenchmarkWriteCSVTable(b *testing.B) {
	df := benchFrame(benchN())
	var buf bytes.Buffer
	b.ResetTimer()
	for b.Loop() {
		buf.Reset()
		if err := df.WriteCSV(&buf); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkStringEq(b *testing.B) {
	cat, _ := benchFrame(benchN()).Column("cat")
	b.ResetTimer()
	for b.Loop() {
		cat.Eq("c7")
	}
}

func BenchmarkStringIsIn(b *testing.B) {
	cat, _ := benchFrame(benchN()).Column("cat")
	b.ResetTimer()
	for b.Loop() {
		cat.IsIn("c1", "c22", "c57")
	}
}

func BenchmarkMemory(b *testing.B) {
	n := benchN()
	var before, after runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&before)
	df := benchFrame(n)
	runtime.GC()
	runtime.ReadMemStats(&after)
	runtime.KeepAlive(df)
	for b.Loop() {
	}
	b.ReportMetric(float64(after.HeapAlloc-before.HeapAlloc)/1e6, "MB")
}
