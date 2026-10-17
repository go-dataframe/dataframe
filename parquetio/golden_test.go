package parquetio

import (
	"bytes"
	"crypto/sha256"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/go-dataframe/dataframe"
	"github.com/go-dataframe/dataframe/arrowio"
)

// goldenFrames are small and large frames of every dtype, with strings that
// fit a dictionary and strings that do not.
func goldenFrames(t *testing.T) map[string]*dataframe.DataFrame {
	n := 300_000
	hi, lo := make([]string, n), make([]string, n)
	ints, small := make([]int64, n), make([]int64, n)
	fl, f32, i32 := make([]float64, n), make([]float32, n), make([]int32, n)
	valid := make([]bool, n)
	for i := range n {
		hi[i] = fmt.Sprintf("unique-%d", i)
		lo[i] = fmt.Sprintf("v%d", i%37)
		ints[i] = int64(i) * 2654435761 % 1000003
		small[i] = int64(i % 1000)
		fl[i] = float64(i) / 7
		if i%13 == 0 {
			fl[i] = math.NaN()
		}
		if i%17 == 0 {
			fl[i] = math.Copysign(0, -1)
		}
		f32[i] = float32(i % 300000)
		i32[i] = int32(i)
		valid[i] = i%11 != 0
	}
	mixed := dataframe.Must(dataframe.FromSeries(
		dataframe.Must(dataframe.NewSeries("hi", hi)),
		dataframe.Must(dataframe.NewSeriesMasked("lo", lo, valid)),
		dataframe.Must(dataframe.NewSeries("ints", ints)),
		dataframe.Must(dataframe.NewSeriesMasked("small", small, valid)),
		dataframe.Must(dataframe.NewSeries("fl", fl)),
		dataframe.Must(dataframe.NewSeries("f32", f32)),
		dataframe.Must(dataframe.NewSeriesMasked("i32", i32, valid)),
	))
	return map[string]*dataframe.DataFrame{
		"all": allTypes(t), "indexed": dataframe.Must(allTypes(t).SetIndex("i64", "s")),
		"bench": benchFrame(200_000), "mixed": mixed, "empty": allTypes(t).Head(0),
	}
}

// TestGoldenFiles writes the same frames in every format and codec, and
// the SHA-256 of each file, into DATAFRAME_GOLDEN_DIR (it does nothing
// without it), so that two builds can be compared file by file:
// bench/harness golden compares two such directories.
func TestGoldenFiles(t *testing.T) {
	out := os.Getenv("DATAFRAME_GOLDEN_DIR")
	if out == "" {
		t.Skip()
	}
	var lines []string
	for name, df := range goldenFrames(t) {
		write := map[string]func(*bytes.Buffer) error{
			"ipc":        func(b *bytes.Buffer) error { return arrowio.WriteIPCFile(b, df) },
			"ipc-lz4":    func(b *bytes.Buffer) error { return arrowio.WriteIPCFile(b, df, arrowio.WithCompression(arrowio.LZ4)) },
			"ipc-zstd":   func(b *bytes.Buffer) error { return arrowio.WriteIPCFile(b, df, arrowio.WithCompression(arrowio.Zstd)) },
			"stream":     func(b *bytes.Buffer) error { return arrowio.WriteIPCStream(b, df) },
			"pq":         func(b *bytes.Buffer) error { return WriteParquet(b, df) },
			"pq-none":    func(b *bytes.Buffer) error { return WriteParquet(b, df, WithCompression(Uncompressed)) },
			"pq-zstd":    func(b *bytes.Buffer) error { return WriteParquet(b, df, WithCompression(Zstd)) },
			"pq-rg50000": func(b *bytes.Buffer) error { return WriteParquet(b, df, WithRowGroupLength(50000)) },
		}
		for kind, f := range write {
			var b bytes.Buffer
			if err := f(&b); err != nil {
				t.Fatal(err)
			}
			file := name + "." + kind
			if err := os.WriteFile(filepath.Join(out, file), b.Bytes(), 0o644); err != nil {
				t.Fatal(err)
			}
			lines = append(lines, fmt.Sprintf("%x %d %s", sha256.Sum256(b.Bytes()), b.Len(), file))
		}
	}
	sort.Strings(lines)
	if err := os.WriteFile(filepath.Join(out, "SHA256SUMS"), []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
}
