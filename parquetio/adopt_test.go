package parquetio

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/go-dataframe/dataframe"
)

// identical fails unless got and want agree on columns, dtypes, values,
// validity bitmaps, text, and the formatted value of every slot of every
// numeric array, missing ones included.
func identical(t *testing.T, where string, got, want *dataframe.DataFrame) {
	t.Helper()
	equalFrames(t, where, got, want)
	gs, ws := got.Series(), want.Series()
	for i := range gs {
		g, w := gs[i], ws[i]
		if !slices.Equal(g.ValidityBitmap(), w.ValidityBitmap()) {
			t.Errorf("%s:%s: validity %x, want %x", where, g.Name(), g.ValidityBitmap(), w.ValidityBitmap())
		}
		if !slices.Equal(g.Strings(), w.Strings()) {
			t.Errorf("%s:%s: text differs", where, g.Name())
		}
		ga, gerr := g.Values()
		wa, werr := w.Values()
		if (gerr == nil) != (werr == nil) {
			t.Fatalf("%s:%s: Values errors %v, %v", where, g.Name(), gerr, werr)
		}
		if gerr == nil && fmt.Sprintf("%#v", ga) != fmt.Sprintf("%#v", wa) {
			t.Errorf("%s:%s: storage differs", where, g.Name())
		}
	}
}

// copying is an allocator other than the Go allocator: a read with it
// copies every buffer, as every read did before buffers were handed over.
func copying() memory.Allocator { return memory.NewCheckedAllocator(memory.NewGoAllocator()) }

// Every fixture and round trip reads the same through the zero-copy path
// (the default Go allocator) as through the copying one.
func TestAdoptMatchesCopy(t *testing.T) {
	files, _ := filepath.Glob(filepath.Join(fixtures, "*.parquet"))
	for _, path := range files {
		if strings.HasPrefix(filepath.Base(path), "bad_") {
			continue
		}
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		got := dataframe.Must(ReadParquet(bytes.NewReader(data)))
		want := dataframe.Must(ReadParquet(bytes.NewReader(data), WithAllocator(copying())))
		identical(t, path, got, want)
	}
	for name, df := range map[string]*dataframe.DataFrame{
		"all types": allTypes(t),
		"indexed":   dataframe.Must(allTypes(t).SetIndex("i64", "s")),
		"bench":     benchFrame(3000),
	} {
		for _, rg := range []int64{1 << 20, 1000} {
			var b bytes.Buffer
			if err := WriteParquet(&b, df, WithRowGroupLength(rg)); err != nil {
				t.Fatal(err)
			}
			got := dataframe.Must(ReadParquet(bytes.NewReader(b.Bytes())))
			want := dataframe.Must(ReadParquet(bytes.NewReader(b.Bytes()), WithAllocator(copying())))
			identical(t, fmt.Sprintf("%s rows/group %d", name, rg), got, want)
			equalFrames(t, fmt.Sprintf("%s rows/group %d vs source", name, rg), got, df)
		}
	}
}
