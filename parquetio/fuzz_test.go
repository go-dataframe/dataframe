package parquetio

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"testing"
)

// Reading any bytes as Parquet gives a frame or an error, never a panic; a
// frame read writes back out.
func FuzzReadParquet(f *testing.F) {
	files, _ := filepath.Glob(filepath.Join(fixtures, "*.parquet"))
	for _, p := range files {
		if b, err := os.ReadFile(p); err == nil {
			f.Add(b)
		}
	}
	var b bytes.Buffer
	if err := WriteParquet(&b, benchFrame(50)); err != nil {
		f.Fatal(err)
	}
	f.Add(b.Bytes())
	f.Fuzz(func(t *testing.T, data []byte) {
		df, err := ReadParquet(bytes.NewReader(data))
		if err != nil {
			return
		}
		if err := WriteParquet(io.Discard, df); err != nil {
			t.Fatalf("a frame read does not write: %v", err)
		}
	})
}
