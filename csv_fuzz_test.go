package dataframe

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

// Reading any bytes as CSV gives a frame or an error, never a panic; a
// frame read writes back out, and what it writes reads again.
func FuzzReadCSV(f *testing.F) {
	files, _ := filepath.Glob(filepath.Join("testdata", "csv", "*.csv"))
	for _, p := range files {
		if b, err := os.ReadFile(p); err == nil {
			f.Add(b)
		}
	}
	f.Add(benchCSV(200))
	f.Fuzz(func(t *testing.T, data []byte) {
		df, err := ReadCSV(bytes.NewReader(data))
		if err != nil {
			return
		}
		var out bytes.Buffer
		if err := df.WriteCSV(&out); err != nil {
			t.Fatalf("a frame read does not write: %v", err)
		}
		if _, err := ReadCSV(bytes.NewReader(out.Bytes())); err != nil {
			t.Fatalf("what a frame writes does not read: %v\n%q", err, out.Bytes())
		}
	})
}
