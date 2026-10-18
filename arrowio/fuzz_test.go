package arrowio

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"testing"
)

// seeds adds every fixture with one of exts, and the frames of the round
// trips written both ways, to the corpus.
func seeds(f *testing.F, write func(io.Writer) error, exts ...string) {
	for _, ext := range exts {
		files, _ := filepath.Glob(filepath.Join("testdata", "*"+ext))
		for _, p := range files {
			if b, err := os.ReadFile(p); err == nil {
				f.Add(b)
			}
		}
	}
	var b bytes.Buffer
	if err := write(&b); err != nil {
		f.Fatal(err)
	}
	f.Add(b.Bytes())
}

// Reading any bytes as an IPC file gives a frame or an error, never a
// panic; a frame read writes back out.
func FuzzReadIPCFile(f *testing.F) {
	seeds(f, func(w io.Writer) error { return WriteIPCFile(w, benchFrame(50)) }, ".feather", ".arrow")
	f.Fuzz(func(t *testing.T, data []byte) {
		df, err := ReadIPCFile(bytes.NewReader(data))
		if err != nil {
			return
		}
		if err := WriteIPCFile(io.Discard, df); err != nil {
			t.Fatalf("a frame read does not write: %v", err)
		}
	})
}

// As FuzzReadIPCFile, for the stream format.
func FuzzReadIPCStream(f *testing.F) {
	seeds(f, func(w io.Writer) error { return WriteIPCStream(w, benchFrame(50)) }, ".arrows")
	f.Fuzz(func(t *testing.T, data []byte) {
		df, err := ReadIPCStream(bytes.NewReader(data))
		if err != nil {
			return
		}
		if err := WriteIPCStream(io.Discard, df); err != nil {
			t.Fatalf("a frame read does not write: %v", err)
		}
	})
}
