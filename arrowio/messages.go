package arrowio

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"sync/atomic"

	"github.com/apache/arrow-go/v18/arrow/ipc"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/go-dataframe/dataframe/internal/arrowfb"
)

// The limits of arrow-go's own readers, kept here: a message's metadata
// and body.
const (
	maxMetadata = 64 << 20
	maxBody     = 256 << 20
)

// checkedMessages reads the messages of an IPC stream as arrow-go's
// message reader does, but checks each message's metadata flatbuffer
// before arrow-go reads it (see arrowfb), and reads a body as its bytes
// arrive rather than allocating its declared length first.
type checkedMessages struct {
	r    io.Reader
	refs atomic.Int64
	msg  *ipc.Message
}

func newCheckedMessages(r io.Reader) *checkedMessages {
	m := &checkedMessages{r: r}
	m.refs.Add(1)
	return m
}

func (m *checkedMessages) Retain() { m.refs.Add(1) }

func (m *checkedMessages) Release() {
	if m.refs.Add(-1) == 0 && m.msg != nil {
		m.msg.Release()
		m.msg = nil
	}
}

func (m *checkedMessages) Message() (*ipc.Message, error) {
	meta, err := readMetadata(m.r)
	if err != nil {
		return nil, err
	}
	n, err := arrowfb.CheckMessage(meta)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrCorrupt, err)
	}
	if n < 0 || n > maxBody {
		return nil, fmt.Errorf("%w: message body of %d bytes", ErrCorrupt, n)
	}
	var body bytes.Buffer
	if got, err := body.ReadFrom(io.LimitReader(m.r, n)); err != nil || got != n {
		return nil, fmt.Errorf("%w: message body of %d bytes, %d read: %v", ErrCorrupt, n, got, err)
	}
	if m.msg != nil {
		m.msg.Release()
	}
	m.msg = ipc.NewMessage(memory.NewBufferBytes(meta), memory.NewBufferBytes(body.Bytes()))
	return m.msg, nil
}

// readMetadata reads a message's framing and metadata: the continuation
// marker (or, in streams older than Arrow 0.15, none), the length, the
// flatbuffer. io.EOF marks the end of the stream.
func readMetadata(r io.Reader) ([]byte, error) {
	var h [4]byte
	if _, err := io.ReadFull(r, h[:]); err != nil {
		if errors.Is(err, io.EOF) {
			return nil, io.EOF
		}
		return nil, fmt.Errorf("%w: message framing: %v", ErrCorrupt, err)
	}
	n := int64(int32(binary.LittleEndian.Uint32(h[:])))
	if binary.LittleEndian.Uint32(h[:]) == 0xFFFFFFFF {
		if _, err := io.ReadFull(r, h[:]); err != nil {
			return nil, fmt.Errorf("%w: message length: %v", ErrCorrupt, err)
		}
		n = int64(int32(binary.LittleEndian.Uint32(h[:])))
	}
	if n == 0 {
		return nil, io.EOF
	}
	if n < 4 || n > maxMetadata {
		return nil, fmt.Errorf("%w: message metadata of %d bytes", ErrCorrupt, n)
	}
	var meta bytes.Buffer
	if got, err := meta.ReadFrom(io.LimitReader(r, n)); err != nil || got != n {
		return nil, fmt.Errorf("%w: message metadata of %d bytes, %d read", ErrCorrupt, n, got)
	}
	return meta.Bytes(), nil
}

// checkFile checks an IPC file's footer and the metadata of every message
// it lists, by their offsets in r, before arrow-go's file reader reads
// them.
func checkFile(r ReadAtSeeker) error {
	size, err := r.Seek(0, io.SeekEnd)
	if err != nil {
		return err
	}
	if _, err := r.Seek(0, io.SeekStart); err != nil {
		return err
	}
	const magic = "ARROW1"
	if size < int64(2*len(magic)+4+2) {
		return nil // too short to be a file: arrow-go says so
	}
	var tail [10]byte
	if _, err := r.ReadAt(tail[:], size-10); err != nil {
		return fmt.Errorf("%w: %v", ErrCorrupt, err)
	}
	n := int64(int32(binary.LittleEndian.Uint32(tail[:4])))
	if string(tail[4:]) != magic {
		return nil // arrow-go refuses it
	}
	if n <= 0 || n > size-10 || n > maxMetadata {
		return fmt.Errorf("%w: footer of %d bytes in a file of %d", ErrCorrupt, n, size)
	}
	footer := make([]byte, n)
	if _, err := r.ReadAt(footer, size-10-n); err != nil {
		return fmt.Errorf("%w: %v", ErrCorrupt, err)
	}
	if err := arrowfb.CheckFooter(footer); err != nil {
		return fmt.Errorf("%w: %v", ErrCorrupt, err)
	}
	for _, b := range arrowfb.FooterBlocks(footer) {
		if b.Offset < 0 || b.MetaDataLength < 8 || b.Offset > size-int64(b.MetaDataLength) || int64(b.MetaDataLength) > maxMetadata {
			return fmt.Errorf("%w: message block at %d of %d bytes", ErrCorrupt, b.Offset, b.MetaDataLength)
		}
		meta, err := readMetadata(io.NewSectionReader(r, b.Offset, int64(b.MetaDataLength)))
		if err != nil {
			return fmt.Errorf("%w: message at %d: %v", ErrCorrupt, b.Offset, err)
		}
		if _, err := arrowfb.CheckMessage(meta); err != nil {
			return fmt.Errorf("%w: message at %d: %v", ErrCorrupt, b.Offset, err)
		}
	}
	return nil
}
