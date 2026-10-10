package dataframe

import (
	"bytes"
	"fmt"
	"math"
	"runtime"
	"unsafe"
)

// The tokenizer turns the input into field boundaries without copying any
// field. The input is cut into chunks at line ends, which are tokenized in
// parallel; a chunk that turns out to start inside a quoted field (the
// previous chunk's last record ran past its end) is discarded and tokenized
// again from the true record start. Each record stores its start and the end
// of each of its fields, as offsets from the chunk's start; a field starts
// one byte after the previous one ends.

// padField marks a field a short record lacks.
const padField = ^uint32(0)

// Tunables, variables so that tests can exercise the parallel paths and the
// offset limit on small inputs.
var (
	// csvMinChunk is the smallest chunk worth a goroutine.
	csvMinChunk = 1 << 20
	// csvMaxOffset is the largest field offset a chunk can store.
	csvMaxOffset = int(min(math.MaxUint32-1, math.MaxInt))
)

// chunkOut is what tokenizing one stretch of input yields.
type chunkOut struct {
	free      bool // records keep their own field count (counts); else ncols each
	keepBlank bool // blank and comment lines are records too
	ncols     int
	base      int // offsets are relative to it
	next      int // where tokenizing stopped
	rec       []uint32
	ends      []uint32
	counts    []int
	nrec      int
	lines     int // lines consumed, as pandas counts them
	err       *csvErr
}

// csvErr is a tokenizing error: a record with too many fields (saw > 0), an
// unterminated quote (saw == 0), or a record too long to index (saw < 0).
type csvErr struct {
	line          int // lines of the chunk before the record
	expected, saw int
}

type tokenizer struct {
	cfg  *csvConfig
	data []byte
	stop [256]bool // bytes ending an unquoted field
}

func newTokenizer(cfg *csvConfig, data []byte) *tokenizer {
	t := &tokenizer{cfg: cfg, data: data}
	t.stop[cfg.sep] = true
	t.stop['\n'] = true
	t.stop['\r'] = true
	if cfg.hasComment {
		t.stop[cfg.comment] = true
	}
	return t
}

// fail turns a tokenizing error into the error ReadCSV returns; lines is the
// number of lines before the chunk.
func (t *tokenizer) fail(e *csvErr, lines int) error {
	line := lines + e.line + 1
	switch {
	case e.saw > 0:
		return fmt.Errorf("%w: expected %d fields in line %d, saw %d", ErrParse, e.expected, line, e.saw)
	case e.saw == 0:
		return fmt.Errorf("%w: EOF inside string starting at line %d", ErrParse, line)
	}
	return fmt.Errorf("%w: record at line %d is too long", ErrParse, line)
}

// eol returns the position after the line end at p (LF, CRLF or CR).
func (t *tokenizer) eol(p int) int {
	if t.data[p] == '\r' && p+1 < len(t.data) && t.data[p+1] == '\n' {
		return p + 2
	}
	return p + 1
}

// skipLine returns the position after the end of the line holding p.
func (t *tokenizer) skipLine(p int) int {
	d := t.data
	for p < len(d) && d[p] != '\n' && d[p] != '\r' {
		p++
	}
	if p == len(d) {
		return p
	}
	return t.eol(p)
}

// blankLine reports whether the line at p holds only spaces and tabs (that
// are not the separator), and returns the position after it.
func (t *tokenizer) blankLine(p int) (int, bool) {
	d := t.data
	for p < len(d) && (d[p] == ' ' || d[p] == '\t') && d[p] != t.cfg.sep {
		p++
	}
	if p == len(d) {
		return p, true
	}
	if d[p] == '\n' || d[p] == '\r' {
		return t.eol(p), true
	}
	return p, false
}

// parse tokenizes the records starting before end, from pos, the last one
// running to its own end even past end; at most maxRec records when maxRec
// is not negative. It returns where it stopped.
func (t *tokenizer) parse(o *chunkOut, pos, end, maxRec int) int {
	d := t.data
	n := len(d)
	cfg := t.cfg
	sep, quote := cfg.sep, cfg.quote
	for pos < end && (maxRec < 0 || o.nrec < maxRec) {
		if !o.keepBlank {
			if cfg.hasComment && d[pos] == cfg.comment {
				pos = t.skipLine(pos)
				o.lines++
				continue
			}
			if cfg.skipBlank {
				if p, ok := t.blankLine(pos); ok {
					pos = p
					o.lines++
					continue
				}
			}
		}
		start := pos
		nf := 0
		for {
			if cfg.hasQuote && pos < n && d[pos] == quote {
				pos++
				for {
					i := bytes.IndexByte(d[pos:], quote)
					if i < 0 {
						o.err = &csvErr{line: o.lines}
						o.next = n
						return n
					}
					pos += i + 1
					if pos < n && d[pos] == quote {
						pos++
						continue
					}
					break
				}
			}
			for pos < n && !t.stop[d[pos]] {
				pos++
			}
			nf++
			if o.free || nf <= o.ncols {
				o.ends = append(o.ends, uint32(pos-o.base))
			}
			if pos == n {
				break
			}
			c := d[pos]
			if c == sep {
				pos++
				continue
			}
			if c == '\r' || c == '\n' {
				pos = t.eol(pos)
			} else {
				pos = t.skipLine(pos) // a comment
			}
			break
		}
		if pos-o.base > csvMaxOffset {
			o.err = &csvErr{line: o.lines, saw: -1}
			o.next = pos
			return pos
		}
		if o.free {
			o.counts = append(o.counts, nf)
		} else if nf > o.ncols {
			o.err = &csvErr{line: o.lines, expected: o.ncols, saw: nf}
			o.next = pos
			return pos
		} else {
			for ; nf < o.ncols; nf++ {
				o.ends = append(o.ends, padField)
			}
		}
		if o.rec == nil && !o.free {
			// Size the chunk's offsets from its first record, plus a
			// margin, rather than grow them by copying.
			est := (end-start)/max(pos-start, 1)*9/8 + 16
			o.rec = make([]uint32, 0, est)
			o.ends = append(make([]uint32, 0, est*o.ncols), o.ends...)
		}
		o.rec = append(o.rec, uint32(start-o.base))
		o.nrec++
		o.lines++
	}
	o.next = pos
	return pos
}

// span is a stretch of input [start, end).
type span struct{ start, end int }

// parseAll tokenizes the data records from pos with ncols fields each, at
// most nrows when nrows is not negative. lines counts the lines before pos.
func (t *tokenizer) parseAll(pos, ncols, nrows, lines int) ([]*chunkOut, error) {
	n := len(t.data)
	workers := runtime.GOMAXPROCS(0)
	size := max(csvMinChunk, (n-pos)/(4*workers))
	if nrows >= 0 || n-pos <= size {
		o := &chunkOut{ncols: ncols, base: pos}
		t.parse(o, pos, n, nrows)
		if o.err != nil {
			return nil, t.fail(o.err, lines)
		}
		return []*chunkOut{o}, nil
	}
	var spans []span
	for b := pos; b < n; {
		e := b + size
		if e >= n {
			e = n
		} else if i := bytes.IndexByte(t.data[e:], '\n'); i < 0 {
			e = n
		} else {
			e += i + 1
		}
		spans = append(spans, span{b, e})
		b = e
	}
	outs := make([]*chunkOut, len(spans))
	parallelEach(len(spans), func(k int) {
		s := spans[k]
		o := &chunkOut{ncols: ncols, base: s.start}
		t.parse(o, s.start, s.end, -1)
		outs[k] = o
	})
	var groups []*chunkOut
	for k, s := range spans {
		if pos >= s.end {
			continue // the previous record ran over this whole chunk
		}
		o := outs[k]
		if pos != s.start {
			// This chunk started inside a record: tokenize it again from
			// where that record ends.
			o = &chunkOut{ncols: ncols, base: pos}
			t.parse(o, pos, s.end, -1)
		}
		if o.err != nil {
			return nil, t.fail(o.err, lines)
		}
		groups = append(groups, o)
		lines += o.lines
		pos = o.next
	}
	return groups, nil
}

// text returns the field [s, e) of a chunk, its quotes removed. An unquoted
// field, and a quoted one without doubled quotes, is a view of the input.
func (t *tokenizer) text(o *chunkOut, s, e uint32) string {
	if s == e {
		return ""
	}
	b := t.data[o.base+int(s) : o.base+int(e)]
	if t.cfg.hasQuote && b[0] == t.cfg.quote {
		return t.unquote(b)
	}
	return unsafe.String(&b[0], len(b))
}

// unquote returns a quoted field's content: the text up to the closing
// quote, doubled quotes made single, then whatever follows the closing
// quote, as pandas reads `"ab"c` as `abc`.
func (t *tokenizer) unquote(b []byte) string {
	q := t.cfg.quote
	in := b[1:]
	if i := bytes.IndexByte(in, q); i == len(in)-1 {
		if i == 0 {
			return ""
		}
		return unsafe.String(&in[0], i)
	}
	out := make([]byte, 0, len(in))
	for len(in) > 0 {
		i := bytes.IndexByte(in, q)
		out = append(out, in[:i]...)
		if i+1 < len(in) && in[i+1] == q {
			out = append(out, q)
			in = in[i+2:]
			continue
		}
		out = append(out, in[i+1:]...)
		break
	}
	return string(out)
}

// raw returns field j of the first record of a free chunk.
func (t *tokenizer) raw(o *chunkOut, j int) string {
	s := o.rec[0]
	if j > 0 {
		s = o.ends[j-1] + 1
	}
	return t.text(o, s, o.ends[j])
}

// field returns field j of record r of a fixed chunk, and false when the
// record lacks it.
func (t *tokenizer) field(o *chunkOut, r, j int) (string, bool) {
	k := r*o.ncols + j
	e := o.ends[k]
	if e == padField {
		return "", false
	}
	s := o.rec[r]
	if j > 0 {
		s = o.ends[k-1] + 1
	}
	return t.text(o, s, e), true
}
