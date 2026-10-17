package dataframe

import (
	"bufio"
	"errors"
	"io"
	"math"
	"os"
	"strconv"
)

// CSVWriteOption configures WriteCSV, mirroring pandas' to_csv parameters.
type CSVWriteOption func(*csvWriteConfig)

type csvWriteConfig struct {
	index   bool
	header  bool
	sep     byte
	naRep   string
	lineEnd string
}

// CSVWriteIndex says whether the index is written as the first columns,
// true by default (pandas' index).
func CSVWriteIndex(write bool) CSVWriteOption { return func(o *csvWriteConfig) { o.index = write } }

// CSVWriteHeader says whether the column names are written, true by default
// (pandas' header).
func CSVWriteHeader(write bool) CSVWriteOption { return func(o *csvWriteConfig) { o.header = write } }

// CSVWriteSeparator sets the field separator, ',' by default (pandas' sep).
func CSVWriteSeparator(c byte) CSVWriteOption { return func(o *csvWriteConfig) { o.sep = c } }

// CSVWriteNARep sets what a missing value is written as, "" by default
// (pandas' na_rep).
func CSVWriteNARep(s string) CSVWriteOption { return func(o *csvWriteConfig) { o.naRep = s } }

// CSVWriteLineTerminator sets the line end, "\n" by default (pandas'
// lineterminator, whose default is the platform's).
func CSVWriteLineTerminator(s string) CSVWriteOption {
	return func(o *csvWriteConfig) { o.lineEnd = s }
}

// csvWriteBlock is the number of rows formatted by one goroutine at a time.
var csvWriteBlock = 1 << 14

// WriteCSV writes the frame as CSV, as pandas' to_csv does by default: a
// header row, the index as the first column(s) (headed by its name, empty
// when unnamed), missing values as empty fields, floats as Python prints
// them ("1.0", "0.1", "1e-05", "1e+16"), bools as True and False, instants
// as dates when all are at midnight and otherwise with the fraction of a
// second the column needs, and quotes only around fields holding the
// separator, a quote or a line break (RFC 4180).
func (df *DataFrame) WriteCSV(w io.Writer, opts ...CSVWriteOption) error {
	cfg := &csvWriteConfig{index: true, header: true, sep: ',', lineEnd: "\n"}
	for _, o := range opts {
		o(cfg)
	}
	var cells []cellFunc
	var head []string
	if cfg.index {
		for k, name := range df.index.names {
			head = append(head, name)
			cells = append(cells, csvIndexCells(df.index, k, cfg.naRep))
		}
	}
	for k, c := range df.cols {
		head = append(head, df.names[k])
		cells = append(cells, csvCells(c, cfg.naRep))
	}
	wr := &csvWriter{cfg: cfg}
	bw := bufio.NewWriterSize(w, 1<<16)
	if cfg.header {
		var line []byte
		for k, h := range head {
			line = wr.sep(line, k)
			line = wr.check(append(line, h...), len(line))
		}
		bw.Write(wr.end(line, len(head)))
	}
	n := df.Len()
	batch := csvWriteBlock * 8
	bufs := make([][]byte, 8)
	for lo := 0; lo < n; lo += batch {
		hi := min(lo+batch, n)
		nb := (hi - lo + csvWriteBlock - 1) / csvWriteBlock
		parallelEach(nb, func(b int) {
			buf := bufs[b][:0]
			for i := lo + b*csvWriteBlock; i < min(lo+(b+1)*csvWriteBlock, hi); i++ {
				start := len(buf)
				for k, cell := range cells {
					buf = wr.sep(buf, k)
					buf = wr.check(cell(buf, i), len(buf))
				}
				buf = wr.endAt(buf, start, len(cells))
			}
			bufs[b] = buf
		})
		for _, b := range bufs[:nb] {
			bw.Write(b)
		}
	}
	return bw.Flush()
}

// WriteCSVFile writes the frame to a file as WriteCSV writes it.
func (df *DataFrame) WriteCSVFile(path string, opts ...CSVWriteOption) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	err = df.WriteCSV(f, opts...)
	return errors.Join(err, f.Close())
}

// cellFunc appends the text of row i of a column.
type cellFunc func(dst []byte, i int) []byte

type csvWriter struct{ cfg *csvWriteConfig }

// sep appends the separator before field k of a line.
func (w *csvWriter) sep(dst []byte, k int) []byte {
	if k > 0 {
		dst = append(dst, w.cfg.sep)
	}
	return dst
}

// check quotes the field appended from start when it needs quotes.
func (w *csvWriter) check(dst []byte, start int) []byte {
	for _, c := range dst[start:] {
		if c == w.cfg.sep || c == '"' || c == '\n' || c == '\r' {
			return w.quote(dst, start)
		}
	}
	return dst
}

// quote rewrites the field from start quoted, its quotes doubled.
func (w *csvWriter) quote(dst []byte, start int) []byte {
	text := string(dst[start:])
	dst = append(dst[:start], '"')
	for k := range len(text) {
		if text[k] == '"' {
			dst = append(dst, '"')
		}
		dst = append(dst, text[k])
	}
	return append(dst, '"')
}

// end ends a line of nf fields starting the buffer.
func (w *csvWriter) end(line []byte, nf int) []byte { return w.endAt(line, 0, nf) }

// endAt ends the line that starts at start. A line of one empty field is
// written "" so that it does not read as a blank line, as Python's csv
// module writes it.
func (w *csvWriter) endAt(dst []byte, start, nf int) []byte {
	if nf == 1 && len(dst) == start {
		dst = append(dst, '"', '"')
	}
	return append(dst, w.cfg.lineEnd...)
}

// indexCells formats level k of an index.
func csvIndexCells(ix *Index, k int, naRep string) cellFunc {
	if ix.IsRange() {
		return func(dst []byte, i int) []byte {
			return strconv.AppendInt(dst, int64(ix.start+i*ix.step), 10)
		}
	}
	return csvCells(ix.levels[k], naRep)
}

// csvCells formats a column's elements.
func csvCells(c column, naRep string) cellFunc {
	var f cellFunc
	switch p := c.(type) {
	case *col[float64]:
		f = func(dst []byte, i int) []byte { return appendFloatRepr(dst, p.data[i], 64) }
	case *col[float32]:
		f = func(dst []byte, i int) []byte { return appendFloatRepr(dst, float64(p.data[i]), 32) }
	case *col[int64]:
		if p.dt == Datetime {
			f = datetimeCells(p)
		} else if p.dt == Int64 {
			f = func(dst []byte, i int) []byte { return strconv.AppendInt(dst, p.data[i], 10) }
		}
	case *col[uint64]:
		f = func(dst []byte, i int) []byte { return strconv.AppendUint(dst, p.data[i], 10) }
	case stringCol:
		f = func(dst []byte, i int) []byte { return append(dst, p.at(i)...) }
	}
	if f == nil {
		f = func(dst []byte, i int) []byte { return append(dst, c.str(i)...) }
	}
	return func(dst []byte, i int) []byte {
		if c.isNA(i) {
			return append(dst, naRep...)
		}
		return f(dst, i)
	}
}

// datetimeCells formats instants as pandas writes them: dates when every
// one is at midnight, else date and time with as many fractional digits (0,
// 3, 6 or 9) as the column needs.
func datetimeCells(c *col[int64]) cellFunc {
	dateOnly, digits := true, 0
	for i, ns := range c.data {
		if c.isNA(i) {
			continue
		}
		rem := ns - floorDiv(ns, 86400e9)*86400e9
		if rem != 0 {
			dateOnly = false
		}
		switch frac := rem % 1e9; {
		case frac%1e3 != 0:
			digits = 9
		case frac%1e6 != 0:
			digits = max(digits, 6)
		case frac != 0:
			digits = max(digits, 3)
		}
	}
	return func(dst []byte, i int) []byte { return appendDatetime(dst, c.data[i], dateOnly, digits) }
}

// appendPadded appends v, not negative, in at least width digits.
func appendPadded(dst []byte, v int64, width int) []byte {
	var tmp [20]byte
	s := strconv.AppendInt(tmp[:0], v, 10)
	for k := len(s); k < width; k++ {
		dst = append(dst, '0')
	}
	return append(dst, s...)
}

func appendDatetime(dst []byte, ns int64, dateOnly bool, digits int) []byte {
	days := floorDiv(ns, 86400e9)
	rem := ns - days*86400e9
	y, m, d := civilFromDays(days)
	dst = appendPadded(dst, int64(y), 4)
	dst = append(dst, '-')
	dst = appendPadded(dst, int64(m), 2)
	dst = append(dst, '-')
	dst = appendPadded(dst, int64(d), 2)
	if dateOnly {
		return dst
	}
	sec := rem / 1e9
	dst = append(dst, ' ')
	dst = appendPadded(dst, sec/3600, 2)
	dst = append(dst, ':')
	dst = appendPadded(dst, sec/60%60, 2)
	dst = append(dst, ':')
	dst = appendPadded(dst, sec%60, 2)
	if digits > 0 {
		dst = append(dst, '.')
		dst = appendPadded(dst, rem%1e9/int64(math.Pow10(9-digits)), digits)
	}
	return dst
}

// appendFloatRepr appends a float as Python's repr writes it: the shortest
// digits that read back, positional with at least one decimal when the
// decimal point falls in (-4, 16], else scientific with a signed exponent of
// at least two digits.
func appendFloatRepr(dst []byte, f float64, bits int) []byte {
	switch {
	case math.IsInf(f, 1):
		return append(dst, "inf"...)
	case math.IsInf(f, -1):
		return append(dst, "-inf"...)
	case f == math.Trunc(f) && math.Abs(f) < 1e15 && !(f == 0 && math.Signbit(f)):
		dst = strconv.AppendInt(dst, int64(f), 10)
		return append(dst, '.', '0')
	}
	var tmp [32]byte
	e := strconv.AppendFloat(tmp[:0], f, 'e', -1, bits)
	if e[0] == '-' {
		dst = append(dst, '-')
		e = e[1:]
	}
	k := 0
	for e[k] != 'e' {
		k++
	}
	exp, _ := strconv.Atoi(string(e[k+1:]))
	digits := make([]byte, 0, 20)
	for _, c := range e[:k] {
		if c != '.' {
			digits = append(digits, c)
		}
	}
	decpt := exp + 1
	switch {
	case decpt > -4 && decpt <= 16:
		if decpt <= 0 {
			dst = append(dst, '0', '.')
			for range -decpt {
				dst = append(dst, '0')
			}
			return append(dst, digits...)
		}
		for len(digits) < decpt+1 {
			digits = append(digits, '0')
		}
		dst = append(dst, digits[:decpt]...)
		return append(append(dst, '.'), digits[decpt:]...)
	}
	dst = append(dst, digits[0])
	if len(digits) > 1 {
		dst = append(append(dst, '.'), digits[1:]...)
	}
	dst = append(dst, 'e')
	if exp < 0 {
		dst = append(dst, '-')
		exp = -exp
	} else {
		dst = append(dst, '+')
	}
	return appendPadded(dst, int64(exp), 2)
}
