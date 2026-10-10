package dataframe

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"slices"
	"strconv"
)

// CSVOption configures ReadCSV. The options mirror the read_csv parameters
// of pandas that notebooks use; each one names the pandas parameter it
// stands for.
type CSVOption func(*csvConfig)

// csvConfig is what the options set.
type csvConfig struct {
	sep, quote, comment byte
	hasQuote            bool
	hasComment          bool
	header              int // -1 infer, 0 absent, 1 present
	names               []string
	usecols             []string
	nrows               int // -1: every row
	skiprows            int
	indexCol            string
	hasIndexCol         bool
	indexPos            int // -1: none
	naValues            []string
	keepDefaultNA       bool
	parseDates          []string
	dateFormat          string
	dtypes              map[string]DType
	trueValues          []string
	falseValues         []string
	skipBlank           bool
}

// defaultNA is pandas' default list of missing-value tokens.
var defaultNA = []string{
	"", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan",
	"1.#IND", "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a",
	"nan", "null",
}

// CSVSeparator sets the field separator, ',' by default (pandas' sep).
func CSVSeparator(c byte) CSVOption { return func(o *csvConfig) { o.sep = c } }

// CSVHeader says whether the first row (after the skipped rows) holds the
// column names. By default it does, unless CSVNames is given (pandas'
// header="infer"); CSVHeader(true) with CSVNames replaces the names the file
// has (header=0 with names), CSVHeader(false) reads every row as data
// (header=None), naming the columns "0", "1", ... when CSVNames is absent.
func CSVHeader(present bool) CSVOption {
	return func(o *csvConfig) {
		o.header = 0
		if present {
			o.header = 1
		}
	}
}

// CSVNames sets the column names (pandas' names). A row with more fields
// than names gives its leading fields to the index, as in pandas.
func CSVNames(names ...string) CSVOption {
	return func(o *csvConfig) { o.names = slices.Clone(names) }
}

// CSVUseCols keeps only the named columns, in file order (pandas' usecols).
func CSVUseCols(names ...string) CSVOption {
	return func(o *csvConfig) { o.usecols = slices.Clone(names) }
}

// CSVNRows reads at most n data rows (pandas' nrows).
func CSVNRows(n int) CSVOption { return func(o *csvConfig) { o.nrows = n } }

// CSVSkipRows skips the first n lines of the file, blank ones included,
// before the header (pandas' skiprows with an integer).
func CSVSkipRows(n int) CSVOption { return func(o *csvConfig) { o.skiprows = n } }

// CSVIndexCol makes the named column the index (pandas' index_col with a
// name).
func CSVIndexCol(name string) CSVOption {
	return func(o *csvConfig) { o.indexCol, o.hasIndexCol = name, true }
}

// CSVIndexColumn makes the column at position i (among the columns read)
// the index (pandas' index_col with an integer). An unnamed column gives an
// unnamed index, so a file written by WriteCSV with its index reads back
// with CSVIndexColumn(0).
func CSVIndexColumn(i int) CSVOption { return func(o *csvConfig) { o.indexPos = i } }

// CSVNAValues adds tokens read as missing values (pandas' na_values).
func CSVNAValues(tokens ...string) CSVOption {
	return func(o *csvConfig) { o.naValues = append(o.naValues, tokens...) }
}

// CSVKeepDefaultNA says whether pandas' default missing-value tokens ("",
// "NA", "NaN", "null", "None", "#N/A", ...) count as missing, true by
// default (pandas' keep_default_na).
func CSVKeepDefaultNA(keep bool) CSVOption { return func(o *csvConfig) { o.keepDefaultNA = keep } }

// CSVParseDates reads the named columns as Datetime (pandas' parse_dates
// with a list of names). A column holding a value that is not a date stays
// a String column, as pandas leaves it.
func CSVParseDates(names ...string) CSVOption {
	return func(o *csvConfig) { o.parseDates = append(o.parseDates, names...) }
}

// CSVDateFormat sets the layout of the CSVParseDates columns: a Go layout or
// a strftime pattern such as "%d/%m/%Y" (pandas' date_format). By default
// ISO 8601 and a list of common layouts are accepted.
func CSVDateFormat(layout string) CSVOption { return func(o *csvConfig) { o.dateFormat = layout } }

// CSVDTypes sets the dtype of the named columns (pandas' dtype with a dict);
// a value that does not convert is an error. Names not in the file are
// ignored, as in pandas.
func CSVDTypes(dtypes map[string]DType) CSVOption {
	return func(o *csvConfig) {
		if o.dtypes == nil {
			o.dtypes = make(map[string]DType)
		}
		for k, v := range dtypes {
			o.dtypes[k] = v
		}
	}
}

// CSVQuote sets the quote character, '"' by default (pandas' quotechar); 0
// disables quoting (quoting=QUOTE_NONE).
func CSVQuote(c byte) CSVOption {
	return func(o *csvConfig) { o.quote, o.hasQuote = c, c != 0 }
}

// CSVComment sets a comment character: the rest of a line from it is
// ignored, and a line starting with it is skipped (pandas' comment).
func CSVComment(c byte) CSVOption {
	return func(o *csvConfig) { o.comment, o.hasComment = c, c != 0 }
}

// CSVTrueValues adds tokens read as true, besides True, TRUE and true
// (pandas' true_values).
func CSVTrueValues(tokens ...string) CSVOption {
	return func(o *csvConfig) { o.trueValues = append(o.trueValues, tokens...) }
}

// CSVFalseValues adds tokens read as false, besides False, FALSE and false
// (pandas' false_values).
func CSVFalseValues(tokens ...string) CSVOption {
	return func(o *csvConfig) { o.falseValues = append(o.falseValues, tokens...) }
}

// CSVSkipBlankLines says whether empty and whitespace-only lines are
// skipped, true by default; otherwise they are rows of missing values
// (pandas' skip_blank_lines).
func CSVSkipBlankLines(skip bool) CSVOption { return func(o *csvConfig) { o.skipBlank = skip } }

func newCSVConfig(opts []CSVOption) (*csvConfig, error) {
	cfg := &csvConfig{sep: ',', quote: '"', hasQuote: true, header: -1, nrows: -1,
		indexPos: -1, keepDefaultNA: true, skipBlank: true}
	for _, o := range opts {
		o(cfg)
	}
	special := func(c byte) bool { return c == '\n' || c == '\r' }
	switch {
	case special(cfg.sep) || (cfg.hasQuote && (special(cfg.quote) || cfg.quote == cfg.sep)):
		return nil, fmt.Errorf("%w: separator %q and quote %q must differ and not be line ends", ErrArgument, cfg.sep, cfg.quote)
	case cfg.hasComment && (special(cfg.comment) || cfg.comment == cfg.sep || (cfg.hasQuote && cfg.comment == cfg.quote)):
		return nil, fmt.Errorf("%w: comment character %q clashes with the separator, quote or line ends", ErrArgument, cfg.comment)
	case cfg.skiprows < 0:
		return nil, fmt.Errorf("%w: skiprows %d is negative", ErrArgument, cfg.skiprows)
	}
	return cfg, nil
}

// ReadCSV reads a CSV table, as pandas' read_csv does: the first row names
// the columns, each column's dtype is inferred from all its values (int64,
// uint64, float64, bool or string; see CSVParseDates and CSVDTypes for the
// others), pandas' default missing-value tokens are missing, and the index
// is a range unless CSVIndexCol says otherwise.
//
// Fields follow RFC 4180: quoted fields may hold separators, doubled quotes
// and line breaks; lines end with LF, CRLF or CR; a UTF-8 byte order mark is
// skipped. A row with fewer fields than the header gets missing values; a
// row with more is an error naming its line, except the first data row,
// whose extra leading fields become the index (as in pandas).
//
// An integer or bool column with missing values keeps its dtype, the
// missing rows marked in its validity, where pandas would make it float64 or
// object. String values share one buffer holding the input: a string kept
// keeps the whole input alive (see docs/design.md).
func ReadCSV(r io.Reader, opts ...CSVOption) (*DataFrame, error) {
	var data []byte
	var err error
	if f, ok := r.(*os.File); ok {
		data, err = readFile(f)
	} else {
		data, err = io.ReadAll(r)
	}
	if err != nil {
		return nil, err
	}
	return readCSV(data, opts)
}

// ReadCSVFile reads a CSV file as ReadCSV reads one.
func ReadCSVFile(path string, opts ...CSVOption) (*DataFrame, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return ReadCSV(f, opts...)
}

// csvReadBlock is the size of the blocks readFile reads in parallel.
var csvReadBlock = 8 << 20

// readFile reads the rest of a file. When its size is known it reads blocks
// in parallel straight into one buffer, which also spreads the cost of
// faulting in fresh memory over the cores; a pipe is read as it comes.
func readFile(f *os.File) ([]byte, error) {
	info, err := f.Stat()
	if err != nil {
		return nil, err
	}
	off, err := f.Seek(0, io.SeekCurrent)
	size := int(info.Size() - off)
	if err != nil || size <= 0 {
		return io.ReadAll(f)
	}
	data := make([]byte, size)
	nb := (size + csvReadBlock - 1) / csvReadBlock
	errs := make([]error, nb)
	parallelEach(nb, func(k int) {
		lo := k * csvReadBlock
		_, errs[k] = f.ReadAt(data[lo:min(lo+csvReadBlock, size)], off+int64(lo))
	})
	if err := errors.Join(errs...); err != nil {
		return nil, err
	}
	f.Seek(0, io.SeekEnd)
	return data, nil
}

// readCSV parses data, which it owns: string values point into it, so it
// must never change afterwards.
func readCSV(data []byte, opts []CSVOption) (*DataFrame, error) {
	cfg, err := newCSVConfig(opts)
	if err != nil {
		return nil, err
	}
	data = bytes.TrimPrefix(data, []byte("\xef\xbb\xbf"))
	t := newTokenizer(cfg, data)
	pos, lines := 0, 0

	if cfg.skiprows > 0 {
		o := &chunkOut{free: true, keepBlank: true}
		pos = t.parse(o, pos, len(data), cfg.skiprows)
		lines += o.lines
		if o.err != nil {
			return nil, t.fail(o.err, 0)
		}
	}

	hasHeader := cfg.header == 1 || (cfg.header == -1 && cfg.names == nil)
	var header []string
	if hasHeader {
		o := &chunkOut{free: true}
		next := t.parse(o, pos, len(data), 1)
		if o.err != nil {
			return nil, t.fail(o.err, lines)
		}
		if o.nrec == 1 {
			header = make([]string, o.counts[0])
			for j := range header {
				header[j] = t.raw(o, j)
			}
		}
		lines += o.lines
		pos = next
	}

	names := cfg.names
	if names == nil {
		names = header
	}
	first := &chunkOut{free: true}
	t.parse(first, pos, len(data), 1)
	if first.err != nil {
		return nil, t.fail(first.err, lines)
	}
	if names == nil && !hasHeader && first.nrec == 1 {
		for j := range first.counts[0] {
			names = append(names, strconv.Itoa(j))
		}
	}
	if names == nil {
		return nil, fmt.Errorf("%w: no columns to parse from file", ErrParse)
	}
	var unnamed []bool
	if cfg.names == nil {
		names, unnamed = dedupHeader(names)
	} else {
		if d := firstDuplicate(names); d != "" {
			return nil, fmt.Errorf("%w: duplicate name %q in CSVNames", ErrArgument, d)
		}
		unnamed = make([]bool, len(names))
	}
	implicit := 0
	if first.nrec == 1 && first.counts[0] > len(names) {
		implicit = first.counts[0] - len(names)
	}
	ncols := implicit + len(names)

	groups, err := t.parseAll(pos, ncols, cfg.nrows, lines)
	if err != nil {
		return nil, err
	}
	return buildFrame(t, cfg, groups, names, unnamed, implicit)
}

// dedupHeader names empty header cells "Unnamed: i" and renames duplicates
// "a.1", "a.2", ..., as pandas does: named columns first, then the unnamed
// ones, skipping a suffixed name the header already holds.
func dedupHeader(header []string) ([]string, []bool) {
	cols := slices.Clone(header)
	unnamed := make([]bool, len(cols))
	var order, tail []int
	for i, c := range cols {
		if c == "" {
			cols[i] = "Unnamed: " + strconv.Itoa(i)
			unnamed[i] = true
			tail = append(tail, i)
		} else {
			order = append(order, i)
		}
	}
	counts := make(map[string]int)
	for _, i := range append(order, tail...) {
		col := cols[i]
		old := col
		cur := counts[col]
		for cur > 0 {
			counts[old] = cur + 1
			col = old + "." + strconv.Itoa(cur)
			if slices.Contains(cols, col) {
				cur++
			} else {
				cur = counts[col]
			}
		}
		cols[i] = col
		counts[col] = cur + 1
	}
	return cols, unnamed
}

// firstDuplicate returns the first name given twice, or "".
func firstDuplicate(names []string) string {
	seen := make(map[string]bool, len(names))
	for _, n := range names {
		if seen[n] {
			return n
		}
		seen[n] = true
	}
	return ""
}

// buildFrame converts the parsed fields to columns and assembles the frame.
func buildFrame(t *tokenizer, cfg *csvConfig, groups []*chunkOut, names []string, unnamed []bool, implicit int) (*DataFrame, error) {
	nameOf := func(j int) string {
		if j < implicit {
			return ""
		}
		return names[j-implicit]
	}
	use := make([]bool, implicit+len(names))
	for j := range use {
		use[j] = cfg.usecols == nil || j < implicit
	}
	var missing []string
	for _, u := range cfg.usecols {
		k := slices.Index(names, u)
		if k < 0 {
			missing = append(missing, u)
			continue
		}
		use[implicit+k] = true
	}
	if missing != nil {
		return nil, fmt.Errorf("%w: usecols do not match columns, columns expected but not found: %q", ErrColumn, missing)
	}
	isDate := make(map[string]bool)
	for _, d := range cfg.parseDates {
		k := slices.Index(names, d)
		if k < 0 {
			return nil, fmt.Errorf("%w: missing column provided to parse_dates: %q", ErrColumn, d)
		}
		isDate[d] = use[implicit+k]
	}

	var specs []colSpec
	for j, u := range use {
		if !u {
			continue
		}
		sp := colSpec{j: j, name: nameOf(j), mode: convInfer}
		if j >= implicit {
			if dt, ok := cfg.dtypes[sp.name]; ok {
				sp.mode, sp.dt = convDType, dt
			} else if isDate[sp.name] {
				sp.mode = convDate
			}
		}
		specs = append(specs, sp)
	}
	cols, err := newConverter(t, cfg, groups).run(specs)
	if err != nil {
		return nil, err
	}

	df := &DataFrame{}
	var ixNames []string
	var ixLevels []column
	for k, sp := range specs {
		if sp.j < implicit {
			ixNames = append(ixNames, "")
			ixLevels = append(ixLevels, cols[k])
			continue
		}
		df.names = append(df.names, sp.name)
		df.cols = append(df.cols, cols[k])
	}
	rows := 0
	for _, g := range groups {
		rows += g.nrec
	}
	if implicit > 0 {
		df.index = labelIndex(ixNames, ixLevels)
	} else {
		df.index = NewRangeIndex(rows)
	}

	switch {
	case cfg.hasIndexCol:
		return df.SetIndex(cfg.indexCol)
	case cfg.indexPos >= 0:
		if cfg.indexPos >= len(df.names) {
			return nil, fmt.Errorf("%w: index column %d of %d", ErrIndex, cfg.indexPos, len(df.names))
		}
		name := df.names[cfg.indexPos]
		out, _ := df.SetIndex(name)
		if k := slices.Index(names, name); unnamed[k] {
			out.index, _ = out.index.Rename("")
		}
		return out, nil
	}
	return df, nil
}
