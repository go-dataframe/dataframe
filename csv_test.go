package dataframe

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"testing/iotest"
	"time"
)

func readStr(t *testing.T, s string, opts ...CSVOption) *DataFrame {
	t.Helper()
	df, err := ReadCSV(strings.NewReader(s), opts...)
	if err != nil {
		t.Fatalf("ReadCSV(%q): %v", s, err)
	}
	return df
}

// cells returns a column's values as csvEncode does, "<NA>" for missing.
func cells(t *testing.T, df *DataFrame, name string) []string {
	t.Helper()
	s, err := df.Column(name)
	if err != nil {
		t.Fatal(err)
	}
	var out []string
	for _, p := range csvEncode(s.c) {
		if p == nil {
			out = append(out, "<NA>")
		} else {
			out = append(out, *p)
		}
	}
	return out
}

func wantCells(t *testing.T, df *DataFrame, name string, dt DType, want ...string) {
	t.Helper()
	s, err := df.Column(name)
	if err != nil {
		t.Fatal(err)
	}
	if s.DType() != dt {
		t.Errorf("column %q is %v, want %v", name, s.DType(), dt)
	}
	if got := cells(t, df, name); fmt.Sprint(got) != fmt.Sprint(want) {
		t.Errorf("column %q = %q, want %q", name, got, want)
	}
}

func TestCSVOptionErrors(t *testing.T) {
	for _, opts := range [][]CSVOption{
		{CSVSeparator('\n')},
		{CSVQuote(',')},
		{CSVQuote('\r')},
		{CSVComment(',')},
		{CSVComment('"')},
		{CSVComment('\n')},
		{CSVSkipRows(-1)},
	} {
		if _, err := ReadCSV(strings.NewReader("a\n1\n"), opts...); !errors.Is(err, ErrArgument) {
			t.Errorf("err = %v, want ErrArgument", err)
		}
	}
	// A comment character equal to a disabled quote character is fine.
	readStr(t, "a\n1\n", CSVQuote(0), CSVComment('"'))
}

func TestCSVInputErrors(t *testing.T) {
	boom := errors.New("boom")
	if _, err := ReadCSV(iotest.ErrReader(boom)); !errors.Is(err, boom) {
		t.Errorf("err = %v", err)
	}
	if _, err := ReadCSVFile(filepath.Join(t.TempDir(), "missing.csv")); err == nil {
		t.Error("missing file read")
	}
	// A directory opens but does not read.
	if _, err := ReadCSVFile(t.TempDir()); err == nil {
		t.Error("directory read")
	}
	f, err := os.Open(filepath.Join("testdata", "csv", "basic.csv"))
	if err != nil {
		t.Fatal(err)
	}
	f.Close()
	if _, err := ReadCSV(f); err == nil {
		t.Error("closed file read")
	}
}

func TestCSVReadFile(t *testing.T) {
	defer func(n int) { csvReadBlock = n }(csvReadBlock)
	csvReadBlock = 7 // several blocks read in parallel
	path := filepath.Join("testdata", "csv", "basic.csv")
	df, err := ReadCSVFile(path)
	if err != nil {
		t.Fatal(err)
	}
	wantCells(t, df, "name", String, "alpha", "beta", "gamma")

	// A file already partly read is read from where it stands.
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	io.ReadFull(f, make([]byte, len("id,x,name,flag\n")))
	df, err = ReadCSV(f, CSVHeader(false))
	if err != nil {
		t.Fatal(err)
	}
	wantCells(t, df, "0", Int64, "1", "2", "3")
	if n, _ := f.Read(make([]byte, 1)); n != 0 {
		t.Error("file not read to its end")
	}

	// A pipe has no size: it is read as it comes.
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	go func() { w.WriteString("a,b\n1,2\n"); w.Close() }()
	df, err = ReadCSV(r)
	r.Close()
	if err != nil {
		t.Fatal(err)
	}
	wantCells(t, df, "b", Int64, "2")
}

func TestCSVTokenizeErrors(t *testing.T) {
	for _, tc := range []struct {
		in   string
		opts []CSVOption
		want string
	}{
		{"x\n\"y\n", []CSVOption{CSVSkipRows(2)}, "EOF inside string starting at line 2"},
		{"\"a,b\n1,2\n", nil, "EOF inside string starting at line 1"},
		{"a,b\n\n\"1,2\n", nil, "EOF inside string starting at line 3"},
		{"a,b\n1,2\n3,4,5,6\n", nil, "expected 2 fields in line 3, saw 4"},
		{"", nil, "no columns to parse"},
		{"\n \n", []CSVOption{CSVHeader(false)}, "no columns to parse"},
	} {
		_, err := ReadCSV(strings.NewReader(tc.in), tc.opts...)
		if !errors.Is(err, ErrParse) || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%q: err = %v, want %q", tc.in, err, tc.want)
		}
	}
}

func TestCSVRecordTooLong(t *testing.T) {
	defer func(n int) { csvMaxOffset = n }(csvMaxOffset)
	csvMaxOffset = 10
	_, err := ReadCSV(strings.NewReader("a,b\n1,2\n333333,444444\n"))
	if !errors.Is(err, ErrParse) || !strings.Contains(err.Error(), "line 3 is too long") {
		t.Errorf("err = %v", err)
	}
}

func TestCSVHeaderAndNames(t *testing.T) {
	df := readStr(t, "1,2\n3,4\n", CSVNames("x", "y"))
	wantCells(t, df, "x", Int64, "1", "3")
	if _, err := ReadCSV(strings.NewReader("1,2\n"), CSVNames("x", "x")); !errors.Is(err, ErrArgument) {
		t.Errorf("duplicate names: %v", err)
	}
	// The header row is replaced, its own names ignored.
	df = readStr(t, "a,b\n1,2\n", CSVNames("x", "y"), CSVHeader(true))
	if fmt.Sprint(df.Columns()) != "[x y]" || df.Len() != 1 {
		t.Errorf("columns %v, %d rows", df.Columns(), df.Len())
	}
	// Leading blank and comment lines come before the header.
	df = readStr(t, "\n# note\n\na,b\n1,2\n", CSVComment('#'))
	wantCells(t, df, "b", Int64, "2")
}

func TestCSVIndex(t *testing.T) {
	df := readStr(t, ",a\nx,1\ny,2\n", CSVIndexColumn(0))
	if df.Index().Name() != "" || fmt.Sprint(df.Index().Labels()) != "[x y]" {
		t.Errorf("index %q %v", df.Index().Name(), df.Index().Labels())
	}
	df = readStr(t, "k,a\nx,1\n", CSVIndexColumn(1))
	if df.Index().Name() != "a" || fmt.Sprint(df.Columns()) != "[k]" {
		t.Errorf("index %q, columns %v", df.Index().Name(), df.Columns())
	}
	if _, err := ReadCSV(strings.NewReader("a\n1\n"), CSVIndexColumn(1)); !errors.Is(err, ErrIndex) {
		t.Errorf("err = %v", err)
	}
	if _, err := ReadCSV(strings.NewReader("a\n1\n"), CSVIndexCol("b")); !errors.Is(err, ErrColumn) {
		t.Errorf("err = %v", err)
	}
	// Two extra leading fields make a two-level index.
	df = readStr(t, "c\n1,2,3\n4,5,6\n")
	if df.Index().NLevels() != 2 || fmt.Sprint(df.Index().Label(1)) != "[4 5]" {
		t.Errorf("index %v", df.Index().Labels())
	}
}

// ns encodes a date as csvEncode does.
func ns(y, m, d int) string {
	return fmt.Sprint("ns:", time.Date(y, time.Month(m), d, 0, 0, 0, 0, time.UTC).UnixNano())
}

func TestCSVParseDates(t *testing.T) {
	in := "d,e,f\n2020-01-02,02/03/2021,x\nNA,,y\n2021-03-04T05:06:07.5,04/05/2022,z\n"
	df := readStr(t, in, CSVParseDates("d"))
	wantCells(t, df, "d", Datetime, ns(2020, 1, 2), "<NA>",
		fmt.Sprint("ns:", time.Date(2021, 3, 4, 5, 6, 7, 5e8, time.UTC).UnixNano()))
	df = readStr(t, in, CSVParseDates("e"), CSVDateFormat("%d/%m/%Y"))
	wantCells(t, df, "e", Datetime, ns(2021, 3, 2), "<NA>", ns(2022, 5, 4))
	// A value that is not a date leaves the column text.
	df = readStr(t, in, CSVParseDates("f"))
	wantCells(t, df, "f", String, "x", "y", "z")
	// A date column left out by usecols is not read.
	df = readStr(t, in, CSVParseDates("d"), CSVUseCols("f"))
	if fmt.Sprint(df.Columns()) != "[f]" {
		t.Errorf("columns %v", df.Columns())
	}
	if _, err := ReadCSV(strings.NewReader(in), CSVParseDates("nope")); !errors.Is(err, ErrColumn) {
		t.Errorf("err = %v", err)
	}
}

func TestCSVDTypes(t *testing.T) {
	in := "i,f,b,d,s,c,u,z\n1,1.5,yes,2020-01-01,x,p,1,1\n,NA,0,,,q,2,2+1i\n-3,7,1,2020-01-03,z,p,3,3\n"
	df := readStr(t, in, CSVTrueValues("yes"), CSVDTypes(map[string]DType{"i": Int64}),
		CSVDTypes(map[string]DType{"f": Float64, "b": Bool, "d": Datetime, "s": String,
			"c": Category, "u": Uint8, "z": Complex128, "absent": Int8}))
	wantCells(t, df, "i", Int64, "1", "<NA>", "-3")
	wantCells(t, df, "f", Float64, "1.5", "<NA>", "7.0")
	wantCells(t, df, "b", Bool, "True", "False", "True")
	wantCells(t, df, "d", Datetime, ns(2020, 1, 1), "<NA>", ns(2020, 1, 3))
	wantCells(t, df, "s", String, "x", "<NA>", "z")
	wantCells(t, df, "c", Category, "p", "q", "p")
	wantCells(t, df, "u", Uint8, "1", "2", "3")
	wantCells(t, df, "z", Complex128, "(1+0i)", "(2+1i)", "(3+0i)")
	df = readStr(t, "b\nTrue\n\n", CSVDTypes(map[string]DType{"b": Bool}), CSVSkipBlankLines(false))
	wantCells(t, df, "b", Bool, "True", "<NA>")

	for col, dt := range map[string]DType{"f": Int64, "s": Float64, "b": Bool, "s ": Datetime, "z": Int8} {
		in := "f,s,b,s ,z\n1.5,x,maybe,x,x\n"
		_, err := ReadCSV(strings.NewReader(in), CSVDTypes(map[string]DType{col: dt}))
		if !errors.Is(err, ErrParse) || !strings.Contains(err.Error(), fmt.Sprintf("%q", col)) {
			t.Errorf("%s as %v: err = %v", col, dt, err)
		}
	}
}

func TestCSVInference(t *testing.T) {
	// Each column switches kind part way, inside one chunk.
	in := "a,b,c,d,e,f,g,h\n" +
		"1,1,True,18446744073709551615,1,NA,1,-0\n" +
		"2,2.5,x,1.5,True,NA,99999999999999999999,1\n" +
		"3,NA,False,2,,NA,3,2.5\n"
	df := readStr(t, in)
	wantCells(t, df, "a", Int64, "1", "2", "3")
	wantCells(t, df, "b", Float64, "1.0", "2.5", "<NA>")
	wantCells(t, df, "c", String, "True", "x", "False")
	wantCells(t, df, "d", Float64, "1.8446744073709552e+19", "1.5", "2.0")
	wantCells(t, df, "e", String, "1", "True", "<NA>")
	wantCells(t, df, "f", Float64, "<NA>", "<NA>", "<NA>")
	wantCells(t, df, "g", String, "1", "99999999999999999999", "3")
	wantCells(t, df, "h", Float64, "-0.0", "1.0", "2.5")

	// A numeric missing-value token is checked before numbers.
	df = readStr(t, "a\n-999\n5\n", CSVNAValues("-999"))
	wantCells(t, df, "a", Int64, "<NA>", "5")
	// Without the default tokens nothing but the given ones is missing.
	df = readStr(t, "a,b\nNA,\nx,1\n", CSVKeepDefaultNA(false), CSVNAValues("x"))
	wantCells(t, df, "a", String, "NA", "<NA>")
	wantCells(t, df, "b", String, "", "1")
	df = readStr(t, "a\nyes\nno\nNA\n", CSVTrueValues("yes"), CSVFalseValues("no"))
	wantCells(t, df, "a", Bool, "True", "False", "<NA>")
}

// TestCSVInferenceChunked spreads the kinds of a column over chunks, so
// that chunks read as integers become floats or uints afterwards.
func TestCSVInferenceChunked(t *testing.T) {
	defer func(n int) { csvMinChunk = n }(csvMinChunk)
	csvMinChunk = 1
	df := readStr(t, "a,b,c,d,e\n1,NA,1,NA,1\n-0,NA,2,NA,NA\n3,NA,18446744073709551615,2.5,x\n2.5,1.5,4,NA,4\n")
	wantCells(t, df, "a", Float64, "1.0", "-0.0", "3.0", "2.5")
	wantCells(t, df, "b", Float64, "<NA>", "<NA>", "<NA>", "1.5")
	wantCells(t, df, "c", Uint64, "1", "2", "18446744073709551615", "4")
	wantCells(t, df, "d", Float64, "<NA>", "<NA>", "2.5", "<NA>")
	wantCells(t, df, "e", String, "1", "<NA>", "x", "4")
	// A chunk holding a uint and then a float is read again as floats.
	df = readStr(t, "a\n1\n18446744073709551615\n2.5\n3\n")
	wantCells(t, df, "a", Float64, "1.0", "1.8446744073709552e+19", "2.5", "3.0")
}

func TestCSVParseNumber(t *testing.T) {
	for _, tc := range []struct {
		s   string
		cls int
		f   float64
	}{
		{"12", clsInt, 12}, {" -12 ", clsInt, -12}, {"+7", clsInt, 7},
		{"9223372036854775808", clsUint, 9223372036854775808},
		{"-9223372036854775808", clsInt, -9223372036854775808},
		{"-9223372036854775809", clsBig, -9223372036854775809},
		{"18446744073709551616", clsBig, 18446744073709551616},
		{"1.5", clsFloat, 1.5}, {"1.", clsFloat, 1}, {".5", clsFloat, .5},
		{"1e5", clsFloat, 1e5}, {"1E-3", clsFloat, 1e-3}, {"-2.5e+3", clsFloat, -2500},
		{"1e400", clsFloat, math.Inf(1)}, {"12345678901234567890.5", clsFloat, 12345678901234567890.5},
		{"1e99999", clsFloat, math.Inf(1)}, {"0.30000000000000004", clsFloat, 0.30000000000000004},
		{"inf", clsFloat, math.Inf(1)}, {"-Infinity", clsFloat, math.Inf(-1)},
		{" inf", clsOther, 0}, {"infinit", clsOther, 0}, {"infx", clsOther, 0}, {"nan", clsOther, 0},
		{"1e", clsOther, 0}, {"1e+", clsOther, 0}, {"1.5x", clsOther, 0}, {"", clsOther, 0},
		{"   ", clsOther, 0}, {"-", clsOther, 0}, {".", clsOther, 0}, {"0x10", clsOther, 0},
	} {
		cls, _, f, _ := parseNumber(tc.s)
		if cls != tc.cls || (cls != clsOther && f != tc.f) {
			t.Errorf("parseNumber(%q) = %d %v, want %d %v", tc.s, cls, f, tc.cls, tc.f)
		}
	}
}

func TestCSVLexical(t *testing.T) {
	// Comments: at EOF, after a field; quotes disabled.
	df := readStr(t, "a,b\n1,2#x\n3,4#y", CSVComment('#'))
	wantCells(t, df, "b", Int64, "2", "4")
	df = readStr(t, "a\n\"x\n", CSVQuote(0))
	wantCells(t, df, "a", String, "\"x")
	// Whitespace-only last line; separator that is a space.
	df = readStr(t, "a,b\n1,2\n  \t")
	if df.Len() != 1 {
		t.Errorf("%d rows", df.Len())
	}
	df = readStr(t, "a b\n1 2\n", CSVSeparator(' '))
	wantCells(t, df, "b", Int64, "2")
	// Skipped rows count blank lines; a CR line end.
	df = readStr(t, "\r\nskip\ra,b\r1,2\r", CSVSkipRows(2))
	wantCells(t, df, "a", Int64, "1")
	// nrows stops before a bad row; usecols keeps file order.
	df = readStr(t, "a,b,c\n1,2,3\n4,5\n6,7,8,9\n", CSVNRows(2), CSVUseCols("c", "a"))
	if fmt.Sprint(df.Columns()) != "[a c]" {
		t.Errorf("columns %v", df.Columns())
	}
	wantCells(t, df, "c", Int64, "3", "<NA>")
}

// randomCSV builds an awkward table: quoted fields with separators, doubled
// quotes and line breaks, CRLF, CR and LF line ends, short rows and blank
// lines.
func randomCSV(r *rand.Rand, rows, cols int) string {
	var b strings.Builder
	ends := []string{"\n", "\r\n", "\r"}
	for j := range cols {
		if j > 0 {
			b.WriteByte(',')
		}
		fmt.Fprintf(&b, "c%d", j)
	}
	b.WriteString("\n")
	pieces := []string{"x", "1", "2.5", "NA", "", " ", "a,b", "q\"q", "l\nm", "c\r\nd", "True"}
	for range rows {
		n := cols
		if r.IntN(8) == 0 {
			n = 1 + r.IntN(cols)
		}
		for j := range n {
			if j > 0 {
				b.WriteByte(',')
			}
			p := pieces[r.IntN(len(pieces))]
			if strings.ContainsAny(p, ",\"\r\n") || r.IntN(4) == 0 {
				p = "\"" + strings.ReplaceAll(p, "\"", "\"\"") + "\""
			}
			b.WriteString(p)
		}
		b.WriteString(ends[r.IntN(len(ends))])
		if r.IntN(10) == 0 {
			b.WriteString("\n")
		}
	}
	return b.String()
}

// TestCSVChunkedMatchesSerial checks, on random awkward input, that every
// chunk size gives the frame one serial pass gives.
func TestCSVChunkedMatchesSerial(t *testing.T) {
	defer func(n int) { csvMinChunk = n }(csvMinChunk)
	r := rand.New(rand.NewPCG(3, 4))
	for trial := range 60 {
		in := randomCSV(r, 1+r.IntN(40), 1+r.IntN(4))
		csvMinChunk = 1 << 30
		want := readStr(t, in)
		var wb strings.Builder
		want.WriteCSV(&wb)
		for _, size := range []int{1, 2, 3, 7, 16, 50} {
			csvMinChunk = size
			got := readStr(t, in)
			var gb strings.Builder
			got.WriteCSV(&gb)
			if gb.String() != wb.String() || fmt.Sprint(got.DTypes()) != fmt.Sprint(want.DTypes()) {
				t.Fatalf("trial %d, chunks of %d: %q\ngot  %q %v\nwant %q %v", trial, size, in,
					gb.String(), got.DTypes(), wb.String(), want.DTypes())
			}
		}
	}
}

func TestCSVParallelError(t *testing.T) {
	defer func(n int) { csvMinChunk = n }(csvMinChunk)
	csvMinChunk = 4
	_, err := ReadCSV(strings.NewReader("a,b\n1,2\n3,4\n5,6\n7,8,9\n10,11\n"))
	if err == nil || !strings.Contains(err.Error(), "line 5") {
		t.Errorf("err = %v", err)
	}
}

func TestAppendFloatRepr(t *testing.T) {
	for _, tc := range []struct {
		f    float64
		bits int
		want string
	}{
		{1, 64, "1.0"}, {-3, 64, "-3.0"}, {0.1, 64, "0.1"}, {1e-5, 64, "1e-05"}, {1e-4, 64, "0.0001"},
		{1e16, 64, "1e+16"}, {1e15, 64, "1000000000000000.0"}, {123456789.123, 64, "123456789.123"},
		{math.Copysign(0, -1), 64, "-0.0"}, {math.Inf(1), 64, "inf"}, {math.Inf(-1), 64, "-inf"},
		{1.5e300, 64, "1.5e+300"}, {5e-324, 64, "5e-324"}, {0.000123, 64, "0.000123"},
		{1.2345678901234568e+17, 64, "1.2345678901234568e+17"}, {float64(float32(0.1)), 32, "0.1"},
		{-2.5e-7, 64, "-2.5e-07"}, {1234567890123456.7, 64, "1234567890123456.8"},
	} {
		if got := string(appendFloatRepr(nil, tc.f, tc.bits)); got != tc.want {
			t.Errorf("appendFloatRepr(%v) = %q, want %q", tc.f, got, tc.want)
		}
	}
}

func TestWriteCSV(t *testing.T) {
	day := time.Date(2020, 1, 2, 0, 0, 0, 0, time.UTC)
	df, err := New(
		Col("f", []float64{1.5, math.NaN(), 1e20}),
		Col("f32", []float32{0.1, 2, 3}),
		Col("u", []uint64{1, 18446744073709551615, 3}),
		Col("b", []bool{true, false, true}),
		Col("i8", []int8{-1, 0, 1}),
		Col("s", []string{"a,b", "q\"x", "l\nm"}),
		Col("d", []time.Time{day, day.AddDate(0, 0, 1), day.AddDate(-60, 0, 0)}),
	)
	if err != nil {
		t.Fatal(err)
	}
	var b strings.Builder
	if err := df.WriteCSV(&b); err != nil {
		t.Fatal(err)
	}
	want := ",f,f32,u,b,i8,s,d\n" +
		"0,1.5,0.1,1,True,-1,\"a,b\",2020-01-02\n" +
		"1,,2.0,18446744073709551615,False,0,\"q\"\"x\",2020-01-03\n" +
		"2,1e+20,3.0,3,True,1,\"l\nm\",1960-01-02\n"
	if b.String() != want {
		t.Errorf("got\n%q\nwant\n%q", b.String(), want)
	}

	b.Reset()
	one, _ := df.Select("f")
	one.WriteCSV(&b, CSVWriteIndex(false), CSVWriteHeader(false), CSVWriteNARep(""))
	if b.String() != "1.5\n\"\"\n1e+20\n" {
		t.Errorf("got %q", b.String())
	}
	b.Reset()
	two, _ := df.Select("f", "s")
	two.WriteCSV(&b, CSVWriteSeparator(';'), CSVWriteNARep("NULL"), CSVWriteLineTerminator("\r\n"))
	if b.String() != ";f;s\r\n0;1.5;a,b\r\n1;NULL;\"q\"\"x\"\r\n2;1e+20;\"l\nm\"\r\n" {
		t.Errorf("got %q", b.String())
	}
	// A failing writer fails the write.
	if err := df.WriteCSV(failWriter{}); err == nil {
		t.Error("no error from a failing writer")
	}
}

type failWriter struct{}

func (failWriter) Write([]byte) (int, error) { return 0, errors.New("full") }

func TestWriteCSVDatetimes(t *testing.T) {
	base := time.Date(2021, 3, 4, 5, 6, 7, 0, time.UTC)
	for _, tc := range []struct {
		extra time.Duration
		want  string
	}{
		{0, "2021-03-04 05:06:07"},
		{500 * time.Millisecond, "2021-03-04 05:06:07.500"},
		{time.Microsecond, "2021-03-04 05:06:07.000001"},
		{time.Nanosecond, "2021-03-04 05:06:07.000000001"},
	} {
		s, _ := NewSeriesMasked("d", []time.Time{base.Add(tc.extra), base}, []bool{true, false})
		var b strings.Builder
		s.ToFrame().WriteCSV(&b, CSVWriteIndex(false))
		if want := "d\n" + tc.want + "\n\"\"\n"; b.String() != want {
			t.Errorf("got %q, want %q", b.String(), want)
		}
	}
}

func TestWriteCSVIndexAndCategory(t *testing.T) {
	df := readStr(t, "k,c,v\nx,p,1\ny,q,\n", CSVDTypes(map[string]DType{"c": Category}), CSVIndexCol("k"))
	var b strings.Builder
	df.WriteCSV(&b)
	if b.String() != "k,c,v\nx,p,1\ny,q,\n" {
		t.Errorf("got %q", b.String())
	}
}

func TestWriteCSVRoundTrip(t *testing.T) {
	defer func(n int) { csvWriteBlock = n }(csvWriteBlock)
	csvWriteBlock = 3 // several batches of blocks
	r := rand.New(rand.NewPCG(5, 6))
	n := 100
	ids := make([]int64, n)
	xs := make([]float64, n)
	ss := make([]string, n)
	for i := range n {
		ids[i] = r.Int64() - r.Int64()
		xs[i] = r.NormFloat64() * math.Pow(10, float64(r.IntN(40)-20))
		ss[i] = []string{"plain", "a,b", "q\"q", "l\nm", " sp ", "é😀"}[r.IntN(6)]
	}
	df, err := New(Col("id", ids), Col("x", xs), Col("s", ss))
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "rt.csv")
	if err := df.WriteCSVFile(path); err != nil {
		t.Fatal(err)
	}
	back, err := ReadCSVFile(path, CSVIndexColumn(0))
	if err != nil {
		t.Fatal(err)
	}
	var a, b bytes.Buffer
	df.WriteCSV(&a)
	back.WriteCSV(&b)
	if a.String() != b.String() || fmt.Sprint(back.DTypes()) != "[int64 float64 string]" {
		t.Errorf("round trip changed the frame: %v", back.DTypes())
	}
	for i, x := range xs {
		if got := back.cols[1].(*col[float64]).data[i]; got != x {
			t.Fatalf("row %d: %v read back as %v", i, x, got)
		}
	}
	if err := df.WriteCSVFile(filepath.Join(t.TempDir(), "no", "such", "dir.csv")); err == nil {
		t.Error("wrote into a missing directory")
	}
}
