package dataframe

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"testing"
)

// The corpus test reads every file of testdata/csv with the options of each
// case and compares the frame with what pandas read (recorded by
// testdata/csv/gen_fixtures.py), column by column. The deliberate
// differences, listed in docs/design.md (CSV section), are the rules of csvCompare
// marked D1..D7; anything else must match exactly.

//go:embed testdata/csv/fixtures.json
var csvFixtures []byte

type csvCase struct {
	Name    string
	File    string
	Options map[string]any
	Error   string
	Columns []string
	DTypes  []string
	Values  [][]*string
	Alt     map[string][]*string
	// RoundTrip holds the float columns pandas' correctly rounded parser
	// reads differently from its default one.
	RoundTrip map[string][]*string `json:"round_trip"`
	// ToCSV is how pandas writes the frame it read.
	ToCSV string `json:"to_csv"`
	Index struct {
		Names  []*string
		DTypes []string
		Values [][]*string
	}
}

func loadCSVCases(t *testing.T) []csvCase {
	t.Helper()
	var f struct {
		Pandas string
		Cases  []csvCase
	}
	if err := json.Unmarshal(csvFixtures, &f); err != nil {
		t.Fatal(err)
	}
	t.Logf("fixtures recorded with pandas %s: %d cases", f.Pandas, len(f.Cases))
	return f.Cases
}

// csvCaseOptions maps the read_csv keyword arguments of a case to options.
func csvCaseOptions(t *testing.T, kw map[string]any) []CSVOption {
	t.Helper()
	strs := func(v any) []string {
		var out []string
		for _, x := range v.([]any) {
			out = append(out, x.(string))
		}
		return out
	}
	var opts []CSVOption
	for k, v := range kw {
		switch k {
		case "sep":
			opts = append(opts, CSVSeparator(v.(string)[0]))
		case "header":
			opts = append(opts, CSVHeader(v != nil))
		case "names":
			opts = append(opts, CSVNames(strs(v)...))
		case "usecols":
			opts = append(opts, CSVUseCols(strs(v)...))
		case "nrows":
			opts = append(opts, CSVNRows(int(v.(float64))))
		case "skiprows":
			opts = append(opts, CSVSkipRows(int(v.(float64))))
		case "index_col":
			if s, ok := v.(string); ok {
				opts = append(opts, CSVIndexCol(s))
			} else {
				opts = append(opts, CSVIndexColumn(int(v.(float64))))
			}
		case "na_values":
			opts = append(opts, CSVNAValues(strs(v)...))
		case "keep_default_na":
			opts = append(opts, CSVKeepDefaultNA(v.(bool)))
		case "parse_dates":
			opts = append(opts, CSVParseDates(strs(v)...))
		case "date_format":
			opts = append(opts, CSVDateFormat(v.(string)))
		case "dtype":
			m := map[string]DType{}
			for name, d := range v.(map[string]any) {
				dt, err := ParseDType(d.(string))
				if err != nil {
					t.Fatal(err)
				}
				m[name] = dt
			}
			opts = append(opts, CSVDTypes(m))
		case "quotechar":
			opts = append(opts, CSVQuote(v.(string)[0]))
		case "comment":
			opts = append(opts, CSVComment(v.(string)[0]))
		case "true_values":
			opts = append(opts, CSVTrueValues(strs(v)...))
		case "false_values":
			opts = append(opts, CSVFalseValues(strs(v)...))
		case "skip_blank_lines":
			opts = append(opts, CSVSkipBlankLines(v.(bool)))
		default:
			t.Fatalf("unmapped option %q", k)
		}
	}
	return opts
}

// csvEncode encodes a column's values as the generator does.
func csvEncode(c column) []*string {
	out := make([]*string, c.len())
	for i := range out {
		if c.isNA(i) {
			continue
		}
		var s string
		switch p := c.(type) {
		case *col[float64]:
			s = string(appendFloatRepr(nil, p.data[i], 64))
		case *col[float32]:
			s = string(appendFloatRepr(nil, float64(p.data[i]), 64))
		default:
			s = c.str(i)
			if c.dtype() == Datetime {
				s = "ns:" + strconv.FormatInt(c.(*col[int64]).data[i], 10)
			}
		}
		out[i] = &s
	}
	return out
}

func csvPandasDType(d string) string {
	switch {
	case d == "str" || d == "object":
		return "string"
	case strings.HasPrefix(d, "datetime64"):
		return "datetime"
	}
	return d
}

func strp(p *string) string {
	if p == nil {
		return "<missing>"
	}
	return strconv.Quote(*p)
}

// csvCompare compares one of our columns with pandas'; it returns the
// deliberate difference that applied ("" for none).
func csvCompare(c column, pdt string, want, alt []*string) (string, error) {
	got := csvEncode(c)
	if len(got) != len(want) {
		return "", fmt.Errorf("%d rows, pandas %d", len(got), len(want))
	}
	od := c.dtype()
	text := pdt == "str" || pdt == "object"
	rule := ""
	switch {
	case pdt == "float64" && (od == Int64 || od == Uint64) && c.naCount() > 0:
		// D1: integers with missing values stay integers.
		rule = "D1"
		for i := range got {
			if got[i] != nil {
				v, _ := strconv.ParseFloat(*got[i], 64)
				s := string(appendFloatRepr(nil, v, 64))
				got[i] = &s
			}
		}
	case text && od == Bool && c.naCount() > 0:
		rule = "D2" // bools with missing values stay bools
	case text && od == Uint64 && c.naCount() > 0:
		// D3: uint64 with missing values stays uint64 (pandas keeps the
		// missing-value tokens of such a column as text).
		rule = "D3"
		for i, w := range want {
			if w != nil && slices.Contains(defaultNA, *w) {
				want[i] = nil
			}
		}
	case text && od == Datetime && alt != nil:
		rule = "D4" // parse_dates takes mixed ISO 8601 layouts
		want = alt
	case pdt == "object" && od == String && c.len() > 0:
		rule = "D5" // integers beyond 64 bits stay text
	default:
		if od.String() != csvPandasDType(pdt) {
			return "", fmt.Errorf("dtype %v, pandas %s", od, pdt)
		}
	}
	for i := range got {
		if (got[i] == nil) != (want[i] == nil) || (got[i] != nil && *got[i] != *want[i]) {
			return rule, fmt.Errorf("row %d: %s, pandas %s", i, strp(got[i]), strp(want[i]))
		}
	}
	return rule, nil
}

var lineRE = regexp.MustCompile(`(line|row) (\d+)`)

// TestCSVCorpus runs the corpus once as is and then cut into chunks of a
// few bytes (each chunk runs to the next line end), so that the parallel
// tokenizer starts chunks inside quoted fields and must recover.
func TestCSVCorpus(t *testing.T) {
	cases := loadCSVCases(t)
	t.Run("serial", func(t *testing.T) { runCSVCorpus(t, cases) })
	defer func(n int) { csvMinChunk = n }(csvMinChunk)
	for _, size := range []int{1, 2, 3, 5, 8, 13, 21} {
		t.Run(fmt.Sprintf("chunk%d", size), func(t *testing.T) {
			csvMinChunk = size
			runCSVCorpus(t, cases)
		})
	}
}

func runCSVCorpus(t *testing.T, cases []csvCase) {
	rules := map[string][]string{}
	written := 0
	for _, tc := range cases {
		t.Run(tc.Name, func(t *testing.T) {
			df, err := ReadCSVFile(filepath.Join("testdata", "csv", tc.File), csvCaseOptions(t, tc.Options)...)
			if strings.Contains(tc.Error, "has NA values") && err == nil {
				// D7: an explicit int or bool dtype takes missing values.
				rules["D7"] = append(rules["D7"], tc.Name)
				return
			}
			if tc.Error != "" {
				if err == nil {
					t.Fatalf("no error, pandas: %s\n%v", tc.Error, df)
				}
				if m := lineRE.FindStringSubmatch(tc.Error); m != nil {
					n, _ := strconv.Atoi(m[2])
					if m[1] == "row" {
						n++ // pandas counts rows from 0, lines from 1
					}
					if !strings.Contains(err.Error(), fmt.Sprintf("line %d", n)) {
						t.Errorf("error %q, pandas %q", err, tc.Error)
					}
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			deliberate := false // a deliberate difference makes the written text differ
			if got := df.Columns(); fmt.Sprint(got) != fmt.Sprint(tc.Columns) {
				t.Fatalf("columns %q, pandas %q", got, tc.Columns)
			}
			for k, name := range tc.Columns {
				want := tc.Values[k]
				if rt, ok := tc.RoundTrip[name]; ok {
					// D6: floats are read correctly rounded.
					want = rt
					rules["D6"] = append(rules["D6"], tc.Name+"."+name)
					deliberate = true
				}
				rule, err := csvCompare(df.cols[k], tc.DTypes[k], want, tc.Alt[name])
				if err != nil {
					t.Errorf("column %q: %v", name, err)
				}
				if rule != "" {
					rules[rule] = append(rules[rule], tc.Name+"."+name)
					deliberate = true
				}
				deliberate = deliberate || tc.DTypes[k] == "float32" || strings.Contains(tc.DTypes[k], "UTC")
			}
			ix := df.Index()
			if ix.NLevels() != len(tc.Index.Names) {
				t.Fatalf("%d index levels, pandas %d", ix.NLevels(), len(tc.Index.Names))
			}
			for k, name := range tc.Index.Names {
				if want := strp(name); name == nil && ix.names[k] != "" || name != nil && ix.names[k] != *name {
					t.Errorf("index level %d named %q, pandas %s", k, ix.names[k], want)
				}
				rule, err := csvCompare(ix.level(k), tc.Index.DTypes[k], tc.Index.Values[k], nil)
				if err != nil {
					t.Errorf("index level %d: %v", k, err)
				}
				if rule != "" {
					rules[rule] = append(rules[rule], tc.Name+".index")
					deliberate = true
				}
			}
			if deliberate {
				return
			}
			var sb strings.Builder
			if err := df.WriteCSV(&sb); err != nil {
				t.Fatal(err)
			}
			if sb.String() != tc.ToCSV {
				t.Errorf("WriteCSV wrote\n%q\npandas\n%q", sb.String(), tc.ToCSV)
			}
			written++
		})
	}
	t.Logf("WriteCSV matched pandas' to_csv on %d frames", written)
	for r, where := range rules {
		t.Logf("deliberate difference %s in %d columns: %s", r, len(where), strings.Join(where, ", "))
	}
}
