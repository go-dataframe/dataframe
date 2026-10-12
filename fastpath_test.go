package dataframe

import (
	"math/rand/v2"
	"slices"
	"strings"
	"testing"
	"time"
)

// The fixed-width plan of a layout reads exactly what time.Parse reads:
// whenever it takes a string, time.Parse takes it too, at the same instant.
// Strings are drawn near the layouts' shapes, valid and not.
func TestTimeParserFastAgreesWithParse(t *testing.T) {
	layouts := []string{
		"%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y %H:%M", "%Y%m%d",
		"2006-01-02T15:04:05", "%Y-%m-%dT%H:%M", "%H:%M:%S %Y-%m-%d",
	}
	r := rand.New(rand.NewPCG(3, 4))
	digits := "0123456789"
	took := 0
	for _, l := range layouts {
		p := newTimeParser(l)
		if p.plan == nil {
			t.Fatalf("%q has no plan", l)
		}
		ref := time.Date(2020, 2, 29, 23, 59, 58, 0, time.UTC)
		for k := range 20000 {
			s := ref.Add(time.Duration(r.Int64N(400*365*24)) * time.Hour * time.Duration(1-2*(k%2))).Format(p.layout)
			b := []byte(s)
			switch k % 4 {
			case 1: // a digit changed: out-of-range months, days, hours...
				b[r.IntN(len(b))] = digits[r.IntN(10)]
			case 2: // a byte changed to anything
				b[r.IntN(len(b))] = byte(r.IntN(128))
			case 3: // a byte dropped or doubled
				i := r.IntN(len(b))
				if k%8 == 3 {
					b = append(b[:i:i], b[i+1:]...)
				} else {
					b = append(b[:i+1:i+1], b[i:]...)
				}
			}
			s = string(b)
			ns, ok := p.fast(s)
			if !ok {
				continue
			}
			took++
			tm, err := time.Parse(p.layout, s)
			if err != nil || tm.UnixNano() != ns {
				t.Fatalf("%q by %q: fast %d, time.Parse %v %v", s, l, ns, tm, err)
			}
		}
	}
	if took < 50000 {
		t.Errorf("the plan took only %d strings", took)
	}
	// Leap days, the edges of the plan's years, fields out of range.
	p := newTimeParser("%Y-%m-%d %H:%M:%S")
	for _, s := range []string{"2020-02-29 00:00:00", "2019-02-29 00:00:00", "1678-01-01 00:00:00",
		"1677-12-31 23:59:59", "2261-12-31 23:59:59", "2262-01-01 00:00:00", "2020-13-01 00:00:00",
		"2020-00-10 00:00:00", "2020-04-31 00:00:00", "2020-01-01 24:00:00", "2020-01-01 00:60:00",
		"2020-01-01 00:00:60", "2020-01-01 00:00:00.5", "2020-01-01 7:00:00"} {
		want, werr := func() (int64, error) {
			tm, err := time.Parse(p.layout, s)
			if err != nil {
				return 0, err
			}
			return nanosOf(tm, s)
		}()
		got, gerr := p.parse(s)
		if (werr == nil) != (gerr == nil) || got != want {
			t.Errorf("%q: %d %v, want %d %v", s, got, gerr, want, werr)
		}
	}
}

// Layouts with anything the plan does not read have no plan.
func TestPlanLayout(t *testing.T) {
	for _, l := range []string{"", "2006-01-02 03:04:05 PM", "02 Jan 2006", "2006-01-02.000", "06-01-02",
		"15:04", "2006-01-02 2006", "2006-01-02T15:04:05Z07:00"} {
		if planLayout(l) != nil {
			t.Errorf("%q has a plan", l)
		}
	}
}

// genericCompare is compareCols without its fast paths.
func genericCompare(op cmpOp, a, b column) []bool {
	n := a.len()
	out := make([]bool, n)
	both := a.appendCol(b)
	for i := range n {
		j := i
		if b.len() == 1 {
			j = 0
		}
		if a.isNA(i) || b.isNA(j) {
			out[i] = op == cmpNe
			continue
		}
		out[i] = op.holds(both.compare(i, n+j))
	}
	return out
}

// A string column compared to a string, and tested for membership, gives
// what the general paths give, missing values included.
func TestStringFastPaths(t *testing.T) {
	n := 70_000 // over parallelMin, so that the chunks start mid-bitmap
	data := make([]string, n)
	valid := make([]bool, n)
	for i := range n {
		data[i] = "v" + strings.Repeat("x", i%3) + string(rune('a'+i%7))
		valid[i] = i%5 != 0
	}
	s := Must(NewSeriesMasked("s", data, valid))
	full := Must(NewSeries("s", data))
	for _, ser := range []*Series{s, full} {
		for _, x := range []string{"vxb", "va", "zzz", ""} {
			xc, _ := fromAny([]string{x})
			for _, op := range []cmpOp{cmpEq, cmpNe, cmpLt, cmpLe, cmpGt, cmpGe} {
				got, err := compareCols(op, ser.c, xc)
				if err != nil {
					t.Fatal(err)
				}
				if want := genericCompare(op, ser.c, xc); !slices.Equal(got.data, want) {
					t.Fatalf("%v %q differs", op, x)
				}
			}
		}
		for _, set := range [][]any{{"vxb"}, {"va", "vxxc", "nope"}, {"vb", nil}} {
			got := Must(ser.IsIn(set...)).bools
			v, err := fromValues(set)
			if err != nil {
				t.Fatal(err)
			}
			a, b, m, _ := jointCodes([]column{ser.c}, []column{v})
			in := make([]bool, m)
			for _, c := range b {
				if c >= 0 {
					in[c] = true
				}
			}
			bs, _ := got()
			for i, c := range a {
				if bs[i] != (c >= 0 && in[c]) {
					t.Fatalf("IsIn %v at %d", set, i)
				}
			}
		}
	}
}
