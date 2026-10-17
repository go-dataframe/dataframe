package dataframe

import (
	"fmt"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"
)

// strModel is the reference a string column is checked against: the
// values as a []string, and which are present.
type strModel struct {
	v  []string
	ok []bool
}

func randomStrings(r *rand.Rand, n int) strModel {
	m := strModel{v: make([]string, n), ok: make([]bool, n)}
	pool := []string{"", "a", "b", "abc", "abcdefg", "abcdefgh", "ünï", "a\x00b", strings.Repeat("long", 9)}
	for i := range n {
		switch r.IntN(4) {
		case 0:
			m.v[i] = pool[r.IntN(len(pool))]
		case 1:
			m.v[i] = fmt.Sprintf("k%d", r.IntN(200)) // more short values than one table holds
		case 2:
			m.v[i] = fmt.Sprintf("a-longer-value-%d", r.IntN(50))
		default:
			m.v[i] = fmt.Sprint(r.IntN(10))
		}
		m.ok[i] = r.IntN(9) != 0
	}
	return m
}

func (m strModel) column() column {
	valid := allValid(len(m.v))
	for i, ok := range m.ok {
		if !ok {
			valid.clear(i)
		}
	}
	return newStrCol(m.v, valid)
}

// checkStrings fails unless c holds m: values, presence, formatting.
func checkStrings(t *testing.T, where string, c column, m strModel) {
	t.Helper()
	s, ok := c.(stringCol)
	if !ok || c.dtype() != String || c.len() != len(m.v) {
		t.Fatalf("%s: %T of %d values, want a string column of %d", where, c, c.len(), len(m.v))
	}
	missing := 0
	for i := range m.v {
		if c.isNA(i) == m.ok[i] {
			t.Fatalf("%s[%d]: missing %v", where, i, c.isNA(i))
		}
		if !m.ok[i] {
			missing++
			if c.value(i) != nil {
				t.Fatalf("%s[%d]: value %v of a missing value", where, i, c.value(i))
			}
			continue
		}
		if s.at(i) != m.v[i] || c.str(i) != m.v[i] || c.value(i) != m.v[i] {
			t.Fatalf("%s[%d] = %q, want %q", where, i, s.at(i), m.v[i])
		}
	}
	if c.naCount() != missing {
		t.Fatalf("%s: %d missing, want %d", where, c.naCount(), missing)
	}
}

// eachWidth runs f with int32 offsets, then with int64 offsets everywhere
// (maxStr32 lowered).
func eachWidth(t *testing.T, f func(t *testing.T)) {
	t.Run("int32", f)
	t.Run("int64", func(t *testing.T) {
		old := maxStr32
		maxStr32 = 3
		defer func() { maxStr32 = old }()
		f(t)
	})
}

func TestStrColAgainstModel(t *testing.T) {
	eachWidth(t, func(t *testing.T) {
		r := rand.New(rand.NewPCG(9, 10))
		for _, n := range []int{0, 1, 37, 5000, 4*parallelMin + 11} {
			m := randomStrings(r, n)
			c := m.column()
			checkStrings(t, "built", c, m)

			// take, a negative position missing.
			idx := make([]int, n+3)
			var want strModel
			for k := range idx {
				idx[k] = r.IntN(n+1) - 1
				if k == 0 {
					idx[k] = -1
				}
				if idx[k] < 0 {
					want.v, want.ok = append(want.v, ""), append(want.ok, false)
				} else {
					want.v, want.ok = append(want.v, m.v[idx[k]]), append(want.ok, m.ok[idx[k]])
				}
			}
			checkStrings(t, "take", c.take(idx), want)

			// slice, and a slice's own slice and take.
			i, j := n/3, n-n/4
			sl := c.slice(i, j)
			sm := strModel{m.v[i:j], m.ok[i:j]}
			checkStrings(t, "slice", sl, sm)
			if j-i > 2 {
				checkStrings(t, "slice of slice", sl.slice(1, j-i-1), strModel{sm.v[1 : j-i-1], sm.ok[1 : j-i-1]})
				checkStrings(t, "take of slice", sl.take([]int{0, j - i - 1}), strModel{[]string{sm.v[0], sm.v[j-i-1]}, []bool{sm.ok[0], sm.ok[j-i-1]}})
			}

			// appendCol, with a column of the other width too.
			other := newStrCol([]string{"x", "", "long-enough-to-matter"}, nil)
			both := strModel{append(slices.Clone(m.v), "x", "", "long-enough-to-matter"), append(slices.Clone(m.ok), true, true, true)}
			checkStrings(t, "append", c.appendCol(other), both)
			checkStrings(t, "append to", other.appendCol(c), strModel{append([]string{"x", "", "long-enough-to-matter"}, m.v...), append([]bool{true, true, true}, m.ok...)})
			checkStrings(t, "append present", newStrCol([]string{"p"}, nil).appendCol(newStrCol([]string{"q"}, nil)), strModel{[]string{"p", "q"}, []bool{true, true}})

			// factorize: codes by first appearance, the distinct values.
			codes, uniq := c.factorize()
			seen := map[string]int32{}
			var order []string
			for k, v := range m.v {
				want := int32(-1)
				if m.ok[k] {
					if _, ok := seen[v]; !ok {
						seen[v] = int32(len(order))
						order = append(order, v)
					}
					want = seen[v]
				}
				if codes[k] != want {
					t.Fatalf("n=%d: code %d at %d, want %d", n, codes[k], k, want)
				}
			}
			checkStrings(t, "uniq", uniq, strModel{order, trues(len(order))})

			// factorizeWith codes a second column in the same space.
			a, b, u := c.factorizeWith(other)
			if !slices.Equal(a, codes) {
				t.Fatalf("n=%d: factorizeWith codes differ from factorize", n)
			}
			for k := range 3 {
				if u.str(int(b[k])) != other.str(k) {
					t.Fatalf("n=%d: joint code of %q", n, other.str(k))
				}
			}
			// Fast paths: comparison to a scalar, membership.
			for _, x := range []string{"", "a", "abc", "k7", "a-longer-value-3", "zzz"} {
				xc := newStrCol([]string{x}, nil)
				for _, op := range []cmpOp{cmpEq, cmpNe, cmpLt, cmpLe, cmpGt, cmpGe} {
					got, err := compareCols(op, c, xc)
					if err != nil {
						t.Fatal(err)
					}
					for k := range m.v {
						want := op == cmpNe
						if m.ok[k] {
							want = op.holds(strings.Compare(m.v[k], x))
						}
						if got.data[k] != want {
							t.Fatalf("n=%d: %q %v %q at %d", n, m.v[k], op, x, k)
						}
					}
				}
			}
			set := map[string]bool{"": true, "k7": true, "a-longer-value-3": true, "abcdefgh": true}
			in := make([]bool, n)
			c.(stringCol).isInSet(map[string]struct{}{"": {}, "k7": {}, "a-longer-value-3": {}, "abcdefgh": {}}, in)
			for k := range m.v {
				if in[k] != (m.ok[k] && set[m.v[k]]) {
					t.Fatalf("n=%d: isin at %d", n, k)
				}
			}
			// lookup, fill, nulls, convert, equal, compare.
			look := c.lookup()
			for _, x := range []string{"", "k7", "absent"} {
				var want []int
				for k, v := range m.v {
					if m.ok[k] && v == x {
						want = append(want, k)
					}
				}
				if got := look(x); !slices.Equal(got, want) {
					t.Fatalf("n=%d: lookup %q", n, x)
				}
			}
			if look(7) != nil {
				t.Fatal("lookup of a non-string")
			}
			filled, err := c.fill("F")
			if err != nil {
				t.Fatal(err)
			}
			fm := strModel{slices.Clone(m.v), trues(n)}
			for k := range fm.v {
				if !m.ok[k] {
					fm.v[k] = "F"
				}
			}
			checkStrings(t, "fill", filled, fm)
			if _, err := c.fill(1); err == nil {
				t.Fatal("fill with an int")
			}
			checkStrings(t, "nulls", c.nulls(3), strModel{[]string{"", "", ""}, []bool{false, false, false}})
			if one, ok := c.convert("v"); !ok || one.str(0) != "v" {
				t.Fatal("convert")
			}
			if _, ok := c.convert(1.5); ok {
				t.Fatal("convert a float")
			}
			if n > 1 {
				if c.equal(0, c, 1) != (m.v[0] == m.v[1]) || c.equal(0, other, 0) != (m.v[0] == "x") {
					t.Fatal("equal")
				}
				if c.equal(0, newCol(Int64, []int64{1}, nil), 0) {
					t.Fatal("equal to an int column")
				}
				if c.compare(0, 1) != strings.Compare(m.v[0], m.v[1]) {
					t.Fatal("compare")
				}
			}
			if _, ok := c.ndarray(); ok {
				t.Fatal("a string column has no ndarray")
			}
		}
	})
}

func trues(n int) []bool {
	b := make([]bool, n)
	for i := range b {
		b[i] = true
	}
	return b
}

// A frame of strings works through the public API at both widths: CSV round
// trip, group-by, merge, sort, concatenation, conversion.
func TestStringsPublicAtBothWidths(t *testing.T) {
	eachWidth(t, func(t *testing.T) {
		df := Must(New(Col("k", []string{"b", "a", "b", "", "c"}), Col("v", []int64{1, 2, 3, 4, 5})))
		g := Must(Must(df.GroupBy("k")).Sum())
		if got := fmt.Sprint(g.Index().Labels(), Must(g.Column("v")).Items()); got != "[ a b c] [4 2 4 5]" {
			t.Errorf("group-by: %s", got)
		}
		s := Must(df.SortValues([]string{"k"}))
		if got := fmt.Sprint(Must(s.Column("k")).Items()); got != "[ a b b c]" {
			t.Errorf("sort: %s", got)
		}
		r := Must(New(Col("k", []string{"c", "b"}), Col("w", []float64{0.5, 1.5})))
		m := Must(df.Merge(r, MergeOptions{On: []string{"k"}}))
		if got := fmt.Sprint(Must(m.Column("k")).Items(), Must(m.Column("w")).Items()); got != "[b b c] [1.5 1.5 0.5]" {
			t.Errorf("merge: %s", got)
		}
		cat := Must(Concat(df, df))
		if got := fmt.Sprint(Must(cat.Column("k")).Items()); got != "[b a b  c b a b  c]" {
			t.Errorf("concat: %s", got)
		}
		up := Must(Must(df.Column("v")).AsType(String))
		if got := fmt.Sprint(up.Items()); got != "[1 2 3 4 5]" {
			t.Errorf("astype: %s", got)
		}
	})
}

// A long float column is factorised in parallel chunks, codes by first
// appearance as the sequential path gives them.
func TestDictCodeParallel(t *testing.T) {
	n := 4*parallelMin + 5
	v := make([]float64, n)
	for i := range v {
		v[i] = float64(i*7919%1000) + 0.5
	}
	c := newCol(Float64, v, nil)
	a, _, u := c.factorizeWith(c.slice(0, 0))
	seen := map[float64]int32{}
	for i, x := range v {
		if _, ok := seen[x]; !ok {
			seen[x] = int32(len(seen))
		}
		if a[i] != seen[x] {
			t.Fatalf("code %d at %d, want %d", a[i], i, seen[x])
		}
	}
	if u.len() != len(seen) {
		t.Fatalf("%d distinct, want %d", u.len(), len(seen))
	}
}
