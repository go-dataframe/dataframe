package dataframe

import (
	"fmt"
	"html"
	"math"
	"strconv"
	"strings"
	"unicode/utf8"
)

// Display limits, pandas' defaults: past MaxRows rows only the first and last
// MinRows/2 are shown; past MaxColumns columns only the first and last
// MaxColumns/2.
var (
	MaxRows    = 60
	MinRows    = 10
	MaxColumns = 20
)

// formatColumn formats the elements at rows of a column the way pandas
// prints a column: floats to a common number of decimals (or scientific
// notation when that would hide digits), instants as dates when every one is
// at midnight, missing values as NaN, NaT or <NA>.
func formatColumn(c column, rows []int) []string {
	out := make([]string, len(rows))
	switch p := c.(type) {
	case *col[float64]:
		return formatFloats(p, rows, 64)
	case *col[float32]:
		f := &col[float64]{dt: Float64, o: opsFloat64, valid: p.valid, data: make([]float64, len(p.data))}
		for i, v := range p.data {
			f.data[i] = float64(v)
		}
		return formatFloats(f, rows, 32)
	case *col[int64]:
		if p.dt == Datetime {
			dateOnly := true
			for _, i := range rows {
				if !p.isNA(i) && p.data[i]%(86400e9) != 0 {
					dateOnly = false
				}
			}
			for k, i := range rows {
				switch {
				case p.isNA(i):
					out[k] = "NaT"
				case dateOnly:
					out[k] = timeOf(p.data[i]).Format("2006-01-02")
				default:
					out[k] = p.str(i)
				}
			}
			return out
		}
	}
	for k, i := range rows {
		out[k] = cellString(c, i)
	}
	return out
}

// formatFloats formats float elements to a shared precision, as pandas'
// float formatter does with display.precision = 6.
func formatFloats(c *col[float64], rows []int, bits int) []string {
	const digits = 6
	out := make([]string, len(rows))
	dec := 1
	sci := false
	maxLen := 0
	large := false
	for _, i := range rows {
		v := c.data[i]
		if c.isNA(i) || math.IsInf(v, 0) {
			continue
		}
		a := math.Abs(v)
		if a != 0 && a < 1e-6 {
			sci = true
		}
		if a > 1e6 {
			large = true
		}
		t := strings.TrimRight(strconv.FormatFloat(v, 'f', digits, 64), "0")
		dec = max(dec, len(t)-strings.IndexByte(t, '.')-1)
	}
	for _, i := range rows {
		if v := c.data[i]; !c.isNA(i) && !math.IsInf(v, 0) {
			// The width pandas measures includes the sign or its space.
			maxLen = max(maxLen, len(strconv.FormatFloat(math.Abs(v), 'f', dec, 64))+1)
		}
	}
	if large && maxLen > digits+6 {
		sci = true
	}
	for k, i := range rows {
		v := c.data[i]
		switch {
		case c.isNA(i):
			out[k] = "NaN"
		case math.IsInf(v, 1):
			out[k] = "inf"
		case math.IsInf(v, -1):
			out[k] = "-inf"
		case sci:
			out[k] = strconv.FormatFloat(v, 'e', digits, 64)
		default:
			out[k] = strconv.FormatFloat(v, 'f', dec, 64)
		}
	}
	return out
}

// shownRows returns the positions to show of n rows, and where the "..."
// goes (-1 for nowhere).
func shownRows(n int) ([]int, int) {
	var rows []int
	if n <= MaxRows {
		for i := range n {
			rows = append(rows, i)
		}
		return rows, -1
	}
	h := MinRows / 2
	for i := range h {
		rows = append(rows, i)
	}
	for i := n - h; i < n; i++ {
		rows = append(rows, i)
	}
	return rows, h
}

// shownCols does the same for columns.
func shownCols(n int) ([]int, int) {
	var cols []int
	if n <= MaxColumns {
		for i := range n {
			cols = append(cols, i)
		}
		return cols, -1
	}
	h := MaxColumns / 2
	for i := range h {
		cols = append(cols, i)
	}
	for i := n - h; i < n; i++ {
		cols = append(cols, i)
	}
	return cols, h
}

func width(s string) int { return utf8.RuneCountInString(s) }

func padLeft(s string, w int) string  { return strings.Repeat(" ", max(w-width(s), 0)) + s }
func padRight(s string, w int) string { return s + strings.Repeat(" ", max(w-width(s), 0)) }

// indexCells formats the index levels at rows; repeated outer labels of a
// multi-level index are blanked, as pandas sparsifies them.
func indexCells(ix *Index, rows []int) [][]string {
	levels := make([][]string, ix.NLevels())
	for k := range levels {
		levels[k] = formatColumn(ix.level(k), rows)
	}
	if len(levels) > 1 {
		for r := len(rows) - 1; r > 0; r-- {
			for k := 0; k < len(levels)-1; k++ {
				same := true
				for m := 0; m <= k; m++ {
					same = same && levels[m][r] == levels[m][r-1]
				}
				if same && rows[r] == rows[r-1]+1 {
					levels[k][r] = ""
				}
			}
		}
	}
	return levels
}

// String formats the frame as pandas prints it: every value cell gets a
// leading space (a negative number's sign takes its place), cells are
// right-aligned under right-aligned column names, columns are joined by one
// space, and the index is left-aligned.
func (df *DataFrame) String() string {
	nrows, ncols := df.Shape()
	if ncols == 0 {
		return fmt.Sprintf("Empty DataFrame\nColumns: []\nIndex: %s", df.index)
	}
	rows, rowGap := shownRows(nrows)
	cols, colGap := shownCols(ncols)
	idx := indexCells(df.index, rows)
	named := false
	for _, n := range df.index.names {
		named = named || n != ""
	}
	type outCol struct {
		head  string
		cells []string
		width int
		index bool
	}
	var oc []outCol
	for k, cells := range idx {
		c := outCol{cells: cells, index: true}
		if named {
			c.head = df.index.names[k]
		}
		oc = append(oc, c)
	}
	for pos, j := range cols {
		if pos == colGap {
			oc = append(oc, outCol{head: "...", cells: repeatString(" ...", len(rows))})
		}
		cells := formatColumn(df.cols[j], rows)
		dt := df.cols[j].dtype()
		for r, v := range cells {
			// pandas leaves room for a sign before every value except a
			// negative number's, and none before instants and periods.
			if dt != Datetime && dt != Period && (!strings.HasPrefix(v, "-") || !dt.IsNumeric()) {
				cells[r] = " " + v
			}
		}
		head := df.names[j]
		if df.cols[j].dtype().IsNumeric() {
			// pandas gives numeric headers the leading space their values have.
			head = " " + head
		}
		oc = append(oc, outCol{head: head, cells: cells})
	}
	for k := range oc {
		w := width(oc[k].head)
		for _, s := range oc[k].cells {
			w = max(w, width(s))
		}
		oc[k].width = w
	}
	var b strings.Builder
	line := func(cell func(c outCol) string) {
		parts := make([]string, len(oc))
		for k, c := range oc {
			parts[k] = cell(c)
		}
		b.WriteString(strings.Join(parts, " "))
		b.WriteByte('\n')
	}
	line(func(c outCol) string {
		if c.index {
			return strings.Repeat(" ", c.width)
		}
		return padLeft(c.head, c.width)
	})
	if named {
		line(func(c outCol) string {
			if c.index {
				return padRight(c.head, c.width)
			}
			return strings.Repeat(" ", c.width)
		})
	}
	for r := range rows {
		if r == rowGap {
			line(func(c outCol) string {
				if c.index {
					return padRight(dotsFor(c.width), c.width)
				}
				return padLeft(dotsFor(c.width), c.width)
			})
		}
		line(func(c outCol) string {
			if c.index {
				return padRight(c.cells[r], c.width)
			}
			return padLeft(c.cells[r], c.width)
		})
	}
	out := strings.TrimSuffix(b.String(), "\n")
	if rowGap >= 0 || colGap >= 0 {
		out += fmt.Sprintf("\n\n[%d rows x %d columns]", nrows, ncols)
	}
	return out
}

func repeatString(s string, n int) []string {
	out := make([]string, n)
	for i := range out {
		out[i] = s
	}
	return out
}

// HTML formats the frame as an HTML table, as pandas' notebook display.
func (df *DataFrame) HTML() string {
	nrows, ncols := df.Shape()
	rows, rowGap := shownRows(nrows)
	cols, colGap := shownCols(ncols)
	idx := indexCells(df.index, rows)
	var b strings.Builder
	b.WriteString("<div>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n")
	for range idx {
		b.WriteString("      <th></th>\n")
	}
	cells := make([][]string, len(cols))
	for pos, j := range cols {
		if pos == colGap {
			b.WriteString("      <th>...</th>\n")
		}
		fmt.Fprintf(&b, "      <th>%s</th>\n", html.EscapeString(df.names[j]))
		cells[pos] = formatColumn(df.cols[j], rows)
	}
	b.WriteString("    </tr>\n")
	named := false
	for _, n := range df.index.names {
		named = named || n != ""
	}
	if named {
		b.WriteString("    <tr>\n")
		for _, n := range df.index.names {
			fmt.Fprintf(&b, "      <th>%s</th>\n", html.EscapeString(n))
		}
		for range cells {
			b.WriteString("      <th></th>\n")
		}
		if colGap >= 0 {
			b.WriteString("      <th></th>\n")
		}
		b.WriteString("    </tr>\n")
	}
	b.WriteString("  </thead>\n  <tbody>\n")
	for r := range rows {
		if r == rowGap {
			b.WriteString("    <tr>\n")
			for range idx {
				b.WriteString("      <th>...</th>\n")
			}
			for pos := range cells {
				if pos == colGap {
					b.WriteString("      <td>...</td>\n")
				}
				b.WriteString("      <td>...</td>\n")
			}
			b.WriteString("    </tr>\n")
		}
		b.WriteString("    <tr>\n")
		for k := range idx {
			fmt.Fprintf(&b, "      <th>%s</th>\n", html.EscapeString(idx[k][r]))
		}
		for pos := range cells {
			if pos == colGap {
				b.WriteString("      <td>...</td>\n")
			}
			fmt.Fprintf(&b, "      <td>%s</td>\n", html.EscapeString(cells[pos][r]))
		}
		b.WriteString("    </tr>\n")
	}
	b.WriteString("  </tbody>\n</table>\n")
	if rowGap >= 0 || colGap >= 0 {
		fmt.Fprintf(&b, "<p>%d rows × %d columns</p>\n", nrows, ncols)
	}
	b.WriteString("</div>")
	return b.String()
}

// Display returns the frame for a notebook: text/plain as String, text/html
// as HTML.
func (df *DataFrame) Display() map[string][]byte {
	return map[string][]byte{
		"text/plain": []byte(df.String()),
		"text/html":  []byte(df.HTML()),
	}
}

// String formats the series as pandas prints it.
func (s *Series) String() string {
	n := s.Len()
	rows, gap := shownRows(n)
	idx := indexCells(s.index, rows)
	vals := formatColumn(s.c, rows)
	wi := make([]int, len(idx))
	for k, cells := range idx {
		for _, c := range cells {
			wi[k] = max(wi[k], width(c))
		}
		wi[k] = max(wi[k], width(s.index.names[k]))
	}
	wv := 0
	for _, v := range vals {
		wv = max(wv, width(v))
	}
	var b strings.Builder
	named := false
	for _, nm := range s.index.names {
		named = named || nm != ""
	}
	if named {
		parts := make([]string, len(idx))
		for k := range idx {
			parts[k] = padRight(s.index.names[k], wi[k])
		}
		b.WriteString(strings.TrimRight(strings.Join(parts, " "), " ") + "\n")
	}
	for r := range rows {
		if r == gap {
			parts := make([]string, len(idx))
			for k := range idx {
				parts[k] = padRight("...", wi[k])
			}
			b.WriteString(strings.Join(parts, " ") + "    " + padLeft("...", wv) + "\n")
		}
		parts := make([]string, len(idx))
		for k := range idx {
			parts[k] = padRight(idx[k][r], wi[k])
		}
		b.WriteString(strings.Join(parts, " ") + "    " + padLeft(vals[r], wv) + "\n")
	}
	var foot []string
	if s.name != "" {
		foot = append(foot, "Name: "+s.name)
	}
	if gap >= 0 {
		foot = append(foot, "Length: "+strconv.Itoa(n))
	}
	foot = append(foot, "dtype: "+s.DType().String())
	if n == 0 {
		return fmt.Sprintf("Series([], %s)", strings.Join(foot, ", "))
	}
	return b.String() + strings.Join(foot, ", ")
}

// HTML formats the series as a one-column HTML table.
func (s *Series) HTML() string { return s.ToFrame().HTML() }

// Display returns the series for a notebook.
func (s *Series) Display() map[string][]byte {
	return map[string][]byte{
		"text/plain": []byte(s.String()),
		"text/html":  []byte(s.HTML()),
	}
}

// Display returns the index for a notebook, as text.
func (ix *Index) Display() map[string][]byte {
	return map[string][]byte{"text/plain": []byte(ix.String())}
}

// dotsFor returns the ellipsis pandas puts in a truncated column of width w.
func dotsFor(w int) string {
	switch {
	case w > 3:
		return "..."
	case w > 1:
		return ".."
	}
	return "."
}
