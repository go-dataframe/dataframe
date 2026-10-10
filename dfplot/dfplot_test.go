package dfplot

import (
	"bytes"
	"errors"
	"image/png"
	"math"
	"strings"
	"testing"
	"time"

	"github.com/go-dataframe/dataframe"
	"github.com/go-plot/plot"
	"github.com/go-plot/plot/text"
)

// recorder wraps the figure's text renderer and remembers every string laid
// out, which is how the tests see titles, labels, ticks and legend entries
// (the SVG holds glyph outlines, not text).
type recorder struct {
	text.Renderer
	seen map[string]bool
}

func (r *recorder) Layout(s string, px float64) text.Layout {
	r.seen[s] = true
	return r.Renderer.Layout(s, px)
}

// render draws the figure to SVG and PNG in memory, checks both are sound and
// returns the strings that were drawn.
func render(t *testing.T, fig *plot.Figure) map[string]bool {
	t.Helper()
	rec := &recorder{Renderer: fig.Text, seen: map[string]bool{}}
	fig.Text = rec
	var svg, pn bytes.Buffer
	if err := fig.SaveSVG(&svg); err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(svg.String(), "<?xml") || !strings.Contains(svg.String(), "</svg>") {
		t.Fatalf("not an SVG: %.60q", svg.String())
	}
	if err := fig.SavePNG(&pn); err != nil {
		t.Fatal(err)
	}
	img, err := png.Decode(&pn)
	if err != nil {
		t.Fatal(err)
	}
	w, h := fig.PixelSize()
	if img.Bounds().Dx() != w || img.Bounds().Dy() != h {
		t.Fatalf("png is %v, want %dx%d", img.Bounds(), w, h)
	}
	return rec.seen
}

func want(t *testing.T, seen map[string]bool, present, absent []string) {
	t.Helper()
	for _, s := range present {
		if !seen[s] {
			t.Errorf("%q was not drawn; drew %v", s, keys(seen))
		}
	}
	for _, s := range absent {
		if seen[s] {
			t.Errorf("%q was drawn", s)
		}
	}
}

func keys(m map[string]bool) []string {
	var k []string
	for s := range m {
		k = append(k, s)
	}
	return k
}

func frame(t *testing.T, cols ...dataframe.Column) *dataframe.DataFrame {
	t.Helper()
	df, err := dataframe.New(cols...)
	if err != nil {
		t.Fatal(err)
	}
	return df
}

func indexed(t *testing.T, df *dataframe.DataFrame, name string, data any) *dataframe.DataFrame {
	t.Helper()
	ix, err := dataframe.NewIndex(name, data)
	if err != nil {
		t.Fatal(err)
	}
	out, err := df.WithIndex(ix)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func base(t *testing.T) *dataframe.DataFrame {
	return frame(t,
		dataframe.Col("alpha", []float64{1, 2, 3, 4}),
		dataframe.Col("beta", []int64{4, 3, 2, 1}),
		dataframe.Col("name", []string{"a", "b", "c", "d"}),
		dataframe.Col("flag", []bool{true, false, true, false}),
		dataframe.Col("z", []complex128{1, 2, 3, 4}),
	)
}

func jan(d int) time.Time { return time.Date(2024, 1, d, 0, 0, 0, 0, time.UTC) }

func TestLineNumericIndex(t *testing.T) {
	df := indexed(t, base(t), "t", []float64{0.5, 1, 1.5, 2})
	fig, err := Line(df)
	if err != nil {
		t.Fatal(err)
	}
	if n := len(fig.Axes()); n != 1 {
		t.Fatalf("%d axes", n)
	}
	// Only alpha and beta: name is text, flag is bool, z is complex.
	want(t, render(t, fig), []string{"alpha", "beta", "t"}, []string{"name", "flag", "z"})
}

func TestLineNamedColumnsAndBool(t *testing.T) {
	fig, err := Line(base(t), "flag")
	if err != nil {
		t.Fatal(err)
	}
	want(t, render(t, fig), []string{"flag"}, []string{"alpha", "beta"})
}

func TestLineOptions(t *testing.T) {
	fig, err := LineOpts(base(t), []string{"alpha"}, []Option{
		WithTitle("My title"), WithXLabel("X!"), WithYLabel("Y!"),
		WithSize(4, 3), WithLegend(false), WithGrid(true),
	})
	if err != nil {
		t.Fatal(err)
	}
	if w, h := fig.PixelSize(); w != 400 || h != 300 {
		t.Fatalf("size %dx%d", w, h)
	}
	want(t, render(t, fig), []string{"My title", "X!", "Y!"}, []string{"alpha"})
}

func TestLineMissingValuesBreakTheLine(t *testing.T) {
	gap := frame(t, dataframe.Col("v", []float64{1, 2, math.NaN(), 4, 5}))
	solid := frame(t, dataframe.Col("v", []float64{1, 2, 3, 4, 5}))
	a, err := Line(gap)
	if err != nil {
		t.Fatal(err)
	}
	b, _ := Line(solid)
	var sa, sb bytes.Buffer
	_ = a.SaveSVG(&sa)
	_ = b.SaveSVG(&sb)
	if bytes.Equal(sa.Bytes(), sb.Bytes()) {
		t.Fatal("a missing value changed nothing")
	}
	render(t, a)
}

func TestLineDatetimeIndex(t *testing.T) {
	vals := []float64{1, 2, 3}
	cases := []struct {
		name  string
		times []time.Time
		tick  string
	}{
		{"days", []time.Time{jan(1), jan(3), jan(5)}, "2024-01-01"},
		{"hours", []time.Time{jan(1), jan(1).Add(6 * time.Hour), jan(1).Add(12 * time.Hour)}, "Jan 01 00:00"},
		{"years", []time.Time{jan(1), jan(1).AddDate(2, 0, 0), jan(1).AddDate(4, 0, 0)}, "2024-01"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			df := indexed(t, frame(t, dataframe.Col("v", vals)), "when", c.times)
			fig, err := Line(df)
			if err != nil {
				t.Fatal(err)
			}
			want(t, render(t, fig), []string{c.tick, "when", "v"}, nil)
		})
	}
}

func TestLineDatetimeEdgeCases(t *testing.T) {
	col := dataframe.Col("v", []float64{1, 2, 3})
	// A missing instant is not drawn; the others still label the axis.
	df := indexed(t, frame(t, col), "when", []any{jan(1), nil, jan(3)})
	fig, err := Line(df)
	if err != nil {
		t.Fatal(err)
	}
	want(t, render(t, fig), []string{"2024-01-01"}, nil)
	// One distinct instant: a single tick.
	df = indexed(t, frame(t, col), "when", []time.Time{jan(2), jan(2), jan(2)})
	if fig, err = Line(df); err != nil {
		t.Fatal(err)
	}
	want(t, render(t, fig), []string{"Jan 02 00:00"}, nil)
	// No instant at all: nothing to label.
	a := datetimeFrom([]int64{0, 0}, []bool{false, false})
	if a.ticks != nil || !math.IsNaN(a.x[0]) {
		t.Fatalf("got %+v", a)
	}
}

func TestLineStringIndex(t *testing.T) {
	df := indexed(t, base(t), "who", []string{"ann", "bob", "cy", "di"})
	fig, err := Line(df, "alpha")
	if err != nil {
		t.Fatal(err)
	}
	want(t, render(t, fig), []string{"ann", "bob", "cy", "di", "who"}, nil)
}

func TestLineManyLabelsAreThinned(t *testing.T) {
	n := 25
	names := make([]string, n)
	v := make([]float64, n)
	for i := range names {
		names[i] = string(rune('a'+i%26)) + string(rune('A'+i%26))
		v[i] = float64(i)
	}
	df := indexed(t, frame(t, dataframe.Col("v", v)), "k", names)
	fig, err := Line(df)
	if err != nil {
		t.Fatal(err)
	}
	seen := render(t, fig)
	want(t, seen, []string{"aA"}, []string{"bB"}) // every third label only
}

func TestLineEmptyAndSingleRow(t *testing.T) {
	empty := frame(t, dataframe.Col("v", []float64{}))
	fig, err := Line(empty)
	if err != nil {
		t.Fatal(err)
	}
	render(t, fig)
	one := frame(t, dataframe.Col("v", []float64{3}))
	if fig, err = Line(one); err != nil {
		t.Fatal(err)
	}
	render(t, fig)
}

func TestLineMultiLevelIndexUsesPositions(t *testing.T) {
	df := frame(t,
		dataframe.Col("g", []string{"x", "x", "y"}),
		dataframe.Col("h", []int64{1, 2, 1}),
		dataframe.Col("v", []float64{1, 2, 3}),
	)
	df, err := df.SetIndex("g", "h")
	if err != nil {
		t.Fatal(err)
	}
	fig, err := Line(df)
	if err != nil {
		t.Fatal(err)
	}
	want(t, render(t, fig), []string{"v"}, nil)
}

func TestSeriesLine(t *testing.T) {
	s, _ := dataframe.NewSeries("temp", []float64{1, 3, 2})
	fig, err := SeriesLine(s)
	if err != nil {
		t.Fatal(err)
	}
	want(t, render(t, fig), nil, []string{"temp"}) // no legend for a series
	if fig, err = SeriesLine(s, WithLegend(true)); err != nil {
		t.Fatal(err)
	}
	want(t, render(t, fig), []string{"temp"}, nil)
}

func TestArea(t *testing.T) {
	df := frame(t,
		dataframe.Col("alpha", []float64{1, 2, math.NaN()}),
		dataframe.Col("beta", []float64{3, 2, 1}),
	)
	fig, err := Area(df)
	if err != nil {
		t.Fatal(err)
	}
	want(t, render(t, fig), []string{"alpha", "beta"}, nil)
	fig, err = AreaOpts(df, []string{"beta"}, []Option{WithTitle("stack")})
	if err != nil {
		t.Fatal(err)
	}
	want(t, render(t, fig), []string{"stack", "beta"}, []string{"alpha"})
}

func TestBar(t *testing.T) {
	df := indexed(t, base(t), "who", []string{"ann", "bob", "cy", "di"})
	fig, err := Bar(df, "alpha", "beta")
	if err != nil {
		t.Fatal(err)
	}
	want(t, render(t, fig), []string{"alpha", "beta", "ann", "di", "who"}, nil)
}

func TestBarLabelKinds(t *testing.T) {
	v := dataframe.Col("v", []float64{1, 2})
	cases := []struct {
		data any
		tick string
	}{
		{[]float64{1.5, 2.5}, "1.5"},
		{[]int64{7, 8}, "7"},
		{[]time.Time{jan(1), jan(2).Add(time.Hour)}, "2024-01-02 01:00:00"},
		{[]time.Time{jan(1), jan(2)}, "2024-01-01"},
		{[]any{"p", nil}, "NA"},
	}
	for _, c := range cases {
		fig, err := Bar(indexed(t, frame(t, v), "k", c.data))
		if err != nil {
			t.Fatal(err)
		}
		want(t, render(t, fig), []string{c.tick}, nil)
	}
}

func TestBarMissingValues(t *testing.T) {
	df := frame(t, dataframe.Col("a", []float64{math.NaN(), math.NaN()}), dataframe.Col("b", []float64{1, math.NaN()}))
	fig, err := BarOpts(df, nil, []Option{WithYLabel("height")})
	if err != nil {
		t.Fatal(err)
	}
	want(t, render(t, fig), []string{"height"}, nil)
}

func TestBarh(t *testing.T) {
	df := indexed(t, base(t), "who", []string{"ann", "bob", "cy", "di"})
	fig, err := Barh(df, "alpha", "beta")
	if err != nil {
		t.Fatal(err)
	}
	want(t, render(t, fig), []string{"alpha", "beta", "ann", "who"}, nil)
	if fig, err = BarhOpts(df, []string{"alpha"}, []Option{WithXLabel("count")}); err != nil {
		t.Fatal(err)
	}
	want(t, render(t, fig), []string{"count", "alpha"}, []string{"beta"})
}

func TestSeriesBar(t *testing.T) {
	ix, _ := dataframe.NewIndex("k", []string{"u", "v"})
	s, _ := dataframe.NewSeriesIndexed("n", []int64{5, 6}, ix)
	fig, err := SeriesBar(s)
	if err != nil {
		t.Fatal(err)
	}
	want(t, render(t, fig), []string{"u", "v", "k"}, []string{"n"})
	if fig, err = SeriesBar(s, WithLegend(true)); err != nil {
		t.Fatal(err)
	}
	want(t, render(t, fig), []string{"n"}, nil)
}

func TestHist(t *testing.T) {
	df := frame(t,
		dataframe.Col("alpha", []float64{1, 2, 2, 3, 3, 3, math.NaN(), math.Inf(1)}),
		dataframe.Col("beta", []float64{0, 1, 1, 5, 5, 5, 5, 6}),
	)
	fig, err := Hist(df, 5)
	if err != nil {
		t.Fatal(err)
	}
	want(t, render(t, fig), []string{"alpha", "beta", "Frequency"}, nil)
	// Default bins, a constant column, then nothing but missing values.
	for _, vals := range [][]float64{{2, 2, 2}, {math.NaN()}} {
		fig, err = HistOpts(frame(t, dataframe.Col("c", vals)), 0, nil, []Option{WithXLabel("val")})
		if err != nil {
			t.Fatal(err)
		}
		want(t, render(t, fig), []string{"c", "val"}, nil)
	}
}

func TestSeriesHist(t *testing.T) {
	s, _ := dataframe.NewSeries("h", []float64{1, 2, 2, 3})
	fig, err := SeriesHist(s, 3)
	if err != nil {
		t.Fatal(err)
	}
	want(t, render(t, fig), []string{"Frequency"}, []string{"h"})
	if fig, err = SeriesHist(s, 3, WithLegend(true)); err != nil {
		t.Fatal(err)
	}
	want(t, render(t, fig), []string{"h"}, nil)
}

func TestScatter(t *testing.T) {
	df := frame(t,
		dataframe.Col("x", []float64{1, 2, math.NaN(), 4}),
		dataframe.Col("y", []int64{4, 3, 2, 1}),
	)
	fig, err := Scatter(df, "x", "y", WithTitle("pts"))
	if err != nil {
		t.Fatal(err)
	}
	want(t, render(t, fig), []string{"x", "y", "pts"}, nil)
	if fig, err = Scatter(df, "x", "y", WithLegend(true)); err != nil {
		t.Fatal(err)
	}
	want(t, render(t, fig), []string{"x", "y"}, nil)
}

func TestErrors(t *testing.T) {
	df := base(t)
	textOnly := frame(t, dataframe.Col("name", []string{"a"}))
	sText, _ := dataframe.NewSeries("s", []string{"a", "b"})
	checks := []struct {
		name string
		run  func() error
		is   error
	}{
		{"line unknown", func() error { _, e := Line(df, "nope"); return e }, dataframe.ErrColumn},
		{"line text", func() error { _, e := Line(df, "name"); return e }, dataframe.ErrDType},
		{"line complex", func() error { _, e := Line(df, "z"); return e }, dataframe.ErrDType},
		{"line nil", func() error { _, e := Line(nil); return e }, ErrNoData},
		{"line no numeric", func() error { _, e := Line(textOnly); return e }, ErrNoData},
		{"area", func() error { _, e := Area(df, "name"); return e }, dataframe.ErrDType},
		{"bar", func() error { _, e := Bar(df, "nope"); return e }, dataframe.ErrColumn},
		{"barh", func() error { _, e := Barh(textOnly); return e }, ErrNoData},
		{"hist", func() error { _, e := Hist(df, 3, "name"); return e }, dataframe.ErrDType},
		{"scatter nil", func() error { _, e := Scatter(nil, "x", "y"); return e }, ErrNoData},
		{"scatter x", func() error { _, e := Scatter(df, "nope", "alpha"); return e }, dataframe.ErrColumn},
		{"scatter y", func() error { _, e := Scatter(df, "alpha", "nope"); return e }, dataframe.ErrColumn},
		{"scatter text", func() error { _, e := Scatter(df, "alpha", "name"); return e }, dataframe.ErrDType},
		{"series line", func() error { _, e := SeriesLine(sText); return e }, dataframe.ErrDType},
		{"series bar", func() error { _, e := SeriesBar(sText); return e }, dataframe.ErrDType},
		{"series hist", func() error { _, e := SeriesHist(sText, 3); return e }, dataframe.ErrDType},
	}
	for _, c := range checks {
		if err := c.run(); !errors.Is(err, c.is) {
			t.Errorf("%s: got %v, want %v", c.name, err, c.is)
		}
	}
}

func TestDisplay(t *testing.T) {
	fig, err := Line(base(t), "alpha")
	if err != nil {
		t.Fatal(err)
	}
	d := fig.Display()
	if len(d["image/png"]) == 0 || len(d["image/svg+xml"]) == 0 {
		t.Fatal("empty display")
	}
}
