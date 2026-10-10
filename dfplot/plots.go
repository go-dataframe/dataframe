package dfplot

import (
	"math"

	"github.com/go-dataframe/dataframe"
	"github.com/go-plot/plot"
)

// Line draws one line per column against the index, as df.plot() does in
// pandas. Without column names every numeric column is drawn. The legend
// names the columns and the x label is the name of the index. A missing
// value breaks the line.
func Line(df *dataframe.DataFrame, columns ...string) (*plot.Figure, error) {
	return LineOpts(df, columns, nil)
}

// LineOpts is Line with options; the columns are given as a slice.
func LineOpts(df *dataframe.DataFrame, columns []string, opts []Option) (*plot.Figure, error) {
	cols, vals, err := pick(df, columns)
	if err != nil {
		return nil, err
	}
	return line(df.Index(), cols, vals, newConfig(true, opts)), nil
}

// SeriesLine draws a series against its index. It has no legend unless
// WithLegend(true) is given.
func SeriesLine(s *dataframe.Series, opts ...Option) (*plot.Figure, error) {
	vals, err := floats([]*dataframe.Series{s})
	if err != nil {
		return nil, err
	}
	return line(s.Index(), []*dataframe.Series{s}, vals, newConfig(false, opts)), nil
}

func line(ix *dataframe.Index, cols []*dataframe.Series, vals [][]float64, c *config) *plot.Figure {
	fig, ax := c.figure(ix.Name(), "")
	a := xaxis(ix)
	for i, s := range cols {
		ax.Plot(a.x, vals[i], plot.Label(s.Name()))
	}
	a.apply(ax)
	c.finish(ax, true)
	return fig
}

// Area draws the columns as stacked filled areas against the index, as
// df.plot.area() does; a missing value counts as zero.
func Area(df *dataframe.DataFrame, columns ...string) (*plot.Figure, error) {
	return AreaOpts(df, columns, nil)
}

// AreaOpts is Area with options.
func AreaOpts(df *dataframe.DataFrame, columns []string, opts []Option) (*plot.Figure, error) {
	cols, vals, err := pick(df, columns)
	if err != nil {
		return nil, err
	}
	c := newConfig(true, opts)
	fig, ax := c.figure(df.Index().Name(), "")
	a := xaxis(df.Index())
	base := make([]float64, df.Len())
	for i, s := range cols {
		top := make([]float64, len(base))
		for k := range top {
			top[k] = base[k]
			if v := vals[i][k]; !math.IsNaN(v) {
				top[k] += v
			}
		}
		ax.FillBetween(a.x, base, top, plot.Alpha(0.5), plot.Label(s.Name()))
		base = top
	}
	a.apply(ax)
	c.finish(ax, true)
	return fig, nil
}

// Bar draws grouped vertical bars, one group per index label and one bar per
// column, as df.plot.bar() does. The labels are the x ticks. A missing value
// leaves its bar out.
func Bar(df *dataframe.DataFrame, columns ...string) (*plot.Figure, error) {
	return BarOpts(df, columns, nil)
}

// BarOpts is Bar with options.
func BarOpts(df *dataframe.DataFrame, columns []string, opts []Option) (*plot.Figure, error) {
	cols, vals, err := pick(df, columns)
	if err != nil {
		return nil, err
	}
	return bars(df.Index(), cols, vals, newConfig(true, opts), false), nil
}

// Barh is Bar with horizontal bars: the index labels run up the y axis.
func Barh(df *dataframe.DataFrame, columns ...string) (*plot.Figure, error) {
	return BarhOpts(df, columns, nil)
}

// BarhOpts is Barh with options.
func BarhOpts(df *dataframe.DataFrame, columns []string, opts []Option) (*plot.Figure, error) {
	cols, vals, err := pick(df, columns)
	if err != nil {
		return nil, err
	}
	return bars(df.Index(), cols, vals, newConfig(true, opts), true), nil
}

// SeriesBar draws a series as one bar per index label.
func SeriesBar(s *dataframe.Series, opts ...Option) (*plot.Figure, error) {
	vals, err := floats([]*dataframe.Series{s})
	if err != nil {
		return nil, err
	}
	return bars(s.Index(), []*dataframe.Series{s}, vals, newConfig(false, opts), false), nil
}

func bars(ix *dataframe.Index, cols []*dataframe.Series, vals [][]float64, c *config, horizontal bool) *plot.Figure {
	var fig *plot.Figure
	var ax *plot.Axes
	if horizontal {
		fig, ax = c.figure("", ix.Name())
	} else {
		fig, ax = c.figure(ix.Name(), "")
	}
	n := ix.Len()
	pos := positions(n)
	width := 0.8 / float64(len(cols))
	for i, s := range cols {
		off := (float64(i) - float64(len(cols)-1)/2) * width
		var at, h []float64
		for k, v := range vals[i] {
			if !math.IsNaN(v) {
				at = append(at, pos[k]+off)
				h = append(h, v)
			}
		}
		if horizontal {
			ax.Barh(at, h, plot.Width(width), plot.Label(s.Name()))
		} else {
			ax.Bar(at, h, plot.Width(width), plot.Label(s.Name()))
		}
	}
	a := positionAxis(ix, n)
	if horizontal {
		ax.SetYTicks(a.ticks, a.labels...)
	} else {
		ax.SetXTicks(a.ticks, a.labels...)
	}
	c.finish(ax, true)
	return fig
}

// Hist draws one histogram per column, overlaid with transparency and sharing
// the same bins, as df.plot.hist() does. bins is the number of bins (10 when
// it is not positive). The y label is "Frequency".
func Hist(df *dataframe.DataFrame, bins int, columns ...string) (*plot.Figure, error) {
	return HistOpts(df, bins, columns, nil)
}

// HistOpts is Hist with options.
func HistOpts(df *dataframe.DataFrame, bins int, columns []string, opts []Option) (*plot.Figure, error) {
	cols, vals, err := pick(df, columns)
	if err != nil {
		return nil, err
	}
	return hist(cols, vals, bins, newConfig(true, opts)), nil
}

// SeriesHist draws the histogram of a series.
func SeriesHist(s *dataframe.Series, bins int, opts ...Option) (*plot.Figure, error) {
	vals, err := floats([]*dataframe.Series{s})
	if err != nil {
		return nil, err
	}
	return hist([]*dataframe.Series{s}, vals, bins, newConfig(false, opts)), nil
}

func hist(cols []*dataframe.Series, vals [][]float64, bins int, c *config) *plot.Figure {
	fig, ax := c.figure("", "Frequency")
	if bins <= 0 {
		bins = 10
	}
	lo, hi := math.Inf(1), math.Inf(-1)
	clean := make([][]float64, len(vals))
	for i, col := range vals {
		for _, v := range col {
			if !math.IsNaN(v) && !math.IsInf(v, 0) {
				clean[i] = append(clean[i], v)
				lo, hi = math.Min(lo, v), math.Max(hi, v)
			}
		}
	}
	switch {
	case lo > hi:
		lo, hi = 0, 1
	case lo == hi:
		lo, hi = lo-0.5, hi+0.5
	}
	edges := make([]float64, bins+1)
	for k := range edges {
		edges[k] = lo + (hi-lo)*float64(k)/float64(bins)
	}
	for i, s := range cols {
		ax.Hist(clean[i], plot.BinEdges(edges...), plot.Alpha(0.5), plot.Label(s.Name()))
	}
	c.finish(ax, true)
	return fig
}

// Scatter draws the column y against the column x, with both columns as the
// axis labels. Rows where either value is missing are skipped.
func Scatter(df *dataframe.DataFrame, x, y string, opts ...Option) (*plot.Figure, error) {
	if df == nil {
		return nil, ErrNoData
	}
	sx, err := df.Column(x)
	if err != nil {
		return nil, err
	}
	sy, err := df.Column(y)
	if err != nil {
		return nil, err
	}
	vals, err := floats([]*dataframe.Series{sx, sy})
	if err != nil {
		return nil, err
	}
	var xs, ys []float64
	for i := range vals[0] {
		if !math.IsNaN(vals[0][i]) && !math.IsNaN(vals[1][i]) {
			xs, ys = append(xs, vals[0][i]), append(ys, vals[1][i])
		}
	}
	c := newConfig(false, opts)
	fig, ax := c.figure(x, y)
	ax.Scatter(xs, ys, plot.Label(y))
	c.finish(ax, true)
	return fig, nil
}
