// Package dfplot draws DataFrames and Series, in the manner of the plot
// helpers of pandas (df.plot.line, df.plot.bar, ...) that notebooks lean on.
//
// Every function returns a *plot.Figure from github.com/go-plot/plot, which
// can be displayed in a notebook (Figure.Display) or saved (SavePNG,
// SaveSVG). The index of the frame is the x axis:
//
//   - a numeric index gives the x values themselves;
//   - a datetime index gives numeric x values (days since the Unix epoch)
//     with readable date tick labels;
//   - any other index (strings, categories, several levels) places the
//     points at positions 0, 1, 2, ... and shows the labels as ticks.
//
// Missing values are not drawn: a line is broken where a value is missing
// (as matplotlib breaks it at a NaN), a bar is left out, and a missing value
// does not enter a histogram.
package dfplot

import (
	"errors"
	"fmt"
	"math"
	"strconv"
	"time"

	"github.com/go-dataframe/dataframe"
	"github.com/go-ndarray/ndarray"
	"github.com/go-plot/plot"
)

// ErrNoData reports a plot with nothing to draw: a nil frame, or a frame
// with no numeric column when none was named.
var ErrNoData = errors.New("dfplot: no data to plot")

// An Option adjusts a plot.
type Option func(*config)

type config struct {
	title, xlabel, ylabel string
	setX, setY            bool
	width, height         float64
	legend                bool
	grid                  bool
}

// WithTitle sets the title of the axes.
func WithTitle(s string) Option { return func(c *config) { c.title = s } }

// WithXLabel sets the label of the x axis; by default it is the name of the
// index.
func WithXLabel(s string) Option { return func(c *config) { c.xlabel, c.setX = s, true } }

// WithYLabel sets the label of the y axis.
func WithYLabel(s string) Option { return func(c *config) { c.ylabel, c.setY = s, true } }

// WithSize sets the figure size in inches (the default is 6.4 x 4.8).
func WithSize(width, height float64) Option {
	return func(c *config) { c.width, c.height = width, height }
}

// WithLegend turns the legend on or off. A frame plot has one by default, a
// series plot has none.
func WithLegend(on bool) Option { return func(c *config) { c.legend = on } }

// WithGrid turns the grid on or off (off by default).
func WithGrid(on bool) Option { return func(c *config) { c.grid = on } }

func newConfig(legend bool, opts []Option) *config {
	c := &config{legend: legend}
	for _, o := range opts {
		o(c)
	}
	return c
}

// figure makes the figure and its one Axes, with the title and the labels
// already set. xlabel and ylabel are the defaults, used unless an option
// replaced them.
func (c *config) figure(xlabel, ylabel string) (*plot.Figure, *plot.Axes) {
	var fo []plot.FigOption
	if c.width > 0 && c.height > 0 {
		fo = append(fo, plot.FigSize(c.width, c.height))
	}
	fig, axs := plot.Subplots(1, 1, fo...)
	ax := axs[0][0]
	if !c.setX {
		c.xlabel = xlabel
	}
	if !c.setY {
		c.ylabel = ylabel
	}
	ax.SetTitle(c.title)
	ax.SetXLabel(c.xlabel)
	ax.SetYLabel(c.ylabel)
	if c.grid {
		ax.Grid(true)
	}
	return fig, ax
}

// finish adds the legend when it is wanted and anything carries a label.
func (c *config) finish(ax *plot.Axes, labelled bool) {
	if c.legend && labelled {
		ax.Legend()
	}
}

// numeric reports whether a column is picked when none is named: real
// numbers, not bool or complex.
func numeric(d dataframe.DType) bool {
	return d.IsNumeric() && d != dataframe.Bool && d != dataframe.Complex64 && d != dataframe.Complex128
}

// pick returns the named columns of df, or its numeric columns when none is
// named, with their values.
func pick(df *dataframe.DataFrame, names []string) ([]*dataframe.Series, [][]float64, error) {
	if df == nil {
		return nil, nil, fmt.Errorf("%w: nil frame", ErrNoData)
	}
	var cols []*dataframe.Series
	if len(names) == 0 {
		for _, s := range df.Series() {
			if numeric(s.DType()) {
				cols = append(cols, s)
			}
		}
	}
	for _, n := range names {
		s, err := df.Column(n)
		if err != nil {
			return nil, nil, err
		}
		cols = append(cols, s)
	}
	if len(cols) == 0 {
		return nil, nil, fmt.Errorf("%w: the frame has no numeric column", ErrNoData)
	}
	vals, err := floats(cols)
	return cols, vals, err
}

func floats(cols []*dataframe.Series) ([][]float64, error) {
	vals := make([][]float64, len(cols))
	for i, s := range cols {
		f, err := s.Floats()
		if err != nil {
			return nil, fmt.Errorf("dfplot: column %q: %w", s.Name(), err)
		}
		vals[i] = f
	}
	return vals, nil
}

// axisX is the x coordinate of every row, and the ticks that label them
// (none when the axis numbers itself).
type axisX struct {
	x      []float64
	ticks  []float64
	labels []string
}

const (
	day      = 24 * float64(time.Hour)
	maxTicks = 10
)

// xaxis derives the x coordinates from the index.
func xaxis(ix *dataframe.Index) axisX {
	n := ix.Len()
	if ix.NLevels() == 1 {
		lvl, _ := ix.Level(0) // level 0 of a one-level index always exists
		switch dt := ix.DType(); {
		case dt == dataframe.Datetime:
			return datetimeAxis(lvl)
		case numeric(dt):
			f, _ := lvl.Floats() // numeric: cannot fail
			return axisX{x: f}
		}
	}
	return positionAxis(ix, n)
}

func positionAxis(ix *dataframe.Index, n int) axisX {
	a := axisX{x: make([]float64, n)}
	step := (n + maxTicks - 1) / maxTicks
	for i := 0; i < n; i++ {
		a.x[i] = float64(i)
		if step > 0 && i%step == 0 {
			a.ticks = append(a.ticks, float64(i))
			a.labels = append(a.labels, labelText(ix.Label(i)))
		}
	}
	return a
}

func datetimeAxis(lvl *dataframe.Series) axisX {
	arr, _ := lvl.Values() // a datetime series always has an array form
	ns, _ := ndarray.Data[int64](arr)
	return datetimeFrom(ns, lvl.Valid())
}

func datetimeFrom(ns []int64, valid []bool) axisX {
	a := axisX{x: make([]float64, len(ns))}
	lo, hi := math.Inf(1), math.Inf(-1)
	for i, v := range ns {
		if !valid[i] {
			a.x[i] = math.NaN()
			continue
		}
		a.x[i] = float64(v) / day
		lo, hi = math.Min(lo, a.x[i]), math.Max(hi, a.x[i])
	}
	if lo > hi {
		return a
	}
	layout := dateLayout((hi - lo) * day)
	nt := 6
	for k := 0; k < nt; k++ {
		t := lo + (hi-lo)*float64(k)/float64(nt-1)
		a.ticks = append(a.ticks, t)
		a.labels = append(a.labels, time.Unix(0, int64(math.Round(t*day))).UTC().Format(layout))
		if hi == lo {
			break
		}
	}
	return a
}

// dateLayout picks the tick format for a time span in nanoseconds.
func dateLayout(span float64) string {
	switch {
	case span < 2*day:
		return "Jan 02 15:04"
	case span < 2*365*day:
		return "2006-01-02"
	}
	return "2006-01"
}

// labelText formats an index label.
func labelText(v any) string {
	switch t := v.(type) {
	case nil:
		return "NA"
	case time.Time:
		if t.Hour() == 0 && t.Minute() == 0 && t.Second() == 0 && t.Nanosecond() == 0 {
			return t.Format("2006-01-02")
		}
		return t.Format("2006-01-02 15:04:05")
	case float64:
		return strconv.FormatFloat(t, 'g', -1, 64)
	}
	return fmt.Sprint(v)
}

func positions(n int) []float64 {
	p := make([]float64, n)
	for i := range p {
		p[i] = float64(i)
	}
	return p
}

func (a axisX) apply(ax *plot.Axes) {
	if a.ticks != nil {
		ax.SetXTicks(a.ticks, a.labels...)
	}
}
