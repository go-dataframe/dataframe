package dataframe

import (
	"fmt"
	"math"
)

// Shift returns the series moved down by n rows (up for a negative n), the
// vacated rows missing (pandas' shift).
func (s *Series) Shift(n int) *Series { return s.with(shiftCol(s.c, n)) }

func shiftCol(c column, n int) column {
	idx := make([]int, c.len())
	for i := range idx {
		j := i - n
		if j < 0 || j >= len(idx) {
			j = -1
		}
		idx[i] = j
	}
	return c.take(idx)
}

// Shift shifts every column, as Series.Shift.
func (df *DataFrame) Shift(n int) *DataFrame {
	return df.mapCols(func(c column) column { return shiftCol(c, n) })
}

// Diff returns s minus s shifted by n (pandas' diff), as float64.
func (s *Series) Diff(n int) (*Series, error) {
	f, err := s.Floats()
	if err != nil {
		return nil, err
	}
	out := makeCol[float64](Float64, len(f))
	for i := range f {
		out.data[i] = math.NaN()
		if j := i - n; j >= 0 && j < len(f) {
			out.data[i] = f[i] - f[j]
		}
	}
	return s.with(out), nil
}

// PctChange returns the relative change from n rows before (pandas'
// pct_change, without filling missing values).
func (s *Series) PctChange(n int) (*Series, error) {
	f, err := s.Floats()
	if err != nil {
		return nil, err
	}
	out := makeCol[float64](Float64, len(f))
	for i := range f {
		out.data[i] = math.NaN()
		if j := i - n; j >= 0 && j < len(f) {
			out.data[i] = f[i]/f[j] - 1
		}
	}
	return s.with(out), nil
}

// cumulate runs an accumulation over the present values; missing values
// stay missing and do not interrupt it (pandas' skipna).
func (s *Series) cumulate(init float64, f func(acc, v float64) float64) (*Series, error) {
	x, err := s.Floats()
	if err != nil {
		return nil, err
	}
	out := makeCol[float64](Float64, len(x))
	acc := init
	for i, v := range x {
		if math.IsNaN(v) {
			out.data[i] = math.NaN()
			continue
		}
		acc = f(acc, v)
		out.data[i] = acc
	}
	return s.with(out), nil
}

// CumSum returns the running sum, as float64.
func (s *Series) CumSum() (*Series, error) {
	return s.cumulate(0, func(a, v float64) float64 { return a + v })
}

// CumProd returns the running product, as float64.
func (s *Series) CumProd() (*Series, error) {
	return s.cumulate(1, func(a, v float64) float64 { return a * v })
}

// CumMax returns the running maximum, as float64.
func (s *Series) CumMax() (*Series, error) {
	return s.cumulate(math.Inf(-1), math.Max)
}

// CumMin returns the running minimum, as float64.
func (s *Series) CumMin() (*Series, error) {
	return s.cumulate(math.Inf(1), math.Min)
}

// Rolling is a moving window over a series, ready to aggregate.
type Rolling struct {
	s          *Series
	window     int
	minPeriods int
}

// Rolling returns a window of the given number of rows ending at each row;
// a row gets a value once the window holds window present values (pandas'
// default min_periods).
func (s *Series) Rolling(window int) (*Rolling, error) {
	return s.RollingMin(window, window)
}

// RollingMin is Rolling with min_periods: the least number of present
// values a window needs to give a value.
func (s *Series) RollingMin(window, minPeriods int) (*Rolling, error) {
	if window < 1 || minPeriods < 1 || minPeriods > window {
		return nil, fmt.Errorf("%w: window %d, min periods %d", ErrArgument, window, minPeriods)
	}
	if _, err := s.Floats(); err != nil {
		return nil, err
	}
	return &Rolling{s, window, minPeriods}, nil
}

func (r *Rolling) apply(f func(vals []float64) float64) *Series {
	x, _ := r.s.Floats()
	out := makeCol[float64](Float64, len(x))
	buf := make([]float64, 0, r.window)
	for i := range x {
		buf = buf[:0]
		for j := max(0, i-r.window+1); j <= i; j++ {
			if !math.IsNaN(x[j]) {
				buf = append(buf, x[j])
			}
		}
		out.data[i] = math.NaN()
		if len(buf) >= r.minPeriods {
			out.data[i] = f(buf)
		}
	}
	return r.s.with(out)
}

// Sum returns the moving sum.
func (r *Rolling) Sum() *Series { return r.apply(sumFloat) }

// Mean returns the moving mean.
func (r *Rolling) Mean() *Series { return r.apply(meanOf) }

// Std returns the moving sample standard deviation.
func (r *Rolling) Std() *Series {
	return r.apply(func(v []float64) float64 { return math.Sqrt(varOf(v, 1)) })
}

// Min returns the moving minimum.
func (r *Rolling) Min() *Series {
	return r.apply(func(v []float64) float64 { return quantileOf(v, 0) })
}

// Max returns the moving maximum.
func (r *Rolling) Max() *Series {
	return r.apply(func(v []float64) float64 { return quantileOf(v, 1) })
}

// Median returns the moving median.
func (r *Rolling) Median() *Series {
	return r.apply(func(v []float64) float64 { return quantileOf(v, 0.5) })
}
