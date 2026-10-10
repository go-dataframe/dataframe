package dataframe

import (
	"fmt"
	"slices"
)

// IsNA returns a frame of bools, true where a value is missing.
func (df *DataFrame) IsNA() *DataFrame {
	cols := make([]column, len(df.cols))
	for k, s := range df.Series() {
		cols[k] = s.IsNA().c
	}
	return df.derive(cols)
}

// NotNA is the negation of IsNA.
func (df *DataFrame) NotNA() *DataFrame {
	cols := make([]column, len(df.cols))
	for k, s := range df.Series() {
		cols[k] = s.NotNA().c
	}
	return df.derive(cols)
}

// DropNA returns the rows with no missing value in the subset columns (all
// columns when none are named): pandas' dropna().
func (df *DataFrame) DropNA(subset ...string) (*DataFrame, error) {
	return df.dropNA(false, 0, subset)
}

// DropNAAll returns the rows that have at least one value present in the
// subset columns: pandas' dropna(how="all").
func (df *DataFrame) DropNAAll(subset ...string) (*DataFrame, error) {
	return df.dropNA(true, 0, subset)
}

// DropNAThresh keeps the rows with at least thresh values present in the
// subset columns: pandas' dropna(thresh=thresh).
func (df *DataFrame) DropNAThresh(thresh int, subset ...string) (*DataFrame, error) {
	return df.dropNA(false, thresh, subset)
}

func (df *DataFrame) dropNA(all bool, thresh int, subset []string) (*DataFrame, error) {
	cols := df.cols
	if len(subset) > 0 {
		sel, err := df.Select(subset...)
		if err != nil {
			return nil, err
		}
		cols = sel.cols
	}
	var keep []int
	for i := range df.Len() {
		present := 0
		for _, c := range cols {
			if !c.isNA(i) {
				present++
			}
		}
		ok := present == len(cols)
		switch {
		case thresh > 0:
			ok = present >= thresh
		case all:
			ok = present > 0
		}
		if ok {
			keep = append(keep, i)
		}
	}
	if len(keep) == df.Len() {
		return df, nil
	}
	return df.take(keep), nil
}

// DropNAColumns returns the frame without the columns that have a missing
// value: pandas' dropna(axis=1).
func (df *DataFrame) DropNAColumns() *DataFrame {
	out := &DataFrame{index: df.index}
	for k, c := range df.cols {
		if c.naCount() == 0 {
			out.names = append(out.names, df.names[k])
			out.cols = append(out.cols, c)
		}
	}
	return out
}

// DropNA returns the present values.
func (s *Series) DropNA() *Series {
	var keep []int
	for i := range s.Len() {
		if !s.c.isNA(i) {
			keep = append(keep, i)
		}
	}
	if len(keep) == s.Len() {
		return s
	}
	return s.take(keep)
}

// FillNA returns the series with missing values replaced by v, which must
// convert to the series' dtype (an int fills a float series; a float fills
// an int series only when it is integral).
func (s *Series) FillNA(v any) (*Series, error) {
	c, err := s.c.fill(v)
	if err != nil {
		return nil, err
	}
	return s.with(c), nil
}

// FillNA returns the frame with missing values replaced by v in every column
// whose dtype v converts to; the other columns are left as they are.
func (df *DataFrame) FillNA(v any) *DataFrame {
	cols := slices.Clone(df.cols)
	for k, c := range cols {
		if f, err := c.fill(v); err == nil {
			cols[k] = f
		}
	}
	return df.derive(cols)
}

// FillNAColumns fills missing values column by column, from a map of column
// name to value.
func (df *DataFrame) FillNAColumns(values map[string]any) (*DataFrame, error) {
	cols := slices.Clone(df.cols)
	for name, v := range values {
		k, err := df.pos(name)
		if err != nil {
			return nil, err
		}
		if cols[k], err = cols[k].fill(v); err != nil {
			return nil, fmt.Errorf("column %q: %w", name, err)
		}
	}
	return df.derive(cols), nil
}

// FFill returns the series with each missing value replaced by the last
// present one before it (pandas' ffill); leading missing values stay.
func (s *Series) FFill() *Series { return s.with(propagate(s.c, false)) }

// BFill replaces each missing value by the next present one (pandas' bfill).
func (s *Series) BFill() *Series { return s.with(propagate(s.c, true)) }

// FFill forward-fills every column, as Series.FFill.
func (df *DataFrame) FFill() *DataFrame {
	return df.mapCols(func(c column) column { return propagate(c, false) })
}

// BFill backward-fills every column, as Series.BFill.
func (df *DataFrame) BFill() *DataFrame {
	return df.mapCols(func(c column) column { return propagate(c, true) })
}

func (df *DataFrame) mapCols(f func(column) column) *DataFrame {
	cols := make([]column, len(df.cols))
	for k, c := range df.cols {
		cols[k] = f(c)
	}
	return df.derive(cols)
}

// propagate fills missing values from the previous (or next) present one.
func propagate(c column, backward bool) column {
	n := c.len()
	idx := make([]int, n)
	last := -1
	for k := range n {
		i := k
		if backward {
			i = n - 1 - k
		}
		if !c.isNA(i) {
			last = i
		}
		idx[i] = last
	}
	return c.take(idx)
}
