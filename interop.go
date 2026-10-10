package dataframe

import "fmt"

// This file holds the few entry points that columnar file formats (Arrow,
// Parquet) need to move data in and out of a series without boxing each
// value: a packed validity bitmap in both directions, and a categorical
// series as codes plus categories.

// NewSeriesBitmap is NewSeries with a packed validity bitmap: bit i%64 of
// word i/64 set means value i is present; a clear bit marks it missing (its
// slot in data is kept but ignored). This is the Arrow validity bit order
// read as little-endian 64-bit words. A nil bitmap means every value is
// present; otherwise it must hold at least one bit per value, and the bits
// past the last value are ignored. The bitmap is copied.
func NewSeriesBitmap(name string, data any, validity []uint64) (*Series, error) {
	s, err := NewSeries(name, data)
	if err != nil || validity == nil {
		return s, err
	}
	n := s.Len()
	if len(validity) < (n+63)>>6 {
		return nil, fmt.Errorf("%w: %d validity words for %d values", ErrLength, len(validity), n)
	}
	b := bitmap(append([]uint64(nil), validity[:(n+63)>>6]...))
	if b.count(n) < n {
		s.c = s.c.withValidity(b)
	}
	return s, nil
}

// ValidityBitmap returns a copy of the series' validity bitmap, in the
// layout NewSeriesBitmap takes, or nil when no value is masked. NaN in a
// float series is an in-band missing value, not a cleared bit: test floats
// for NaN as well.
func (s *Series) ValidityBitmap() []uint64 {
	return s.c.validity().clone()
}

// NewCategorical returns a category series from codes into categories: code
// k stands for categories[k], and a negative code marks a missing value. A
// code past the last category is an error.
func NewCategorical(name string, codes []int32, categories []string) (*Series, error) {
	for i, k := range codes {
		if int(k) >= len(categories) {
			return nil, fmt.Errorf("%w: code %d at %d for %d categories", ErrIndex, k, i, len(categories))
		}
	}
	c := catCol(codes, append([]string(nil), categories...))
	return &Series{name: name, index: NewRangeIndex(len(codes)), c: c}, nil
}

// Categorical returns a category series' codes (-1 where a value is missing)
// and its categories. Any other dtype is an ErrDType.
func (s *Series) Categorical() (codes []int32, categories []string, err error) {
	c, ok := s.c.(*col[int32])
	if !ok || c.dt != Category {
		return nil, nil, fmt.Errorf("%w: a %v series has no categories", ErrDType, s.DType())
	}
	codes = make([]int32, len(c.data))
	for i, k := range c.data {
		if !c.valid.get(i) {
			k = -1
		}
		codes[i] = k
	}
	return codes, append([]string(nil), c.levels...), nil
}
