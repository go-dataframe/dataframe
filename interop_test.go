package dataframe

import (
	"errors"
	"slices"
	"testing"
)

func TestNewSeriesBitmap(t *testing.T) {
	s, err := NewSeriesBitmap("x", []int64{1, 2, 3}, []uint64{0b101})
	if err != nil {
		t.Fatal(err)
	}
	if got := s.Valid(); !slices.Equal(got, []bool{true, false, true}) {
		t.Fatalf("valid = %v", got)
	}
	if got := s.ValidityBitmap(); !slices.Equal(got, []uint64{0b101}) {
		t.Fatalf("bitmap = %v", got)
	}
	// Bits past the last value are ignored; a full bitmap leaves no mask.
	s, _ = NewSeriesBitmap("x", []int64{1, 2}, []uint64{0b11 | 1<<40, 7})
	if s.ValidityBitmap() != nil || s.Count() != 2 {
		t.Fatalf("full bitmap kept a mask: %v", s.ValidityBitmap())
	}
	s, _ = NewSeriesBitmap("x", []float64{1}, nil)
	if s.ValidityBitmap() != nil {
		t.Fatal("nil bitmap made a mask")
	}
	if _, err := NewSeriesBitmap("x", make([]int8, 65), []uint64{0}); !errors.Is(err, ErrLength) {
		t.Fatalf("short bitmap: %v", err)
	}
	if _, err := NewSeriesBitmap("x", 3, nil); !errors.Is(err, ErrArgument) {
		t.Fatalf("bad data: %v", err)
	}
}

func TestCategorical(t *testing.T) {
	s, err := NewCategorical("c", []int32{1, -1, 0}, []string{"a", "b"})
	if err != nil {
		t.Fatal(err)
	}
	if s.DType() != Category || !slices.Equal(s.Strings(), []string{"b", "", "a"}) {
		t.Fatalf("got %v %v", s.DType(), s.Strings())
	}
	codes, cats, err := s.Categorical()
	if err != nil || !slices.Equal(codes, []int32{1, -1, 0}) || !slices.Equal(cats, []string{"a", "b"}) {
		t.Fatalf("Categorical = %v %v %v", codes, cats, err)
	}
	if _, err := NewCategorical("c", []int32{2}, []string{"a", "b"}); !errors.Is(err, ErrIndex) {
		t.Fatalf("out of range code: %v", err)
	}
	i, _ := NewSeries("i", []int32{1})
	if _, _, err := i.Categorical(); !errors.Is(err, ErrDType) {
		t.Fatalf("int32 series: %v", err)
	}
}
