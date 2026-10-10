package dataframe

import (
	"fmt"
	"strings"

	"github.com/go-ndarray/ndarray"
)

// DType is the element type of a column. The first thirteen values are the
// go-ndarray dtypes, in go-ndarray's order, so a numeric column converts to
// and from an ndarray dtype by a plain conversion. The others are the column
// kinds an array cannot hold: text, instants, categories and periods.
type DType uint8

// The column dtypes.
const (
	Float64 DType = DType(ndarray.Float64)
	Bool    DType = DType(ndarray.Bool)
	Int8    DType = DType(ndarray.Int8)
	Int16   DType = DType(ndarray.Int16)
	Int32   DType = DType(ndarray.Int32)
	Int64   DType = DType(ndarray.Int64)
	Uint8   DType = DType(ndarray.Uint8)
	Uint16  DType = DType(ndarray.Uint16)
	Uint32  DType = DType(ndarray.Uint32)
	Uint64  DType = DType(ndarray.Uint64)
	Float32 DType = DType(ndarray.Float32)
	// Complex64 and Complex128 columns hold and show complex numbers; they
	// take part in arithmetic and sums but have no order.
	Complex64  DType = DType(ndarray.Complex64)
	Complex128 DType = DType(ndarray.Complex128)
	// String is a column of Go strings.
	String DType = 13
	// Datetime is a column of instants, stored as int64 nanoseconds since the
	// Unix epoch, UTC.
	Datetime DType = 14
	// Category is a column of int32 codes into a list of string categories.
	Category DType = 15
	// Period is a column of calendar periods (a month, a quarter, ...), stored
	// as int64 ordinals counted from the period holding the Unix epoch.
	Period DType = 16

	numDTypes = 17
)

var dtypeNames = [numDTypes]string{
	"float64", "bool", "int8", "int16", "int32", "int64",
	"uint8", "uint16", "uint32", "uint64", "float32", "complex64", "complex128",
	"string", "datetime", "category", "period",
}

// String returns the dtype's name: "float64", "int64", "string", "datetime",
// "category", ...
func (d DType) String() string {
	if d >= numDTypes {
		return fmt.Sprintf("DType(%d)", uint8(d))
	}
	return dtypeNames[d]
}

// ParseDType returns the dtype with the given name. Besides the names String
// returns it accepts "int" (int64), "float" (float64), "str" and "object"
// (string), and "datetime64[ns]" (datetime).
func ParseDType(name string) (DType, error) {
	n := strings.ToLower(strings.TrimSpace(name))
	switch n {
	case "int":
		return Int64, nil
	case "float":
		return Float64, nil
	case "str", "object":
		return String, nil
	case "datetime64[ns]", "datetime64":
		return Datetime, nil
	}
	for i, s := range dtypeNames {
		if s == n {
			return DType(i), nil
		}
	}
	return 0, fmt.Errorf("%w: unknown dtype %q", ErrDType, name)
}

// storage returns the ndarray dtype that holds the column's data, and false
// for a column kept in a Go slice (String).
func (d DType) storage() (ndarray.DType, bool) {
	switch {
	case d < String:
		return ndarray.DType(d), true
	case d == Datetime || d == Period:
		return ndarray.Int64, true
	case d == Category:
		return ndarray.Int32, true
	}
	return 0, false
}

// IsNumeric reports whether the dtype is a number type (integer, float or
// complex) or bool, i.e. a column that converts to an ndarray as is.
func (d DType) IsNumeric() bool { return d < String }

// isReal reports a real number type: integer or float, not bool or complex.
func (d DType) isReal() bool {
	return d < String && d != Bool && d != Complex64 && d != Complex128
}

// isFloat reports float32 or float64.
func (d DType) isFloat() bool { return d == Float64 || d == Float32 }

// isInt reports a signed or unsigned integer dtype.
func (d DType) isInt() bool { return d.isReal() && !d.isFloat() }
