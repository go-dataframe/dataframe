package dataframe

import "errors"

// Errors returned by the package. They are wrapped with detail; test with
// errors.Is.
var (
	// ErrDType reports an operation the column's dtype does not support.
	ErrDType = errors.New("dataframe: unsupported dtype")
	// ErrColumn reports a column name that is not in the frame.
	ErrColumn = errors.New("dataframe: no such column")
	// ErrLabel reports an index label that is not in the index.
	ErrLabel = errors.New("dataframe: label not in index")
	// ErrLength reports data whose length does not match the frame or index.
	ErrLength = errors.New("dataframe: length mismatch")
	// ErrIndex reports a position out of range.
	ErrIndex = errors.New("dataframe: position out of range")
	// ErrArgument reports an invalid argument.
	ErrArgument = errors.New("dataframe: invalid argument")
	// ErrQuery reports a query expression that does not parse or evaluate.
	ErrQuery = errors.New("dataframe: bad query")
	// ErrParse reports input (CSV, dates, ...) that does not parse.
	ErrParse = errors.New("dataframe: parse error")
)

// Must returns v, and panics if err is not nil. It is meant for notebooks and
// examples, where an error should stop the cell:
//
//	df := dataframe.Must(dataframe.ReadCSVFile("data.csv"))
func Must[T any](v T, err error) T {
	if err != nil {
		panic(err)
	}
	return v
}
