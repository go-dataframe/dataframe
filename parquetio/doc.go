// Package parquetio reads and writes frames as Parquet files, through
// github.com/apache/arrow-go's pure-Go Parquet implementation (no cgo).
//
// The frame is laid out as package arrowio lays out an Arrow record: the
// same type mapping, the same handling of missing values, and the same
// "pandas" metadata, so that pandas reads the files back with their index
// and dtypes, and files pandas writes read here with theirs. See the
// arrowio documentation for the mapping.
//
// Parquet lives in its own package because arrow-go's Parquet bridge links
// far more code (Thrift, every codec, and, through arrow/flight, gRPC) than
// the Arrow IPC formats: a program that only needs IPC imports arrowio
// alone.
package parquetio
