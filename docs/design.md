# Design

`dataframe` is a library of labelled, columnar tables for Go notebooks and
programs: a `Series` is one typed column with an `Index` of row labels, a
`DataFrame` is named columns sharing one index. It is pure Go (CGO=0), sits
on [go-ndarray](https://github.com/go-ndarray/ndarray) for numbers, and
takes pandas as its behavioural reference: where it answers the same
question, it gives pandas' answer, and every place it does not is listed
below, on purpose.

## Columns

A column is one generic implementation, `col[T]`, over a storage element
type `T`, plus a small table of per-dtype operations (`ops[T]`: ordering,
formatting, boxing to a Go value, recognising NaN). Writing the column
machinery once and instantiating it per element type keeps the hot loops
monomorphic (a gather over `[]float64` is a gather over `[]float64`, not
over `[]any`) and keeps the code small enough to hold to 100% statement
coverage.

| dtype | storage | notes |
|---|---|---|
| float64, float32, int8…int64, uint8…uint64, bool, complex64, complex128 | a go-ndarray array | the same 13 dtypes as go-ndarray, in its order |
| string | `[]string` | |
| datetime | int64 nanoseconds since the epoch, UTC, in an ndarray | pandas' `datetime64[ns]` |
| category | int32 codes in an ndarray + the category labels | categories are sorted when inferred |
| period | int64 ordinals + a frequency | Y, Q, M, W, D, h, min, s |

**Numeric columns are go-ndarray arrays.** A column that owns its storage
allocates it *inside* an `ndarray.Array` (`ZerosOf` then `Data[T]`, which
returns the backing slice), so `Series.Values()` hands the array out without
copying, and arithmetic is done by go-ndarray (`Add`, `Sub`, `Mul`, `Div`,
`AsType`), which gives NumPy's type promotion for free. A column that is a
slice of another (`Head`, `Iloc(Slice(...))`) shares the parent's storage
and copies only if `Values()` is asked for.

**Missing values are a validity bitmap**, one bit per row, `nil` when every
value is present (the common case costs nothing). For float columns NaN also
counts as missing, as in pandas, so data that arrives with NaN behaves as
expected. Every dtype can be missing, including integers and bools.

Columns and frames are **immutable**: operations return new values and share
whatever they did not change (renaming, selecting columns, `SetIndex` copy
no data). This removes pandas' view-versus-copy questions
(`SettingWithCopyWarning`) by construction.

## Index

An `Index` is either a range (nothing stored: `start + i*step`), or one or
more label columns of any dtype (one: an ordinary or datetime index; several:
a multi-level index such as `GroupBy` over several keys produces). Lookup by
label builds a hash map once, lazily, under a `sync.Once`, so an index is
safe to share between goroutines.

Taking rows of a range index keeps it a range when the positions are an
arithmetic progression (`Head`, `Tail`, `Iloc(Slice)`), and gives an int64
label index otherwise, as pandas does.

## Algorithms

* **Hashing and keys.** Group-by, merge, alignment, `IsIn`, `Unique` and
  `ValueCounts` all reduce to one primitive, *joint factorisation*: code the
  key tuples of one or two tables into dense integers (`jointCodes`). One
  int64 key whose values span a range no wider than a few times the row
  count is coded with a lookup table instead of a hash map (the common case
  of small integer keys); otherwise each key column is factorised with a Go
  map of its element type and the per-key codes are combined, compacting
  when the combined space would overflow.
* **Group-by** codes the keys once, sorts only the *groups* (not the rows)
  to get pandas' `sort=True` order, then aggregates in one pass per column
  with per-group accumulators (sum, count, mean, var/std two-pass), or by
  picking one row per group (min, max, first, last), which keeps the dtype.
* **Sorting** is stable and LSD: one stable pass per key, least significant
  first. A pass over a float64, int64 (also datetime and period) or int32
  key is a radix sort on an order-preserving uint64 image of the key
  (11-bit digits, skipping digits all keys share); other dtypes use a merge
  sort with the dtype's comparison. Missing values go last for every key.
* **Merge** factorises both sides' keys jointly, buckets the right rows by
  code, and emits row pairs in pandas' order; then gathers every column once.
* **Gathers** (`take`) run in parallel over row chunks for large columns,
  and frames gather their columns concurrently.

## Notebook display

`DataFrame`, `Series` and `Index` have `Display() map[string][]byte`, which
go-notebook shows: `text/plain` laid out exactly as pandas prints (checked
against pandas' own `repr` on eight fixtures: column alignment, the leading
space convention, float precision with a shared number of decimals,
scientific notation when it would hide digits, truncation to 5+5 rows past
60 rows and 10+10 columns past 20, sparsified multi-level indexes, index
names on their own line) and `text/html`, a table like pandas' notebook
one. The limits are package variables (`MaxRows`, `MinRows`, `MaxColumns`).

## Query

`DataFrame.Query` evaluates a small expression language — columns, `@vars`,
literals, arithmetic, comparisons (chained), `in`/`not in`, `and`/`or`/`not`
and `&`/`|`/`~`, `isna`/`notna`/`abs` — by a recursive-descent parser that
computes as it parses, with vectorised series operations. Nothing is ever
executed as code: the grammar is the whole of what an expression can do.

## Deliberate differences from pandas

Each is exercised by the fixture tests, which allow exactly these
differences and no other.

1. **Integer and bool columns keep their dtype when values are missing.**
   pandas turns an int64 column with a missing value into float64 (NaN),
   and a bool one into object; here they stay int64/bool with a validity
   bitmap — what pandas calls its nullable `Int64`/`boolean` dtypes. This
   shows up after `merge` (unmatched rows), `reindex`, `concat` with a
   missing column, `unstack`, `shift`, and in CSV columns with empty fields.
   Missing integers print as `<NA>`.
2. **Column labels are strings.** pandas allows any hashable; `T()` and
   `Unstack` name columns by the formatted labels.
3. **Several aggregations of one column are named `column_func`** (e.g.
   `x_sum`), where pandas makes a two-level column index. A column with a
   single aggregation keeps its name.
4. **Aggregations are `numeric_only`.** `Sum`, `Mean`, `Std`, `Var`,
   `Median` of a frame or a group-by skip non-numeric columns (pandas
   concatenates strings for `sum` and raises for `mean`); `Min`, `Max`,
   `First`, `Last`, `Count`, `NUnique` take every column in group-by, and
   numeric ones in frame reductions. `Describe` skips bool columns, as
   pandas does.
5. **Sorting is always stable** (pandas' `kind="stable"`); pandas' default
   quicksort may order ties differently.
6. **No object dtype.** A transpose or melt of mixed columns gives string
   columns; `Concat` of incompatible columns (int and string) is an error
   where pandas makes an object column.
7. **Instants are UTC nanoseconds.** Time-zone offsets in parsed strings
   are applied, and the result is UTC (pandas keeps a tz-aware dtype).
   Out-of-range instants (before 1677 or after 2262) are an error, as
   pandas' `OutOfBoundsDatetime`.
8. **Date parsing is per element.** `ToDatetime` reads ISO 8601 and a list
   of common layouts element by element; pandas infers one format from the
   first element and fails on a different one. Month-first is assumed for
   `01/02/2024`, as pandas does by default.
9. **Resampling** supports fixed bins (`s`, `min`, `h`, `D` with a count,
   origin at midnight of the first day) and calendar bins `W`, `ME`, `QE`,
   `YE`, `MS`, `QS`, `YS` with a count of one. Other rules (business days,
   anchored offsets other than Sunday weeks, `closed`/`label` overrides) are
   not implemented.
10. **A boolean mask series must cover the frame's labels.** Like pandas,
    a mask with another index is aligned by label; a frame label the mask
    lacks is an error (pandas raises too).
11. **`Join` with overlapping column names needs suffixes**, as pandas.
12. **Rolling windows** are by row count only (no time-based windows) and
    are computed directly per window (O(n·w)).

## Errors

Fallible operations return an `error` wrapping one of the package's
sentinel errors (`ErrColumn`, `ErrLabel`, `ErrDType`, `ErrLength`, ...), to
test with `errors.Is`. Operations that cannot fail (`Head`, `Copy`,
`Describe`) return the value alone. `Must` turns an `(v, err)` pair into `v`
for notebook cells and examples.

## CSV reading and writing

`ReadCSV(r, opts...)`, `ReadCSVFile(path, opts...)`, `(*DataFrame).WriteCSV(w, opts...)`
and `(*DataFrame).WriteCSVFile(path, opts...)`. The options mirror the
`read_csv` / `to_csv` parameters notebooks use; pandas serves as the
reference, and its parse of a corpus of awkward files is the oracle of the
tests (`testdata/csv/gen_fixtures.py`, `csv_corpus_test.go`).

| read option | pandas parameter |
|---|---|
| `CSVSeparator(c)` | `sep` (one byte) |
| `CSVHeader(bool)` | `header=0` / `header=None` (default: infer) |
| `CSVNames(...)` | `names` |
| `CSVUseCols(...)` | `usecols` (names) |
| `CSVNRows(n)` | `nrows` |
| `CSVSkipRows(n)` | `skiprows` (a count) |
| `CSVIndexCol(name)`, `CSVIndexColumn(i)` | `index_col` (a name, a position) |
| `CSVNAValues(...)`, `CSVKeepDefaultNA(bool)` | `na_values`, `keep_default_na` |
| `CSVParseDates(...)`, `CSVDateFormat(layout)` | `parse_dates` (names), `date_format` |
| `CSVDTypes(map[string]DType)` | `dtype` (a dict) |
| `CSVQuote(c)` (0 disables) | `quotechar`, `quoting=QUOTE_NONE` |
| `CSVComment(c)` | `comment` |
| `CSVTrueValues(...)`, `CSVFalseValues(...)` | `true_values`, `false_values` |
| `CSVSkipBlankLines(bool)` | `skip_blank_lines` |

| write option | pandas parameter |
|---|---|
| `CSVWriteIndex(bool)` | `index` |
| `CSVWriteHeader(bool)` | `header` |
| `CSVWriteSeparator(c)` | `sep` |
| `CSVWriteNARep(s)` | `na_rep` |
| `CSVWriteLineTerminator(s)` | `lineterminator` |

### How the reader works

1. **Whole input in memory.** A file is read with parallel `ReadAt` calls
   straight into one buffer (which also spreads the page faults of fresh
   memory over the cores); any other reader through `io.ReadAll`. A UTF-8
   byte order mark is skipped.
2. **Header first, serially.** Skipped rows, the header and the first data
   row are tokenized one record at a time; the first data row's field count
   decides an implicit index (more fields than names), as in pandas.
3. **Speculative parallel tokenizing.** The rest is cut into about
   4×GOMAXPROCS chunks (at least 1 MiB) at line ends. Each chunk is
   tokenized on its own as if it began a record. Stitching them in order
   checks that guess: a chunk is kept only if the previous chunk's last
   record ended exactly at its start; otherwise (it began inside a quoted
   field spanning lines) it is tokenized again from where that record truly
   ends, or dropped when that record covers it. Correct on every input; as
   parallel as the input allows. The randomized test
   `TestCSVChunkedMatchesSerial` and the corpus run at chunk sizes 1–21
   bytes check it.
4. **Fields are offsets.** A record stores its start and the end of each of
   its fields as `uint32` offsets from the chunk start (a field starts one
   byte after the previous one ends; a field a short row lacks is marked).
   No field is copied.
5. **Two-pass column conversion, in parallel per column × chunk.** The
   first pass parses integers — or, once a float shows up, floats —
   straight into the column's final storage, allocated with `makeCol` so it
   is born inside an ndarray and `Values()` is zero-copy, and records which
   kinds of value the chunk holds. The kinds of all chunks decide the dtype
   from the whole column, as pandas decides it. A second pass converts only
   the chunks whose first pass did not already produce that dtype (an
   integer chunk of a float column is converted by `float64(int64)`, which
   rounds as parsing the digits would, except for `-0`, read again).
   Integers use a hand-written parser; floats a Clinger fast path (exact
   mantissa below 2^53, power of ten up to 22) and `strconv.ParseFloat`
   otherwise, so every float is correctly rounded.
6. **Strings are views.** An unquoted field, and a quoted one without
   doubled quotes, becomes a string pointing into the input buffer
   (`unsafe.String`); the buffer is never written after reading. Only fields
   with doubled quotes, or text after a closing quote, are copied.

#### Memory trade-off

Zero-copy strings make string columns cost 16 bytes per value plus nothing,
and make the reader fast, but **any string kept alive keeps the whole input
buffer alive**: a frame of one short string column read from a 500 MB file
holds 500 MB. In the benchmark (`docs/perf.md (CSV section)`) the frame read from a
497 MB file holds about 1 GB of Go heap (the input, the 16-byte string
headers of two string columns, three numeric columns), against 370 MB for
polars, which stores strings in Arrow buffers. The peak resident memory
during the read is about the same as pandas' and polars'. A caller who
keeps only a small string column for long should copy it (for example
`AsType(Category)`); an option to copy strings into a compact per-column
buffer is a possible addition.

Fields are indexed with `uint32` offsets from their chunk's start, so a
single record longer than 4 GiB is an error rather than silently wrong.

### Deliberate differences from pandas

Each of these is checked by the corpus test, which names the rule that
applied to each column (D1–D7); everything else must match pandas 3.0.6
exactly, values and dtypes, and `WriteCSV` must write byte for byte what
`to_csv` writes for every frame read identically (47 of the 69 cases).

- **D1 — integers with missing values stay int64.** pandas makes such a
  column float64 with NaN (losing exactness above 2^53); ours is Int64 with
  a validity bitmap, the equivalent of pandas' nullable `Int64`.
- **D2 — bools with missing values stay bool**, with a validity bitmap
  (pandas: object).
- **D3 — uint64 with missing values stays uint64**, with a validity bitmap
  (pandas: text, keeping the missing-value tokens as text).
- **D4 — `parse_dates` reads each value on its own.** Any ISO 8601 form
  (date, date and time, fractional seconds, `Z` or an offset, which is
  applied) and a list of common layouts are accepted per value; pandas 3
  infers one format from the first value and leaves the column as text
  when another value differs (`2020-01-01` then `2020-01-02 10:00`). A value
  that is no date leaves the column text in both. Instants are naive UTC
  nanoseconds: a `Z` or offset is converted to UTC (pandas: a tz-aware
  dtype), and the unit is always ns (pandas 3 infers one, e.g. `us`).
- **D5 — integers beyond 64 bits stay text** (String; pandas: object
  holding Python ints).
- **D6 — floats are correctly rounded**, as pandas' `float_precision=
  "round_trip"` reads them. pandas' default parser can be one ulp off, and
  reads `0.30000000000000004` as `0.3`; the corpus compares our floats with
  pandas' round-trip parse.
- **D7 — an explicit int or bool dtype accepts missing values** (validity
  bitmap); pandas raises "Integer column has NA values".

Differences outside what the corpus can show:

- Text columns are String; a column with no rows (header only, `nrows=0`)
  is String where pandas says object.
- Error messages count lines from 1: pandas' "EOF inside string starting at
  row 2" is our "EOF inside string starting at line 3". Too-many-fields
  errors give the same line number as pandas ("expected 2 fields in line 4,
  saw 3"). Errors wrap `ErrParse`, `ErrColumn`, `ErrIndex`, `ErrArgument`.
- An explicit Bool dtype also accepts `1` and `0` (pandas does for a column
  of only those). Duplicate `CSVNames` are an error, as in pandas.
- Not supported: `index_col` lists or `False`, `usecols` by position or
  callable, `skiprows` as a list or callable, `header` rows other than the
  first (use `CSVSkipRows`), per-column `na_values`, matching numeric
  missing values by value (`-999` does not match `-999.0`), `thousands`,
  `decimal`, `escapechar`, `dayfirst`, `converters`, `chunksize`,
  compression, and encodings other than UTF-8.
- Writer: the line end is `"\n"` on every platform (pandas: `os.linesep`);
  float32 columns are written with float32's shortest digits under
  Python's repr rule (`1000000.0`; pandas/NumPy write `1e+06`); complex
  numbers are written as Go formats them (`(1+2i)`, pandas `(1+2j)`); an
  int column with missing values is written as integers (pandas, holding
  floats, writes `1.0`).

## Arrow and Parquet: design

Two packages move frames in and out of Apache Arrow data:

- `arrowio` converts frames to and from Arrow record batches and tables, and
  reads and writes Arrow IPC (the file format, also called Feather v2, and
  the streaming format).
- `parquetio` reads and writes Parquet. It reuses `arrowio`'s public
  conversions (`ToRecord`, `FromTable`) and adds only the Parquet layer.

Parquet is a separate package so that a program that only needs IPC does not
link Parquet and gRPC (see "Binary size" below). This note records which Arrow
implementation both are built on, why, what that costs, and how the types and
the index map.

### Which implementation

Three candidates were examined (versions of 2026-10-10).

**`github.com/go-ruby-arrow/arrow` and `github.com/go-ruby-parquet/parquet`.**
These are reimplementations of the *Ruby surface* of the red-arrow and
red-parquet gems. Their README says so directly: "It does not reimplement the
columnar format; it re-presents arrow-go through Ruby's naming and semantics."
Reading the source confirms it. `arrow` (1,333 lines) wraps
`apache/arrow-go/v18` arrays, schemas, tables and IPC behind Ruby-shaped types.
Values go in and out boxed, one `any` per value (`NewArrayOf(dt, []any)`,
`Array.Get(i) any`, `ToSlice() []any`). `parquet` (694 lines) is a bridge
from `pqarrow` to those Ruby-shaped tables. Neither has a format core that
could be reused. Using them would mean using arrow-go through a boxing layer
built for an interpreter. At 10^6 to 10^7 rows, boxing every value costs more
than the whole conversion. Their go.mod files also show that they bring in
exactly the dependencies of arrow-go. **Not used.** The fleet's own work
points the same way: both modules build on arrow-go.

**`github.com/apache/arrow-go/v18` v18.8.0**, the Apache Arrow project's own
Go implementation. **Used.**

- *Pure Go.* The only cgo in the packages we link (`arrow`, `arrow/array`,
  `arrow/ipc`, `arrow/memory`, `parquet`, `parquet/file`, `parquet/pqarrow`,
  `parquet/compress`) is the optional C allocator, which sits behind the
  `cgo && ccalloc` build tags. `go list -deps` with `CGO_ENABLED=1` finds no
  package with cgo files in our dependency graph. The default allocator is
  the Go allocator. SIMD kernels exist for amd64 and arm64, and every other
  architecture has a pure-Go fallback.
- *Endianness.* arrow-go keeps buffers in native byte order, and its IPC
  reader swaps a file written on a machine of the other order. Its Parquet
  encoders write little-endian on every machine.
- *Compression codecs*, all pure Go:
  - Snappy and Gzip: `klauspost/compress`
  - Zstd: `klauspost/compress/zstd`
  - LZ4: `pierrec/lz4/v4` (LZ4_RAW in Parquet, LZ4 frames in IPC)
  - Brotli: `andybalholm/brotli`

  The `parquet/compress` package registers all of them, so all of them are
  linked whichever one is used.
- *Other dependencies linked:*
  - Parquet metadata: `apache/thrift`
  - IPC messages: `google/flatbuffers`
  - hashing: `zeebo/xxh3`, `cespare/xxhash`
  - also `google/uuid` and `goccy/go-json`
  - CPU feature detection: `klauspost/cpuid`
  - `golang.org/x/exp`, `golang.org/x/sync`, `golang.org/x/sys`
  - `pqarrow` imports `arrow/flight` just to (de)serialise the stored Arrow
    schema (`flight.SerializeSchema`/`DeserializeSchema`). That one import
    links **gRPC and protobuf**: 64 `grpc` packages, plus `x/net/http2`.
- *Binary size.* These are stripped `linux/amd64` binaries that write one
  frame:

  | program | size |
  |---|---|
  | the core alone | 2.0 MB |
  | arrow-go IPC only (no arrowio) | 4.7 MB |
  | arrow-go `parquet/file` only | 20.2 MB |
  | using `arrowio` (IPC writer) | 7.7 MB |
  | using `parquetio` (Parquet writer) | 32.9 MB |

  Before the split, when Parquet lived in `arrowio`, the IPC-only program
  weighed 29.9 MB. Now it links no `parquet` package and no gRPC package
  (checked with `go list -deps`). The Parquet program links 15 arrow-go
  `parquet` packages and 64 `grpc` packages.
- *Architectures.* The test binaries were checked on every target.
  - `go vet ./...` and `go test -c` of `arrowio` and `parquetio` (with
    `CGO_ENABLED=0`) pass for
    linux/{amd64, arm64, s390x, ppc64le, riscv64, loong64} and windows/amd64.
  - The full `arrowio` and `parquetio` suites were run on linux/s390x, ppc64le, riscv64 and
    amd64 under qemu-user, through Docker Desktop's binfmt handlers, and
    passes on all four. That includes the pandas fixtures, which are
    little-endian files. A probe binary confirmed that the s390x run really
    was big-endian.
  - Files written on s390x were read by pandas on the host and came out equal
    to the ones written on arm64.
  - linux/arm64 ran natively.
  - loong64 has no binfmt handler on this host (`exec format error`), so for
    loong64 the code is only built and vetted, not run.

The alternative to arrow-go was to write our own Arrow IPC and Parquet
codecs. That means flatbuffers and Thrift metadata, five codecs, and the
Parquet encodings and definition levels. It is thousands of lines that
arrow-go already maintains upstream and tests against the other Arrow
implementations. Reuse wins.

### How data moves

- **Write**: `ToRecord` shares the frame's storage. Numeric and datetime
  columns are wrapped as Arrow buffers without copying
  (`memory.NewBufferBytes` over the column's own slice). Validity bitmaps
  are re-encoded word by word. Bools are bit-packed, and strings are copied
  once into one offsets buffer and one data buffer.
- **Read**: each column is decoded straight into a fresh ndarray that the
  new series then owns. There is one copy and no boxing. Validity bitmaps are
  copied whole with `bitutil.CopyBitmap`, at any chunk offset. The strings
  of each chunk are copied into one Go string that every value then slices.
  The frame never points into Arrow buffers, so it outlives the record and
  is safe whatever allocator was used.

To make this possible without boxing, the core gained four small public
entry points (`interop.go`):

- `NewSeriesBitmap(name, data, validity []uint64)` and
  `(*Series).ValidityBitmap()` move a packed validity bitmap in and out. The
  words are LSB-first, which is the Arrow bit order read as little-endian
  words.
- `NewCategorical(name, codes, categories)` and `(*Series).Categorical()`
  move a categorical column as codes plus categories.

### Type mapping

| dtype | written as | read from |
|---|---|---|
| float64, float32 | double, float | double, float |
| int8..int64, uint8..uint64 | same width | same width |
| bool | bool | bool |
| string | utf8 (large_utf8 past 2 GiB) | utf8, large_utf8; `null` type (an all-None object column) |
| datetime | timestamp[ns, tz=UTC] (`WithTimeZone`) | timestamp[s, ms, us or ns], any zone |
| category | dictionary<int32, utf8> | dictionary<any integer, utf8 or large_utf8> |
| period | utf8, its text (`Series.Strings`) | n/a |
| complex64/128 | `ErrUnsupported` | n/a |

Anything else on read is `ErrUnsupported`: lists, structs, decimals,
dates, durations, and dictionaries of non-strings.

**Missing values** use the validity bitmaps in both directions. A float
column marks missing values in band with NaN, as pandas does.

- On write, NaN becomes null, which is also what pandas writes.
- On read, a null float becomes NaN. Arrow distinguishes a non-null NaN from
  null, but our model cannot, so both read as missing.
- A null in any other column becomes a masked value.
- In the pandas metadata, an integer or bool column with a mask is described
  with pandas' nullable dtype (`Int64`, `UInt16`, `boolean`, ...). pandas
  then reads it back as nullable instead of casting it to float.

**Datetimes** are UTC instants.

- Writing tags them "UTC" by default. pandas then reads
  `datetime64[ns, UTC]`. `WithTimeZone("")` writes naive timestamps, which
  pandas reads as `datetime64[ns]`.
- On read, the zone is ignored: Arrow stores UTC instants whatever the zone.
  A naive timestamp is taken as UTC wall time.
- Coarser units are scaled to nanoseconds. A value outside the nanosecond
  range (years 1677 to 2262) is `ErrOutOfRange`.

**Periods** are written as text. Arrow has no calendar-period type, and the
pandas period extension type is pyarrow-specific. The core has no public
period constructor yet, so this path is untested with real periods. It is
the same code path as strings.

### The index

The `"pandas"` schema metadata is followed in both directions.

**On write:**

- A range index becomes a range descriptor `{kind: range, start, stop,
  step, name}`, and no column.
- Any other index becomes one column per level, after the data columns.
  Each is named after its level, or `__index_level_k__` when the level is
  unnamed, and listed in `index_columns`.
- The column entries carry `pandas_type`/`numpy_type`, so pandas restores
  categories, zones and nullable dtypes.
- `WithoutIndex` drops the index, like pandas' `index=False`.

**On read:**

- A descriptor `start=0, step=1` gives a range index, keeping its name.
- Any other range gives an int64 labelled index with the same labels. The
  core's range index always starts at 0 with step 1.
- Index columns become the levels of a (multi-)index. Level names come from
  the metadata, where null means unnamed.
- Without metadata the frame gets a range index.
- Malformed metadata is `ErrMetadata`.

pandas 3.0.6 with pyarrow 26.0.0 reads back the files arrowio and parquetio
write with the right dtypes and index. It was checked on every format and
every kind of index: range, named, multi-level with an unnamed level, stepped
range, and datetime.

This check is manual, because it needs Python, and does not run in CI:

```
DATAFRAME_READBACK_DIR=/tmp/rb go test -run WriteReadback ./parquetio
uv run --with pandas --with numpy --with pyarrow \
  python parquetio/testdata/check_readback.py /tmp/rb
```

The test is skipped unless the variable is set. The script exits non-zero
on any dtype, index or cross-format difference.

### Not done

- **Dropping gRPC from Parquet programs.** That needs an upstream change
  that stops `pqarrow` importing `arrow/flight`.
- **Reading row groups or columns selectively**, and reading in batches.
  `ReadParquet` reads the whole file.
- **Read types beyond the mapping**: date32/64, decimals, durations, nested
  types, string_view, and non-string dictionaries.
- **Zero-copy read.** The core cannot adopt foreign memory. Its arrays are
  go-ndarray arrays, and go-ndarray has no exported wrap-without-copy
  constructor. One copy per column remains.
- **loong64** was cross-built and vetted only, not run.
