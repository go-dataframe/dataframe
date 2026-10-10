# Performance

All numbers are wall-clock medians of interleaved runs (Go, then pandas and
polars, repeated), on an Apple M4 Max (16 cores, 128 GB), macOS, Go 1.27.2,
pandas 3.0.6, polars (latest from PyPI via uv), numpy 2.5. **The host is
shared and was loaded by other work throughout** (`uptime` load averages are
given for each run); treat differences under ~1.5x as noise, and the polars
column especially, since polars is multi-threaded too and suffers most from
a busy machine.

## Operations on a table

The table is built identically, value for value, by `bench_test.go` and
`bench/bench_pandas.py`: `key` int64 in [0, 1000), `x` and `y` float64 in
[0, 1), `cat` a string among 100 values. Operations:

* `groupby_sum`: `df.groupby("key").sum()` — `GroupBy("key").Sum()`
* `groupby_sum_string`: the same, keyed by the string column
* `sort_values`: stable sort by `x` (pandas `kind="stable"`, polars `maintain_order=True`)
* `filter`: `df[df.x > 0.5]`
* `merge`: inner join with a 1000-row table on `key` (unique on the right)
* memory: bytes the table holds (Go: heap growth when building it;
  pandas: `memory_usage(deep=True)`; polars: `estimated_size()`)

Reproduce: `sh bench/run.sh 1000000 5 > out.txt && python3 bench/medians.py out.txt`.

### Run A — load average 13–18 (10^6: 5 rounds, 10^7: 3 rounds)

10^6 rows, milliseconds:

| operation | go-dataframe | pandas | polars |
|---|---:|---:|---:|
| groupby_sum | 8.7 | 9.3 | 6.5 |
| groupby_sum_string | 19.9 | 23.7 | 6.0 |
| sort_values | 34.5 | 63.3 | 22.0 |
| filter | 6.6 | 7.1 | 2.3 |
| merge | 15.9 | 20.0 | 4.6 |
| memory (MB) | 40.0 | 34.9 | 26.9 |

10^7 rows, milliseconds (this run predates the parallel factorisation and
parallel filter, which help the string group-by and the filter):

| operation | go-dataframe | pandas | polars |
|---|---:|---:|---:|
| groupby_sum | 88.3 | 103.4 | 69.3 |
| groupby_sum_string | 221.7 | 203.7 | 59.7 |
| sort_values | 445.6 | 972.8 | 298.4 |
| filter | 74.2 | 74.6 | 19.8 |
| merge | 161.1 | 233.8 | 54.5 |
| memory (MB) | 400.0 | 349.0 | 269.0 |

### Run B — final code, load average 35–60

10^6 rows:

| operation | go-dataframe | pandas | polars |
|---|---:|---:|---:|
| groupby_sum | 11.4 | 11.1 | 35.5 |
| groupby_sum_string | 16.6 | 25.3 | 19.5 |
| sort_values | 51.6 | 81.5 | 63.0 |
| filter | 8.9 | 9.1 | 4.5 |
| merge | 17.5 | 29.3 | 17.5 |
| memory (MB) | 40.0 | 34.9 | 26.9 |

10^7 rows:

| operation | go-dataframe | pandas | polars |
|---|---:|---:|---:|
| groupby_sum | 99.2 | 243.6 | 140.7 |
| groupby_sum_string | 120.3 | 384.8 | 89.1 |
| sort_values | 559.8 | 1350.7 | 571.3 |
| filter | 49.2 | 85.5 | 39.7 |
| merge | 181.9 | 302.6 | 115.1 |
| memory (MB) | 400.0 | 349.0 | 269.0 |

### Reading these honestly

* Against pandas: sorting is about 2x faster (stable LSD radix sort), merge
  1.3–1.7x, group-by on an int key at parity to 2x, group-by on a string key
  and the filter at parity or better. Nothing measured is slower than pandas
  by more than noise, except the string group-by in run A at 10^7, before
  factorisation went parallel.
* Against polars: polars is faster on most operations on a quiet machine
  (2–4x on filter, merge and string group-by); sorting is close.
* **Memory is the weak point.** Numeric columns cost what pandas' cost (8
  bytes a value). A string column is a Go `[]string`: 16 bytes of header per
  value, plus the bytes (shared here, since there are only 100 distinct
  values). pandas 3 and polars store strings Arrow-style (offsets + one
  byte buffer), about 11 bytes a value here — so this table is 40 MB here
  against 35 MB in pandas and 27 MB in polars. Converting such a column to
  `Category` (4 bytes a value) makes it the smallest of the three. An
  Arrow-style string column is the obvious next step.
* The merge benchmark joins on keys unique on the right with every left row
  matched, so the left columns are shared rather than gathered (columns are
  immutable); a join that duplicates or drops left rows gathers them.
* Go's garbage collector scans `[]string` columns; with large string columns
  a share of the time in allocation-heavy operations (filter, merge) is GC.


## CSV reader performance

Measured 2026-10-10 with `bench/csv/bench.py` (reproduce with the command in
its docstring; the data files go to a temporary directory outside the
repository).

### Conditions

- Apple M4 Max, 16 cores, 128 GB, macOS 26.6.2; Go 1.27.2; pandas 3.0.6,
  polars 2.0.0, pyarrow (pandas' `engine="pyarrow"`), Python 3.12.12.
- **The machine was shared and loaded**: load average 14–16 for most runs,
  rising to 45 during the last (10^7 rows, dates) series. Every reader ran
  under the same load, but absolute times are pessimistic and the spread is
  wide; the ratios are the meaningful part.
- Table: `id` (int64), `x` (standard normal, full 17-digit precision),
  `y` (two decimals), `cat` (`cat0`..`cat49`), `date` (`YYYY-MM-DD`), as
  written by pandas' `to_csv`. 10^6 rows = 49 MB, 10^7 rows = 497 MB.
- Method: each read in its own process under `/usr/bin/time -l`; each
  process times only the read call (warm page cache, no process start-up).
  The readers run interleaved (ours, pandas C, pandas pyarrow, polars,
  ours, ...), 5 rounds; the table gives the median (min–max).
- "Data MB": Go heap live after the read and a GC (`runtime.MemStats.
  HeapAlloc`), pandas `memory_usage(deep=True).sum()`, polars
  `estimated_size()`. These are not quite the same measure: ours includes
  the input buffer the string columns point into (see the memory
  trade-off in docs/design.md (CSV section)).
- "Peak RSS": maximum resident set size of the whole process. The Python
  processes include ~118 MB for the interpreter and the imports alone.

### Default read (date column left as text)

| rows | reader | median s (min–max) | MB/s | data MB | peak RSS MB |
|---|---|---:|---:|---:|---:|
| 10^6 | **go ReadCSV** | **0.039** (0.038–0.047) | 1247 | 103 | 136 |
| 10^6 | pandas, C engine | 0.385 (0.383–0.392) | 127 | 52 | 236 |
| 10^6 | pandas, pyarrow engine | 0.052 (0.050–0.057) | 936 | 73 | 262 |
| 10^6 | polars | 0.037 (0.033–0.044) | 1310 | 37 | 226 |
| 10^7 | **go ReadCSV** | **0.407** (0.339–0.546) | 1222 | 1031 | 1273 |
| 10^7 | pandas, C engine | 4.223 (3.849–4.434) | 118 | 523 | 1349 |
| 10^7 | pandas, pyarrow engine | 0.525 (0.468–0.790) | 946 | 732 | 1553 |
| 10^7 | polars | 0.360 (0.355–0.425) | 1379 | 370 | 1314 |

### With the date column parsed (`CSVParseDates("date")`, `parse_dates=["date"]`, `try_parse_dates=True`)

| rows | reader | median s (min–max) | MB/s | data MB | peak RSS MB |
|---|---|---:|---:|---:|---:|
| 10^6 | **go ReadCSV** | **0.054** (0.043–0.069) | 899 | 95 | 128 |
| 10^6 | pandas, C engine | 0.532 (0.487–0.598) | 92 | 43 | 221 |
| 10^6 | pandas, pyarrow engine | 0.699 (0.608–0.838) | 70 | 43 | 336 |
| 10^6 | polars | 0.048 (0.036–0.132) | 1009 | 31 | 217 |
| 10^7 | **go ReadCSV** | **0.362** (0.332–0.452) | 1372 | 955 | 1197 |
| 10^7 | pandas, C engine | 5.896 (4.963–13.218) | 84 | 427 | 1257 |
| 10^7 | pandas, pyarrow engine | 7.729 (5.647–13.264) | 64 | 427 | 2300 |
| 10^7 | polars | 0.417 (0.344–0.871) | 1190 | 313 | 1223 |

The 10^7-row date series ran while the load average climbed from 15 to 45;
its pandas maxima (13 s) and polars spread show it.

### Reading

- Against pandas' default C parser: about **10× faster** (9.9× at 10^6,
  10.4× at 10^7), and 10–16× with dates parsed.
- Against polars: within 5% at 10^6 and about 13% slower at 10^7 on the
  default read; level (within noise) with dates. Faster than pandas'
  pyarrow engine on every run but one.
- Memory: the peak resident size of a read is in line with pandas and
  polars (lower once their interpreter baseline is counted), but the frame
  read holds about 2× pandas' and 2.8× polars' bytes, because the strings
  share the input buffer and Go strings carry 16-byte headers. That is the
  price of zero-copy strings; see docs/design.md (CSV section).

Go benchmarks (`go test -bench 'ReadCSV|WriteCSV'`, in-memory, same table
shape, load ~35): ReadCSV 10^6 rows ≈ 40–48 ms (≈1.1–1.3 GB/s); WriteCSV
10^5 rows ≈ 14 ms (≈400 MB/s).

### Where the time goes (profile, 10^7 rows)

About half the CPU is the first conversion pass (integer and float parsing;
`strconv.ParseFloat` for 17-digit values), a fifth the tokenizer, an eighth
building the string columns, and a tenth the garbage collector marking the
string columns. The remaining gap to polars is mostly float parsing (an
Eisel–Lemire parser of our own would skip `ParseFloat`'s re-scan) and GC
work on string headers.

## Arrow and Parquet: performance

All measurements write and read a 10^6-row frame in memory, comparing
`arrowio` (IPC) and `parquetio` (Parquet) with pandas (pyarrow engine). They were taken on 2026-10-10.

### Setup

**Frame.** Six columns, built identically on both sides
(`benchFrame` in `arrowio/bench_test.go`, for IPC, and in
`parquetio/bench_test.go`, for Parquet; `arrowio/testdata/bench_pandas.py`
on the pandas side):

| column | type | missing |
|---|---|---|
| float64 | float64 | 1% NaN |
| int64 | int64 | none |
| string | 1,000 distinct values | 1% |
| category | 10 levels | none |
| bool | bool | none |
| datetime | datetime | none |

**Operations.**

- Parquet with Snappy compression, the default on both sides. parquetio
  produces a 15.2 MB file.
- The Arrow IPC file format without compression. For pandas this is
  `to_feather(compression="uncompressed")`.

**Method.**

- 7 runs, with Go and pandas interleaved: a Go run, then a pandas run, and
  so on.
- Each run reports the mean of 3 repetitions of each operation, in
  milliseconds.
- The table gives the median of the 7 runs.

**Machine.** Apple M4 Max, 16 cores. The host is shared and was loaded
throughout: `uptime` load averages were 14.97 / 18.50 / 18.88 before the
runs and 14.15 / 17.96 / 18.67 after.

**Versions.** Go 1.27.2, apache/arrow-go v18.8.0. pandas 3.0.6, pyarrow
26.0.0, Python 3.12.12.

### Results

Median of 7 runs, in milliseconds:

| operation | go-dataframe | pandas | ratio |
|---|---|---|---|
| write Parquet | 248.7 | 170.7 | 1.46x slower |
| read Parquet | 40.1 | 29.3 | 1.37x slower |
| write IPC file | 21.0 | 33.5 | 0.63x (faster) |
| read IPC file | 14.7 | 5.5 | 2.7x slower |

<details>
<summary>Raw runs (milliseconds)</summary>

Each line: write Parquet, read Parquet, write IPC, read IPC.

```
go 1 248.7 36.8 19.5 11.4
py 1 214.3 56.8 34.1 4.8
go 2 240.0 36.8 17.9 12.9
py 2 151.9 26.7 29.6 4.7
go 3 233.2 36.1 21.0 15.6
py 3 168.8 26.0 32.4 5.3
go 4 247.0 40.1 23.4 14.7
py 4 181.2 30.7 34.4 6.2
go 5 291.0 48.2 26.5 16.1
py 5 176.7 30.5 37.1 9.0
go 6 307.6 40.1 20.3 10.7
py 6 170.7 29.3 33.5 5.5
go 7 278.4 46.6 24.6 15.1
py 7 161.3 27.1 33.2 5.8
```

</details>

### Where the time goes

**Parquet write.** About 70% of the allocations happen in arrow-go's
dictionary encoder for byte arrays: `hashing.(*BinaryMemoTable).InsertOrGet`
makes about one allocation per string value (1.0 M allocations per write of the 10^6-row frame). pyarrow uses the same
encoding, with dictionaries on by default, but in C++. Our conversion
itself shares the numeric buffers and adds one pass for strings.

**Parquet read.** The time is dominated by arrow-go's column decoding, which
runs in parallel across columns. Our conversion is a minority of the
profile:

- copying the validity bitmaps (with `bitutil.CopyBitmap`)
- one copy of each numeric buffer into the frame's own array
- one string allocation per chunk for the text

**IPC read.** pyarrow's `to_pandas` is mostly zero-copy for numeric
columns, so pandas pays almost nothing to read them. We copy every column
once into storage the frame owns, because go-ndarray cannot adopt foreign
memory. The string column also gets its `[]string` headers built, and then
copied once more by the core's constructor.

**IPC write.** This is zero-copy for numeric and datetime columns, which is
why it beats pandas.

### How to rerun

```
go test -c -o /tmp/arrowio.test ./arrowio
go test -c -o /tmp/parquetio.test ./parquetio
for i in 1 2 3 4 5 6 7; do
  /tmp/parquetio.test -test.run '^$' -test.bench . -test.benchtime 3x
  /tmp/arrowio.test -test.run '^$' -test.bench . -test.benchtime 3x
  (cd arrowio/testdata &&
    uv run --with pandas --with numpy --with pyarrow python bench_pandas.py)
done
```

The figures above were taken before Parquet moved to `parquetio`. The
Parquet code itself did not change in the split, only its package.
