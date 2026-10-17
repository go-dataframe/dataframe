# Performance

All numbers are wall-clock medians of interleaved runs (Go, then pandas and
polars, repeated), on an Apple M4 Max (16 cores, 128 GB), macOS, Go 1.27.2,
pandas 3.0.6, polars (latest from PyPI via uv), numpy 2.5. **The host is
shared and was loaded by other work throughout** (`uptime` load averages are
given for each run); treat differences under ~1.5x as noise, and the polars
column especially, since polars is multi-threaded too and suffers most from
a busy machine.

## Against pandas and pyarrow, before and after (2026-10-17)

Every operation with a pandas counterpart, measured on main (ed92ca8) and
on this branch, interleaved with pandas in the same rounds: main's Go,
this branch's Go, pandas, seven rounds; each Go figure is the mean of 3
runs (`-test.benchtime 3x`), each pandas figure the median (operations)
or mean (I/O) of 3. The tables give the median over the rounds, in
milliseconds, and the ratio to pandas: **below 1.00 is faster than
pandas**. Same machine as below (Apple M4 Max, 16 cores), Go 1.27.2,
arrow-go v18.8.0, pandas 3.0.6, pyarrow 26.0.0, numpy 2.5.4, Python
3.12.12. The host was shared and loaded: load average 13–19 for the
one-thread run, 16–22 for the default one, and the spread between rounds
was wide; treat differences under ~10% as noise.

I/O is on the 10^6-row mixed frame of the Arrow and Parquet section
below; the operations on the 10^6-row table of the operations section.
to_datetime parses 10^6 "YYYY-MM-DD HH:MM:SS" strings, inferred and with
`%Y-%m-%d %H:%M:%S`; write_csv writes the table with its index, as
`df.to_csv()` does; string_eq is `cat == "c7"`, string_isin
`cat.isin(["c1", "c22", "c57"])`. go-dataframe has no `.str` accessor,
so there is no other string operation to compare.

### One thread each

GOMAXPROCS=1 for Go; pyarrow's CPU and I/O pools sized 1, and
`use_threads=False` on the reads (`THREADS=1`). pandas' own operations
are single-threaded anyway.

| operation | main | this PR | pandas | main/pandas | PR/pandas |
|---|---:|---:|---:|---:|---:|
| write Parquet | 160.0 | 107.3 | 90.6 | 1.77 | 1.18 |
| read Parquet | 45.5 | 28.9 | 33.9 | 1.34 | 0.85 |
| write IPC file | 13.7 | 9.7 | 20.6 | 0.67 | 0.47 |
| read IPC file | 6.0 | 3.5 | 5.1 | 1.18 | 0.68 |
| groupby_sum | 5.9 | 4.8 | 5.3 | 1.11 | 0.91 |
| groupby_sum_string | 12.6 | 7.6 | 11.4 | 1.11 | 0.67 |
| sort_values | 34.6 | 34.5 | 35.9 | 0.96 | 0.96 |
| filter | 7.3 | 6.9 | 4.0 | 1.82 | 1.73 |
| merge | 12.5 | 11.1 | 12.2 | 1.03 | 0.91 |
| to_datetime | 18.5 | 18.3 | 104.5 | 0.18 | 0.18 |
| to_datetime_format | 1518.4 | 24.8 | 103.4 | 14.68 | 0.24 |
| write_csv | 181.1 | 177.4 | 1175.8 | 0.15 | 0.15 |
| string_eq | 13.0 | 0.9 | 1.2 | 10.85 | 0.77 |
| string_isin | 10.3 | 3.6 | 4.1 | 2.52 | 0.87 |

### Default threading

Go uses every core where it parallelises (column decoding of Parquet
reads, CSV writing, filters, factorisation, sorting); pyarrow its thread
pools (Parquet and Feather reads, the conversion from pandas before a
Parquet write); pandas' operations are single-threaded. For the
operations this compares a multi-threaded Go with a single-threaded
pandas, so the table above is the fair one; for the I/O both sides are
multi-threaded.

| operation | main | this PR | pandas | main/pandas | PR/pandas |
|---|---:|---:|---:|---:|---:|
| write Parquet | 139.3 | 101.1 | 90.2 | 1.54 | 1.12 |
| read Parquet | 24.8 | 11.8 | 15.9 | 1.56 | 0.74 |
| write IPC file | 11.4 | 8.0 | 17.7 | 0.64 | 0.45 |
| read IPC file | 6.9 | 3.2 | 3.4 | 2.03 | 0.95 |
| groupby_sum | 4.1 | 3.2 | 5.8 | 0.70 | 0.55 |
| groupby_sum_string | 4.3 | 3.1 | 11.7 | 0.36 | 0.26 |
| sort_values | 22.0 | 19.6 | 37.9 | 0.58 | 0.52 |
| filter | 2.2 | 2.0 | 4.2 | 0.53 | 0.48 |
| merge | 8.6 | 6.9 | 12.8 | 0.67 | 0.54 |
| to_datetime | 16.8 | 17.8 | 108.3 | 0.16 | 0.16 |
| to_datetime_format | 1183.4 | 23.3 | 106.7 | 11.09 | 0.22 |
| write_csv | 37.6 | 36.2 | 1174.9 | 0.03 | 0.03 |
| string_eq | 11.3 | 0.3 | 1.2 | 9.40 | 0.25 |
| string_isin | 3.2 | 0.6 | 4.1 | 0.78 | 0.14 |

### What changed

- **Arrow IPC and Parquet reads hand the buffers over.** A read with the
  Go allocator owns its records, whose buffers are ordinary Go memory that
  `Release` never frees or reuses: a column in one chunk keeps the chunk's
  values buffer, strings slice the chunk's bytes, category codes are the
  dictionary indices when the dictionary is already the categories. Any
  other allocator, and `FromRecord`, `FromRecords` and `FromTable` (whose
  caller keeps the record), still copy.
- **Parquet string columns stored dictionary-encoded throughout** (per the
  page encoding statistics) are read as dictionaries and laid out as
  strings sharing each entry's bytes.
- **Parquet writes spare arrow-go two kinds of per-value hashing** that it
  would undo anyway: numeric chunks with more distinct values than the
  dictionary page holds are written plain from the start, and string
  chunks whose distinct values fit it are handed over as dictionary
  arrays. See parquetio/plan.go.
- **Datetime layouts are planned once** (the strftime translation was
  rebuilt for every value): a fixed-width layout is read field by field,
  anything else still goes to `time.Parse`.
- **String comparison and isin** have direct loops; coding strings
  (group-by keys, isin) caches by a string's address and length, so values
  that share their bytes are not hashed again; one int64 group key of
  narrow range is coded in key order from a presence table; inner and left
  merges skip the left-row buckets they never read; a filtered range index
  keeps the positions as its labels; a large read converts its columns
  concurrently.

### Results unchanged

- Every Arrow IPC file and stream written is byte-identical to main's
  (SHA-256 over 20 files: 5 frames, file and stream, LZ4 and Zstd).
- Parquet files whose chunks are small (row groups under 131,072 rows) or
  uncompressed are byte-identical too. The others differ in encodings and
  page layout only: a column with too many distinct values for a
  dictionary has no dictionary page (and the file is smaller: the
  200,000-row bench frame 3.73 → 2.87 MB, a 300,000-row mixed one 8.53 →
  6.73 MB), and a string column handed over as a dictionary has one data
  page per row group. Statistics (null count, number of values, min, max)
  are identical in every column chunk, and pyarrow
  (`pq.read_table(...).equals`, metadata included) and pandas
  (`read_parquet(...).equals`, dtypes and index) read identical tables
  from main's files and from this branch's, for all 20 files compared.
- Reads through the zero-copy path give frames identical to the copying
  path, the bytes of every numeric slot included (`TestAdoptMatchesCopy`
  in arrowio and parquetio, over every fixture and round trip).
- Each new fast path is tested against the general path it short-cuts.

### Still slower than pandas or pyarrow

- **write Parquet: ~1.2x pyarrow on one thread** (~1.1x with default
  threading). A third of the remaining time is arrow-go's Snappy encoder
  (klauspost/compress `s2.EncodeSnappyBetter`, ~33 ms of ~110), and ~12 ms
  the `UniqueArray` arrow-go runs over each dictionary column's indices
  for its statistics. s2's standard Snappy mode is 2.7x faster on these
  pages, for 12% larger output, but can only be swapped in through
  arrow-go's process-wide codec registry, which would change every arrow-go
  writer in a program importing parquetio; that is not done here.
- **filter: ~1.7x pandas on one thread** (0.5x with default threading).
  The work itself is faster than pandas' (3.1 ms with the collector off,
  against 4.0 ms); the rest is Go's garbage collector, which on one core
  spends ~45% of the time marking the table's string column, 10^6 string
  headers (pandas 3 stores strings Arrow-style, without pointers). An
  Arrow-style string column is the fix, and a larger change than this one.

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

The runs below predate the changes of 2026-10-17 (see the section above).

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
`arrowio` (IPC) and `parquetio` (Parquet) with pandas (pyarrow engine). They were taken on 2026-10-10,
before the changes of 2026-10-17: the section at the top of this page has
the current figures, before and after; "Where the time goes" below is
updated.

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

(2026-10-17, after the changes; single-threaded profiles.)

**Parquet write.** About a third of the time is the Snappy encoder
(`s2.EncodeSnappyBetter`), an eighth the `UniqueArray` arrow-go runs over
the indices of each dictionary column for its statistics, and most of the
rest arrow-go's page building and level and RLE encoding. Our conversion
and planning (`ToRecord`, the distinct-value probe, the string
dictionaries) is under a tenth.

**Parquet read.** Snappy decoding, arrow-go's record readers (whose output
buffers the frame now keeps), and the Go runtime faulting in fresh memory:
arrow-go allocates about 160 MB to read this frame.

**IPC read.** The copy of the file's bytes into arrow-go's buffers (a
`bytes.Reader` gives no zero-copy access), then 10^6 string headers to
build; the frame keeps the numeric, timestamp and category buffers and the
string bytes as read.

**IPC write.** Zero-copy for numeric, datetime and category columns; the
string column is laid out once.

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

For one thread each, run it with `GOMAXPROCS=1 THREADS=1` in the
environment (`bench_pandas.py` reads `THREADS`). The operations:
`[THREADS=1] sh bench/run.sh 1000000 7 > out.txt && python3 bench/medians.py out.txt`.
