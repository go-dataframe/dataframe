# dataframe

Labelled, columnar tables in pure Go (CGO=0): `Series`, `DataFrame` and
`Index`, with selection, missing data, group-by, joins, reshaping, time
series, a fast CSV reader, Arrow IPC and Parquet, notebook display and
plots. Numeric columns are [go-ndarray](https://github.com/go-ndarray/ndarray)
arrays, handed out without copying.

pandas is the behavioural reference: the operations below give pandas'
answers (values, dtypes, index, row order), checked against pandas 3.0 on
173 recorded cases; the differences that remain are deliberate and listed in
[docs/design.md](docs/design.md#deliberate-differences-from-pandas).

```go
import "github.com/go-dataframe/dataframe"

df := dataframe.Must(dataframe.ReadCSVFile("weather.csv", dataframe.CSVParseDates("date")))
fmt.Println(df.Head(5))
```

`DataFrame`, `Series` and `Index` have `Display()`, so a
[go-notebook](https://github.com/go-notebook/notebook) cell shows them as
pandas does (text, and an HTML table).

## From pandas

| pandas | dataframe |
|---|---|
| `pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})` | `dataframe.New(dataframe.Col("a", []int{1, 2}), dataframe.Col("b", []string{"x", "y"}))` |
| `pd.Series([1.5, None])` | `dataframe.NewSeries("", []any{1.5, nil})` |
| `pd.read_csv(path, usecols=..., nrows=..., parse_dates=[...])` | `dataframe.ReadCSVFile(path, dataframe.CSVUseCols(...), dataframe.CSVNRows(n), dataframe.CSVParseDates(...))` |
| `df.to_csv(path)` | `df.WriteCSVFile(path)` |
| `pd.read_parquet(p)`, `df.to_parquet(p)` | `parquetio.ReadParquetFile(p)`, `parquetio.WriteParquetFile(p, df)` |
| `pd.read_feather(p)`, `df.to_feather(p)` | `arrowio.ReadFeather(p)`, `arrowio.WriteFeather(p, df)` |
| `df.head()`, `df.tail(3)` | `df.Head(5)`, `df.Tail(3)` |
| `df.shape`, `df.columns`, `df.dtypes`, `df.index` | `df.Shape()`, `df.Columns()`, `df.DTypes()`, `df.Index()` |
| `df["a"]`, `df[["a", "b"]]` | `df.Column("a")`, `df.Select("a", "b")` |
| `df.values`, `s.values` | `df.Values()`, `s.Values()` (`*ndarray.Array`) |
| `df.iloc[0:5]`, `df.iloc[[0, 2], 1:3]` | `df.Iloc(dataframe.Slice(0, 5))`, `df.Iloc(dataframe.Pos(0, 2), dataframe.Slice(1, 3))` |
| `df.loc["a":"c", ["x"]]` | `df.Loc(dataframe.LabelRange("a", "c"), "x")` |
| `df[df.x > 0]` | `df.Filter(dataframe.Must(x.Gt(0)))` |
| `df.query("x > 0 and s in ['a', 'b']")` | `df.Query("x > 0 and s in ['a', 'b']")` |
| `df.set_index("k")`, `df.reset_index()` | `df.SetIndex("k")`, `df.ResetIndex(false)` |
| `df.sort_values(["a", "b"], ascending=[True, False])` | `df.SortValues([]string{"a", "b"}, true, false)` |
| `df.sort_index()`, `df.reindex(labels)` | `df.SortIndex(true)`, `df.Reindex(labels)` |
| `df.dropna()`, `df.fillna(0)`, `df.isna()` | `df.DropNA()`, `df.FillNA(0)`, `df.IsNA()` |
| `df.sum()`, `df.mean()`, `df.describe()` | `df.Sum()`, `df.Mean()`, `df.Describe()` |
| `df.round(2)`, `df.T`, `df.astype({"a": "float64"})` | `df.Round(2)`, `df.T()`, `df.AsType(map[string]dataframe.DType{"a": dataframe.Float64})` |
| `df.rename(columns={"a": "b"})`, `df.copy()` | `df.Rename(map[string]string{"a": "b"})`, `df.Copy()` |
| `df.a + df.b` (aligned on the index) | `a.Add(b)` |
| `s.apply(f)`, `df.apply(f, axis=1)` | `s.Map(f)`, `df.Apply("name", f)` |
| `df.groupby(["k1", "k2"]).sum()` | `g, _ := df.GroupBy("k1", "k2"); g.Sum()` |
| `df.groupby("k").agg(x=("x", "mean"))` | `g.Agg(dataframe.Agg("x", "mean"))` |
| `pd.concat([a, b])`, `pd.concat([a, b], axis=1)` | `dataframe.Concat(a, b)`, `dataframe.ConcatWith(dataframe.ConcatOptions{Axis: 1}, a, b)` |
| `pd.merge(l, r, on="k", how="left")` | `dataframe.Merge(l, r, dataframe.MergeOptions{On: []string{"k"}, How: "left"})` |
| `df.pivot(...)`, `df.pivot_table(...)`, `df.melt(...)` | `df.Pivot(...)`, `df.PivotTable(...)`, `df.Melt(...)` |
| `df.stack()`, `s.unstack()` | `df.Stack()`, `s.Unstack()` |
| `pd.to_datetime(s)`, `pd.Timestamp("2024-01-01")` | `dataframe.ToDatetime(s)`, `dataframe.Timestamp("2024-01-01")` |
| `df.resample("D").sum()` | `r, _ := df.Resample("D"); r.Sum()` |
| `df.to_period("M")`, `s.dt.year` | `df.ToPeriod("M")`, `d, _ := s.Dt(); d.Year()` |
| `s.shift(1)`, `s.diff()`, `s.cumsum()`, `s.rolling(7).mean()` | `s.Shift(1)`, `s.Diff(1)`, `s.CumSum()`, `r, _ := s.Rolling(7); r.Mean()` |
| `df.plot()`, `df.plot.bar()`, `df.hist()`, `df.plot.scatter(x, y)` | `dfplot.Line(df)`, `dfplot.Bar(df)`, `dfplot.Hist(df, 10)`, `dfplot.Scatter(df, x, y)` |

Fallible calls return `(value, error)`; `dataframe.Must` unwraps one in a
notebook cell. Every value is immutable: operations return new frames that
share the columns they did not change.

## Speed and memory

Medians of interleaved runs, 10⁷ rows, Apple M4 Max, on a shared host under
load (load average 13–18; see [docs/perf.md](docs/perf.md) for the method, a
second run under heavier load, 10⁶ rows, and Arrow/Parquet). Milliseconds:

| operation | go-dataframe | pandas 3.0 | polars |
|---|---:|---:|---:|
| `read_csv` (5 mixed columns) | 407 | 4220 (C parser) / 525 (pyarrow) | 360 |
| `read_csv`, dates parsed | 362 | 5900 | 417 |
| `sort_values` (stable, float key) | 446 | 973 | 298 |
| `merge` (inner, int key) | 161 | 234 | 55 |
| `groupby("int key").sum()` | 88 | 103 | 69 |
| `df[df.x > 0.5]` | 74 | 75 | 20 |
| `groupby("str key").sum()` | 222 | 204 | 60 |

go-dataframe is faster than pandas on all of these but the string-key
group-by, which is at parity (CSV by about 10x, sorting by 2x); polars
remains faster on most of them. Memory is at par with pandas for
numeric columns but worse for strings (a Go `[]string` costs 16 bytes a
value; pandas 3 and polars use Arrow-style string buffers): the benchmark
table takes 400 MB here, 349 MB in pandas, 269 MB in polars.

## Packages

* `dataframe` — the core: Series, DataFrame, Index, CSV, display.
* `dataframe/arrowio` — Arrow IPC (file/Feather and stream) and Arrow record conversion, over apache/arrow-go.
* `dataframe/parquetio` — Parquet, over apache/arrow-go.
* `dataframe/dfplot` — plots with [go-plot](https://github.com/go-plot/plot).

## Verification

* `testdata/gen_fixtures.py` records pandas' answers (values, dtypes,
  index, NA handling, sort stability, group-by order, merge row order,
  `repr` text) for 173 cases; `TestPandasFixtures` replays them.
* A corpus of 47 awkward CSV files (69 option sets) is compared with pandas' parse, and Arrow and
  Parquet files written by pandas/pyarrow are read back and compared.
* 100% statement coverage of the whole module under the race detector, on
  Linux, macOS and Windows; tests also run natively on amd64/arm64 and under
  qemu on riscv64, loong64, ppc64le and s390x (big-endian).

## Licence

BSD 3-Clause, see [LICENSE](LICENSE).
