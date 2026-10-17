"""The pandas (and polars) side of the operation benchmarks.

Builds the same table as bench_test.go, value for value, and times
groupby-sum, sort_values, a filter and a merge; prints the median of
--repeat runs in milliseconds, and the frame's memory.

    uv run --with pandas --with numpy --with polars python bench/bench_pandas.py 1000000

It also times the conversion of text to datetimes (inferred, and with a
format), writing the table as CSV, and comparing the string column, which
polars is not timed on. With THREADS=1 in the environment, pyarrow (behind
pandas' str dtype) is held to one thread, to compare with the Go side
under GOMAXPROCS=1.
"""

import statistics
import sys
import time

import numpy as np
import pandas as pd
import os

if os.environ.get("THREADS") == "1":
    import pyarrow as pa

    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)

n = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
repeat = int(sys.argv[2]) if len(sys.argv) > 2 else 5
i = np.arange(n, dtype=np.uint64)
df = pd.DataFrame({
    "key": ((i * np.uint64(7919)) % np.uint64(1000)).astype(np.int64),
    "x": (i * np.uint64(2654435761) % np.uint64(1000003)).astype(np.float64) / 1000003,
    "y": (i * np.uint64(40503) % np.uint64(999983)).astype(np.float64) / 999983,
    "cat": pd.Series(np.array(["c%d" % k for k in range(100)])[(np.arange(n) * 31) % 100]).astype("str"),
})
right = pd.DataFrame({"key": np.arange(1000, dtype=np.int64), "v": np.arange(1000) / 10})


def bench(name, f):
    ts = []
    for _ in range(repeat):
        t = time.perf_counter()
        f()
        ts.append((time.perf_counter() - t) * 1000)
    print(f"pandas {name} {statistics.median(ts):.1f} ms")


bench("groupby_sum", lambda: df.groupby("key").sum(numeric_only=True))
bench("groupby_sum_string", lambda: df.groupby("cat").sum(numeric_only=True))
bench("sort_values", lambda: df.sort_values("x", kind="stable"))
bench("filter", lambda: df[df["x"] > 0.5])
bench("merge", lambda: pd.merge(df, right, on="key"))
dates = pd.Series(pd.to_datetime(1577836800 + (i * np.uint64(7919)) % np.uint64(31536000), unit="s")
                  .strftime("%Y-%m-%d %H:%M:%S")).astype("str")
bench("to_datetime", lambda: pd.to_datetime(dates))
bench("to_datetime_format", lambda: pd.to_datetime(dates, format="%Y-%m-%d %H:%M:%S"))
bench("write_csv", lambda: df.to_csv())
bench("string_eq", lambda: df["cat"] == "c7")
bench("string_isin", lambda: df["cat"].isin(["c1", "c22", "c57"]))
print(f"pandas memory {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

try:
    import polars as pl
except ImportError:
    sys.exit(0)
pdf = pl.from_pandas(df)
pright = pl.from_pandas(right)


def pbench(name, f):
    ts = []
    for _ in range(repeat):
        t = time.perf_counter()
        f()
        ts.append((time.perf_counter() - t) * 1000)
    print(f"polars {name} {statistics.median(ts):.1f} ms")


pbench("groupby_sum", lambda: pdf.group_by("key").agg(pl.col("x").sum(), pl.col("y").sum()).sort("key"))
pbench("groupby_sum_string", lambda: pdf.group_by("cat").agg(pl.col("key").sum(), pl.col("x").sum(), pl.col("y").sum()).sort("cat"))
pbench("sort_values", lambda: pdf.sort("x", maintain_order=True))
pbench("filter", lambda: pdf.filter(pl.col("x") > 0.5))
pbench("merge", lambda: pdf.join(pright, on="key", how="inner", maintain_order="left"))
print(f"polars memory {pdf.estimated_size() / 1e6:.1f} MB")
