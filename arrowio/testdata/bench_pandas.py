"""The pandas side of docs/perf.md (Arrow and Parquet section): the frame of benchFrame in
bench_test.go, written and read as Parquet (pyarrow engine, snappy) and
Feather (uncompressed) in memory. Prints one line of milliseconds per
operation, each the mean of REPS repetitions:

    write_parquet read_parquet write_ipc read_ipc

    uv run --with pandas --with numpy --with pyarrow python bench_pandas.py
"""

import io
import time

import numpy as np
import pandas as pd

N = 1_000_000
REPS = 3

i = np.arange(N)
f = i * 0.5
f[i % 100 == 0] = np.nan
s = pd.Series([f"s{v}" for v in i % 1000], dtype=object)
s[i % 100 == 1] = None
df = pd.DataFrame(
    {
        "f": f,
        "k": (i * 7) % 1000003,
        "s": s,
        "c": pd.Categorical.from_codes(i % 10, [f"c{v}" for v in range(10)]),
        "b": i % 3 == 0,
        "t": pd.to_datetime(1_600_000_000 + i, unit="s").astype("datetime64[ns]"),
    }
)


def timed(fn):
    t0 = time.perf_counter()
    for _ in range(REPS):
        fn()
    return (time.perf_counter() - t0) / REPS * 1e3


pq = io.BytesIO()
df.to_parquet(pq, engine="pyarrow")
fe = io.BytesIO()
df.reset_index(drop=True).to_feather(fe, compression="uncompressed")

w_pq = timed(lambda: df.to_parquet(io.BytesIO(), engine="pyarrow"))
r_pq = timed(lambda: pd.read_parquet(io.BytesIO(pq.getvalue()), engine="pyarrow"))
w_fe = timed(lambda: df.to_feather(io.BytesIO(), compression="uncompressed"))
r_fe = timed(lambda: pd.read_feather(io.BytesIO(fe.getvalue())))
print(f"{w_pq:.1f} {r_pq:.1f} {w_fe:.1f} {r_fe:.1f}")
