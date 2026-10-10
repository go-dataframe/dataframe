"""Manual check, not run in CI: pandas reads the files go-dataframe writes
with the expected dtypes and index.

    DATAFRAME_READBACK_DIR=/tmp/rb go test -run WriteReadback ./parquetio
    uv run --with pandas --with numpy --with pyarrow python parquetio/testdata/check_readback.py /tmp/rb

Each frame is written as Parquet, Feather (IPC file) and an IPC stream by
TestWriteReadback; the three must read as the same pandas frame, with the
dtypes and index listed in EXPECTED. Exits non-zero on any difference.
"""

import sys

import pandas as pd
import pyarrow.ipc as ipc

out = sys.argv[1]

COLUMNS = {
    "f64": "float64", "f32": "float32", "i8": "int8", "i64m": "Int64",
    "u16m": "UInt16", "u64": "uint64", "b": "bool", "bm": "boolean",
    "s": ("str", "object"), "cat": "category", "ts": "datetime64[ns, UTC]",
    "key": ("str", "object"), "lvl": "int64",
}
# frame: (index class, index names)
EXPECTED = {
    "range": ("RangeIndex", [None]),
    "named": ("Index", ["key"]),
    "multi": ("MultiIndex", ["key", None]),
    "stepped": ("RangeIndex", [None]),
    "dtindex": ("DatetimeIndex", ["ts"]),
}


def read(name, ext):
    p = f"{out}/{name}.{ext}"
    if ext == "parquet":
        return pd.read_parquet(p)
    if ext == "feather":
        return pd.read_feather(p)
    with open(p, "rb") as f:
        return ipc.open_stream(f).read_all().to_pandas()


bad = []
for name, (kind, names) in EXPECTED.items():
    frames = [read(name, e) for e in ("parquet", "feather", "arrows")]
    for ext, d in zip(("feather", "arrows"), frames[1:]):
        try:
            pd.testing.assert_frame_equal(frames[0], d)
        except AssertionError as e:
            bad.append(f"{name}: {ext} differs from parquet: {e}")
    d = frames[0]
    if type(d.index).__name__ != kind or list(d.index.names) != names:
        bad.append(f"{name}: index {type(d.index).__name__} {list(d.index.names)}, want {kind} {names}")
    for col, dt in d.dtypes.astype(str).items():
        want = COLUMNS[col]
        # A column with no missing value left after a row selection keeps
        # its non-nullable dtype.
        if name == "stepped" and col in ("i64m", "u16m", "bm"):
            want = {"i64m": "int64", "u16m": "uint16", "bm": "bool"}[col]
        if dt not in (want if isinstance(want, tuple) else (want,)):
            bad.append(f"{name}.{col}: dtype {dt}, want {want}")
    print(f"{name}: {kind} {names} ok" if not any(b.startswith(name) for b in bad) else f"{name}: FAILED")

naive = pd.read_parquet(f"{out}/naive.parquet")["ts"].dtype
if str(naive) != "datetime64[ns]":
    bad.append(f"naive: {naive}")
print(f"pandas {pd.__version__}")
if bad:
    print("\n".join(bad))
    sys.exit(1)
print("readback OK")
