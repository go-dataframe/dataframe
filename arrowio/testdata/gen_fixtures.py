"""Generate the Arrow IPC and Parquet fixtures the arrowio and parquetio
tests read (parquetio reads the .parquet files from here).

pandas and pyarrow are used here only as an independent oracle: they write
the files, and expected.json records, for every fixture, the dtype and the
values the Go reader must produce. Run from this directory with

    uv run --with pandas --with numpy --with pyarrow python gen_fixtures.py

The files are small and committed; CI does not run this script.
"""

import json
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
expected = {"versions": {"pandas": pd.__version__, "pyarrow": pa.__version__}, "fixtures": {}}


def go_dtype(s):
    """The go-dataframe dtype name a pandas column is expected to read as."""
    d = s.dtype
    if isinstance(d, pd.CategoricalDtype):
        return "category"
    if isinstance(d, pd.DatetimeTZDtype) or str(d).startswith("datetime64"):
        return "datetime"
    if str(d) == "boolean":
        return "bool"
    if str(d) in ("Int64",):
        return "int64"
    if d == object or str(d) in ("str", "string"):
        return "string"
    return str(d)


def values(s):
    """JSON values of a pandas column: null where missing, datetimes as int64
    nanoseconds since the epoch (UTC instants; naive wall time read as UTC)."""
    kind = go_dtype(s)
    out = []
    if kind == "datetime":
        if isinstance(s.dtype, pd.DatetimeTZDtype):
            s = s.dt.tz_convert("UTC").dt.tz_localize(None)
        s = s.astype("datetime64[ns]")
        ns = s.astype("int64")
        for v, na in zip(ns, s.isna()):
            out.append(None if na else int(v))
        return out
    for v in s.astype(object):
        if v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v)):
            out.append(None)
        elif isinstance(v, (bool, np.bool_)):
            out.append(bool(v))
        elif isinstance(v, (int, np.integer)):
            out.append(int(v))
        elif isinstance(v, (float, np.floating)):
            out.append(float(v))
        else:
            out.append(str(v))
    return out


def describe(df):
    cols = [{"name": str(c), "dtype": go_dtype(df[c]), "values": values(df[c])} for c in df.columns]
    ix = df.index
    if isinstance(ix, pd.RangeIndex) and ix.start == 0 and ix.step == 1:
        index = {"range": True, "name": ix.name or ""}
    else:
        levels = []
        for k in range(ix.nlevels):
            lv = pd.Series(ix.get_level_values(k))
            levels.append({"name": ix.names[k] or "", "dtype": go_dtype(lv), "values": values(lv)})
        index = {"range": False, "levels": levels}
    return {"columns": cols, "index": index, "rows": len(df)}


def write_all(name, df, feather_compression="lz4"):
    """Write df as Parquet (snappy, pandas' default), Feather v2 (lz4, the
    feather default) and an Arrow IPC stream, and record what to expect."""
    df.to_parquet(os.path.join(HERE, name + ".parquet"))
    df.to_feather(os.path.join(HERE, name + ".feather"), compression=feather_compression)
    table = pa.Table.from_pandas(df)
    with pa.OSFile(os.path.join(HERE, name + ".arrows"), "wb") as f:
        with pa.ipc.new_stream(f, table.schema) as w:
            w.write_table(table)
    expected["fixtures"][name] = describe(df)


def dt(xs, **kw):
    return pd.to_datetime(xs, format="ISO8601", **kw)


n = 7
types = pd.DataFrame(
    {
        "f64": [1.5, np.nan, -0.25, 1e300, 0.0, -1.0, 3.0],
        "f32": np.array([1.5, 2.25, np.nan, -4.0, 0.5, 8.0, 0.0], dtype="float32"),
        "i8": np.array([-128, 127, 0, 1, -1, 5, 6], dtype="int8"),
        "i16": np.array([-32768, 32767, 0, 1, -1, 5, 6], dtype="int16"),
        "i32": np.array([-(2**31), 2**31 - 1, 0, 1, -1, 5, 6], dtype="int32"),
        "i64": np.array([-(2**63), 2**63 - 1, 0, 1, -1, 5, 6], dtype="int64"),
        "u8": np.array([0, 255, 1, 2, 3, 4, 5], dtype="uint8"),
        "u16": np.array([0, 65535, 1, 2, 3, 4, 5], dtype="uint16"),
        "u32": np.array([0, 2**32 - 1, 1, 2, 3, 4, 5], dtype="uint32"),
        "u64": np.array([0, 2**64 - 1, 1, 2, 3, 4, 5], dtype="uint64"),
        "b": [True, False, True, True, False, False, True],
        "b_na": pd.array([True, None, False, True, None, False, True], dtype="boolean"),
        "i_na": pd.array([1, None, 3, -4, None, 6, 7], dtype="Int64"),
        "s": pd.Series(["alpha", None, "", "ünïcödé", "x" * 40, "b", "c"], dtype=object),
        "cat": pd.Categorical(["lo", "hi", None, "lo", "mid", "hi", "lo"], categories=["lo", "mid", "hi"]),
        "ts": dt(["2020-01-01", None, "1969-12-31 23:59:59.999999999", "2262-04-11", "1970-01-01", "2000-02-29 12:00", "1677-09-22"]).astype("datetime64[ns]"),
        "ts_utc": dt(["2020-01-01T00:00:00Z", "2021-06-01T12:34:56.789Z", None, "1999-12-31T23:59:59Z", "2020-01-01", "2020-01-02", "2020-01-03"], utc=True),
        "ts_paris": dt(["2020-01-01 01:00", "2020-07-01 02:00", "2020-03-29 03:30", None, "2020-10-25 01:30", "2020-01-01", "2020-01-02"]).tz_localize("Europe/Paris"),
        "ts_ms": dt(["2020-01-01", "2020-01-02", None, "2020-01-04", "2020-01-05", "2020-01-06", "2020-01-07"]).astype("datetime64[ms]"),
        "ts_s": dt(["2020-01-01", None, "2020-01-03", "2020-01-04", "2020-01-05", "2020-01-06", "2020-01-07"]).astype("datetime64[s]"),
    }
)
write_all("types", types)
# The same frame with the other compressions each format offers.
types.to_parquet(os.path.join(HERE, "types_zstd.parquet"), compression="zstd")
types.to_parquet(os.path.join(HERE, "types_gzip.parquet"), compression="gzip")
types.to_parquet(os.path.join(HERE, "types_none.parquet"), compression=None)
types.to_feather(os.path.join(HERE, "types_zstd.feather"), compression="zstd")
types.to_feather(os.path.join(HERE, "types_none.feather"), compression="uncompressed")

named = pd.DataFrame(
    {"v": [1.0, 2.0, 3.0]}, index=pd.Index(["a", "b", "c"], name="key")
)
write_all("named_index", named)

unnamed = pd.DataFrame({"v": [10, 20, 30]}, index=pd.Index([7, 3, 5]))
write_all("unnamed_index", unnamed)

rng = pd.DataFrame({"v": ["x", "y", "z", "w"]}, index=pd.RangeIndex(10, 30, 5, name="r"))
write_all("range_index", rng)

named_rng = pd.DataFrame({"v": [1, 2]}, index=pd.RangeIndex(0, 2, name="row"))
write_all("named_range_index", named_rng)

multi = pd.DataFrame(
    {"v": [1.0, np.nan, 3.0, 4.0]},
    index=pd.MultiIndex.from_tuples(
        [("a", 1), ("a", 2), ("b", 1), ("b", 2)], names=["letter", None]
    ),
)
write_all("multi_index", multi)

dt_index = pd.DataFrame(
    {"v": [1, 2, 3]},
    index=pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03"], name="day").astype("datetime64[ns]"),
)
write_all("datetime_index", dt_index)

empty = pd.DataFrame({"f": pd.Series([], dtype="float64"), "s": pd.Series([], dtype=object)})
write_all("empty", empty)

# Plain Arrow data without pandas metadata: several record batches, a
# dictionary column whose dictionary changes from one batch to the next,
# int32-indexed, large strings, nulls, and all-null columns.
b1 = pa.record_batch(
    {
        "d": pa.DictionaryArray.from_arrays(pa.array([0, 1, None], pa.int32()), pa.array(["p", "q"])),
        "ls": pa.array(["one", None, "three"], pa.large_string()),
        "n": pa.array([None, None, None], pa.null()),
        "u": pa.array([1, None, 3], pa.uint16()),
        "t": pa.array([0, 1, None], pa.timestamp("us", tz="America/New_York")),
    }
)
b2 = pa.record_batch(
    {
        "d": pa.DictionaryArray.from_arrays(pa.array([1, 0], pa.int32()), pa.array(["r", "q"])),
        "ls": pa.array(["four", "five"], pa.large_string()),
        "n": pa.array([None, None], pa.null()),
        "u": pa.array([4, 5], pa.uint16()),
        "t": pa.array([3, 4], pa.timestamp("us", tz="America/New_York")),
    }
)
with pa.OSFile(os.path.join(HERE, "batches.arrows"), "wb") as f:
    with pa.ipc.new_stream(f, b1.schema) as w:
        w.write_batch(b1)
        w.write_batch(b2)
with pa.OSFile(os.path.join(HERE, "batches.arrow"), "wb") as f:
    # The file format forbids replacing a dictionary: unify first.
    t = pa.Table.from_batches([b1, b2]).unify_dictionaries()
    with pa.ipc.new_file(f, t.schema) as w:
        for b in t.to_batches():
            w.write_batch(b)
pq.write_table(pa.Table.from_batches([b1, b2]), os.path.join(HERE, "batches.parquet"), row_group_size=2)
expected["fixtures"]["batches"] = {
    "rows": 5,
    "index": {"range": True, "name": ""},
    "columns": [
        {"name": "d", "dtype": "category", "values": ["p", "q", None, "q", "r"]},
        {"name": "ls", "dtype": "string", "values": ["one", None, "three", "four", "five"]},
        {"name": "n", "dtype": "string", "values": [None] * 5},
        {"name": "u", "dtype": "uint16", "values": [1, None, 3, 4, 5]},
        {"name": "t", "dtype": "datetime", "values": [0, 1000, None, 3000, 4000]},
    ],
}

# Files the reader must refuse, each with the reason in the name.
pa.feather.write_feather(
    pa.table({"t": pa.array([10**11], pa.timestamp("s"))}),
    os.path.join(HERE, "bad_timestamp_overflow.feather"),
    compression="uncompressed",
)
pa.feather.write_feather(
    pa.table({"l": pa.array([[1, 2], [3]], pa.list_(pa.int64()))}),
    os.path.join(HERE, "bad_list.feather"),
    compression="uncompressed",
)
pq.write_table(
    pa.table({"l": pa.array([[1, 2], [3]], pa.list_(pa.int64()))}),
    os.path.join(HERE, "bad_list.parquet"),
)
pa.feather.write_feather(
    pa.table({"d": pa.DictionaryArray.from_arrays(pa.array([0], pa.int8()), pa.array([5], pa.int64()))}),
    os.path.join(HERE, "bad_int_dictionary.feather"),
    compression="uncompressed",
)

with open(os.path.join(HERE, "expected.json"), "w") as f:
    json.dump(expected, f, indent=1, ensure_ascii=False)
    f.write("\n")
