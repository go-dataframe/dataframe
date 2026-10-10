"""Record pandas' answers for the operations the Go tests check.

Run from the repository root:

    uv run --with pandas --with numpy --with pyarrow python testdata/gen_fixtures.py

It writes testdata/fixtures.json: for each case, the input frame(s), the
operation and its arguments, and what pandas returns (values, dtypes, index),
or the fact that pandas raises. Nothing here runs in CI; the JSON is
committed and embedded by the Go tests.
"""

import json
import math
import sys

import numpy as np
import pandas as pd


def norm_dtype(dt):
    s = str(dt)
    if s in ("str", "object", "string"):
        return "string"
    if s.startswith("datetime64"):
        return "datetime"
    if s == "category":
        return "category"
    if s.startswith("period"):
        return "period"
    if s in ("Int64",):
        return "int64"
    if s in ("boolean",):
        return "bool"
    return s


def enc_value(v, kind):
    if v is None or v is pd.NaT or (isinstance(v, float) and math.isnan(v)):
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if kind == "datetime":
        return int(pd.Timestamp(v).as_unit("ns").value)
    if kind in ("category", "period", "string"):
        return str(v)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        if math.isinf(f):
            return "inf" if f > 0 else "-inf"
        return f
    return v


def enc_values(values, dtype):
    kind = norm_dtype(dtype)
    return [enc_value(v, kind) for v in values]


def enc_index(ix):
    if isinstance(ix, pd.RangeIndex):
        return {"range": [ix.start, ix.stop, ix.step], "names": [ix.name or ""]}
    levels = []
    for k in range(ix.nlevels):
        lv = ix.get_level_values(k)
        levels.append({"dtype": norm_dtype(lv.dtype), "values": enc_values(list(lv), lv.dtype)})
    return {"names": [n or "" if n is not None else "" for n in ix.names], "levels": levels}


def enc_frame(df):
    return {
        "columns": [
            {"name": str(c), "dtype": norm_dtype(df[c].dtype), "values": enc_values(list(df[c]), df[c].dtype)}
            for c in df.columns
        ],
        "index": enc_index(df.index),
    }


def enc_series(s):
    return {
        "name": "" if s.name is None else str(s.name),
        "dtype": norm_dtype(s.dtype),
        "values": enc_values(list(s), s.dtype),
        "index": enc_index(s.index),
    }


def enc_result(r):
    if isinstance(r, pd.DataFrame):
        return {"kind": "frame", "frame": enc_frame(r)}
    if isinstance(r, pd.Series):
        return {"kind": "series", "series": enc_series(r)}
    if isinstance(r, str):
        return {"kind": "text", "text": r}
    return {"kind": "scalar", "value": enc_value(r, "")}


cases = []


def case(name, op, inputs, compute, **args):
    try:
        r = compute()
        expect = enc_result(r)
    except Exception as e:  # pandas raising is an answer too
        expect = {"kind": "error", "error": type(e).__name__ + ": " + str(e)}
    cases.append({"name": name, "op": op, "args": args, "inputs": [enc_frame(i) for i in inputs], "expect": expect})


rng = np.random.default_rng(42)
N = 24
base = pd.DataFrame({
    "key": rng.choice(["a", "b", "c", None], N),
    "k2": rng.integers(0, 3, N),
    "x": np.round(rng.normal(10, 3, N), 3),
    "y": rng.integers(-50, 50, N),
    "flag": rng.choice([True, False], N),
    "s": rng.choice(["lyon", "paris", "nice", "brest"], N),
})
base.loc[[2, 7, 11], "x"] = np.nan
base.loc[[4, 9], "y"] = 7  # ties for the stable sort
small = pd.DataFrame({"a": [3.0, 1.0, np.nan, 2.0, 1.0], "b": ["x", "y", "z", "w", "v"], "c": [5, 3, 1, 4, 2]})
labelled = small.set_index("b")

# 1. Core
case("head", "head", [base], lambda: base.head(5), n=5)
case("head_neg", "head", [base], lambda: base.head(-20), n=-20)
case("tail", "tail", [base], lambda: base.tail(3), n=3)
case("tail_big", "tail", [small], lambda: small.tail(10), n=10)
case("transpose", "T", [small[["a", "c"]]], lambda: small[["a", "c"]].T)
case("transpose_mixed", "T", [small], lambda: small.T.astype(str).replace("nan", "NaN"))
case("round", "round", [base], lambda: base.round(1), decimals=1)
case("round_neg", "round", [pd.DataFrame({"v": [1234.5, 2345.5, -15.0]})], lambda: pd.DataFrame({"v": [1234.5, 2345.5, -15.0]}).round(-1), decimals=-1)
case("rename", "rename", [small], lambda: small.rename(columns={"a": "A", "zz": "q"}), mapping={"a": "A", "zz": "q"})
case("astype_float", "astype", [small], lambda: small.astype({"c": "float64"}), dtypes={"c": "float64"})
case("astype_str", "astype", [small], lambda: small.astype({"c": "str"}), dtypes={"c": "string"})
case("astype_int32", "astype", [small], lambda: small.astype({"c": "int32"}), dtypes={"c": "int32"})
case("astype_category", "astype", [small], lambda: small.astype({"b": "category"}), dtypes={"b": "category"})
case("copy", "copy", [small], lambda: small.copy())

# 2. Selection
case("iloc_slice", "iloc_slice", [base], lambda: base.iloc[3:9], start=3, stop=9)
case("iloc_slice_neg", "iloc_slice", [base], lambda: base.iloc[-4:-1], start=-4, stop=-1)
case("iloc_pos", "iloc_pos", [base], lambda: base.iloc[[5, 0, -1]], pos=[5, 0, -1])
case("iloc_cols", "iloc_cols", [base], lambda: base.iloc[[1, 2], 1:3], pos=[1, 2], cstart=1, cstop=3)
case("loc_labels", "loc_labels", [labelled], lambda: labelled.loc[["z", "x"]], labels=["z", "x"])
case("loc_range", "loc_range", [labelled.sort_index()], lambda: labelled.sort_index().loc["w":"y"], start="w", stop="y")
case("loc_range_absent", "loc_range", [labelled.sort_index()], lambda: labelled.sort_index().loc["wa":"xa"], start="wa", stop="xa")
case("mask_gt", "filter_gt", [base], lambda: base[base["x"] > 10], column="x", value=10)
case("mask_eq_str", "filter_eq", [base], lambda: base[base["s"] == "lyon"], column="s", value="lyon")
case("mask_ne_nan", "filter_ne", [base], lambda: base[base["x"] != 10.0], column="x", value=10.0)
case("set_index", "set_index", [base], lambda: base.set_index("s"), keys=["s"])
case("set_index_multi", "set_index", [base], lambda: base.set_index(["key", "k2"]), keys=["key", "k2"])
case("reset_index", "reset_index", [labelled], lambda: labelled.reset_index(), drop=False)
case("reset_index_range", "reset_index", [small], lambda: small.reset_index(), drop=False)
case("reset_index_drop", "reset_index", [labelled], lambda: labelled.reset_index(drop=True), drop=True)
case("reset_index_multi", "reset_index", [base.set_index(["key", "k2"])], lambda: base.set_index(["key", "k2"]).reset_index(), drop=False)
case("sort_values", "sort_values", [base], lambda: base.sort_values("y", kind="stable"), by=["y"], ascending=[True])
case("sort_values_desc", "sort_values", [base], lambda: base.sort_values("y", ascending=False, kind="stable"), by=["y"], ascending=[False])
case("sort_values_nan", "sort_values", [base], lambda: base.sort_values("x", kind="stable"), by=["x"], ascending=[True])
case("sort_values_nan_desc", "sort_values", [base], lambda: base.sort_values("x", ascending=False, kind="stable"), by=["x"], ascending=[False])
case("sort_values_multi", "sort_values", [base], lambda: base.sort_values(["s", "y"], ascending=[True, False], kind="stable"), by=["s", "y"], ascending=[True, False])
case("sort_values_str_nan", "sort_values", [base], lambda: base.sort_values(["key", "k2"], kind="stable"), by=["key", "k2"], ascending=[True])
case("sort_index", "sort_index", [labelled], lambda: labelled.sort_index(), ascending=True)
case("sort_index_desc", "sort_index", [base.set_index("y")], lambda: base.set_index("y").sort_index(ascending=False, kind="stable"), ascending=False)
case("reindex", "reindex", [labelled], lambda: labelled.reindex(["y", "q", "x"]), labels=["y", "q", "x"])
case("query_simple", "query", [base], lambda: base.query("x > 10 and y < 20"), expr="x > 10 and y < 20")
case("query_str", "query", [base], lambda: base.query("s == 'lyon' or s == \"nice\""), expr="s == 'lyon' or s == \"nice\"")
case("query_in", "query", [base], lambda: base.query("s in ['lyon', 'brest'] and not flag"), expr="s in ['lyon', 'brest'] and not flag")
case("query_not_in", "query", [base], lambda: base.query("s not in ['lyon', 'brest']"), expr="s not in ['lyon', 'brest']")
case("query_arith", "query", [base], lambda: base.query("x * 2 + y > 30"), expr="x * 2 + y > 30")
case("query_chain", "query", [base], lambda: base.query("5 < x <= 11"), expr="5 < x <= 11")
case("query_var", "query", [base], lambda: base.query("y >= @lim", local_dict={"lim": 10}), expr="y >= @lim", vars={"lim": 10})
case("query_paren", "query", [base], lambda: base.query("(k2 == 1 | k2 == 2) & ~(y < 0)"), expr="(k2 == 1 | k2 == 2) & ~(y < 0)")
case("query_index", "query", [small], lambda: small.query("index > 1"), expr="index > 1")
case("query_mod", "query", [base], lambda: base.query("y % 3 == 0 and -y < 10"), expr="y % 3 == 0 and -y < 10")
case("query_pow", "query", [base], lambda: base.query("x ** 2 > 120"), expr="x ** 2 > 120")
case("query_isna", "query", [base], lambda: base.query("x.isna()"), expr="x.isna()")
case("query_floordiv", "query", [base], lambda: base.query("y // 7 == 1"), expr="y // 7 == 1")

# 3. Missing data
case("dropna", "dropna", [base], lambda: base.dropna(), how="any", subset=[])
case("dropna_subset", "dropna", [base], lambda: base.dropna(subset=["x"]), how="any", subset=["x"])
two_na = pd.DataFrame({"p": [1.0, np.nan, np.nan, 4.0], "q": [np.nan, np.nan, 3.0, 4.0]})
case("dropna_all", "dropna", [two_na], lambda: two_na.dropna(how="all"), how="all", subset=[])
case("dropna_thresh", "dropna_thresh", [two_na], lambda: two_na.dropna(thresh=2), thresh=2)
case("fillna", "fillna", [two_na], lambda: two_na.fillna(0.5), value=0.5)
case("fillna_map", "fillna_map", [two_na], lambda: two_na.fillna({"p": -1.0}), values={"p": -1.0})
case("ffill", "ffill", [two_na], lambda: two_na.ffill())
case("bfill", "bfill", [two_na], lambda: two_na.bfill())
case("isna", "isna", [two_na], lambda: two_na.isna())

# 4. Computation
num = base[["x", "y", "flag"]]
case("sum", "reduce", [num], lambda: num.sum(), fn="sum")
case("mean", "reduce", [num], lambda: num.mean(), fn="mean")
case("std", "reduce", [num], lambda: num.std(), fn="std")
case("var", "reduce", [num], lambda: num.var(), fn="var")
case("min", "reduce", [base[["x", "y"]]], lambda: base[["x", "y"]].min(), fn="min")
case("max", "reduce", [base[["x", "y"]]], lambda: base[["x", "y"]].max(), fn="max")
case("median", "reduce", [num], lambda: num.median(), fn="median")
case("count", "reduce", [base], lambda: base.count(), fn="count")
case("describe", "describe", [base], lambda: base.describe())
case("describe_text", "describe", [base[["s"]]], lambda: base[["s"]].describe().astype(str))
case("value_counts", "value_counts", [base], lambda: base["s"].value_counts(), column="s")
case("unique", "unique", [base], lambda: pd.Series(base["s"].unique(), name="s"), column="s")
case("nunique", "nunique", [base], lambda: base["s"].nunique(), column="s")
case("quantile", "quantile", [base], lambda: base["x"].quantile(0.3), column="x", q=0.3)
s1 = pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"], name="v")
s2 = pd.Series([10.0, 20.0, 30.0], index=["b", "c", "d"], name="v")
pair = pd.DataFrame({"v": [1.0, 2.0, 3.0]}, index=["a", "b", "c"]), pd.DataFrame({"v": [10.0, 20.0, 30.0]}, index=["b", "c", "d"])
case("align_add", "series_arith", list(pair), lambda: s1 + s2, opname="add")
case("align_mul", "series_arith", list(pair), lambda: s1 * s2, opname="mul")
case("col_add", "col_arith", [base], lambda: base["x"] + base["y"], a="x", b="y", opname="add")
case("col_div_int", "col_arith", [base], lambda: base["y"] / base["k2"], a="y", b="k2", opname="div")
case("col_floordiv", "col_arith", [base], lambda: base["y"] // 7, a="y", scalar=7, opname="floordiv")
case("col_mod", "col_arith", [base], lambda: base["y"] % 7, a="y", scalar=7, opname="mod")
case("col_mod_float", "col_arith", [base], lambda: base["x"] % -3.0, a="x", scalar=-3.0, opname="mod")
case("col_pow", "col_arith", [base], lambda: base["k2"] ** 2, a="k2", scalar=2, opname="pow")
case("col_sub_scalar", "col_arith", [base], lambda: base["x"] - 1.5, a="x", scalar=1.5, opname="sub")
case("str_concat", "col_arith", [base], lambda: base["s"] + base["s"], a="s", b="s", opname="add")
case("isin", "isin", [base], lambda: base["s"].isin(["nice", "brest"]), column="s", values=["nice", "brest"])
case("between", "between", [base], lambda: base["y"].between(-10, 10), column="y", lo=-10, hi=10)
case("frame_add_scalar", "frame_arith", [base[["x", "y"]]], lambda: base[["x", "y"]] + 1, opname="add", scalar=1)
fa = pd.DataFrame({"p": [1.0, 2.0], "q": [3.0, 4.0]}, index=[0, 1])
fb = pd.DataFrame({"q": [10.0, 20.0], "r": [5.0, 6.0]}, index=[1, 2])
case("frame_add_frame", "frame_arith", [fa, fb], lambda: fa + fb, opname="add")

# Windows and running totals
wv = pd.DataFrame({"v": [1.0, 3.0, np.nan, 4.0, 10.0, 2.0, 7.0]})
case("shift", "window", [wv], lambda: wv["v"].shift(2), fn="shift", n=2)
case("shift_neg", "window", [wv], lambda: wv["v"].shift(-1), fn="shift", n=-1)
case("diff", "window", [wv], lambda: wv["v"].diff(), fn="diff", n=1)
case("pct_change", "window", [wv], lambda: wv["v"].pct_change(fill_method=None), fn="pct_change", n=1)
case("cumsum", "window", [wv], lambda: wv["v"].cumsum(), fn="cumsum")
case("cumprod", "window", [wv], lambda: wv["v"].cumprod(), fn="cumprod")
case("cummax", "window", [wv], lambda: wv["v"].cummax(), fn="cummax")
case("cummin", "window", [wv], lambda: wv["v"].cummin(), fn="cummin")
for fn in ["sum", "mean", "std", "min", "max", "median"]:
    case("rolling3_" + fn, "rolling", [wv], lambda fn=fn: getattr(wv["v"].rolling(3), fn)(), window=3, min_periods=3, fn=fn)
case("rolling_minp", "rolling", [wv], lambda: wv["v"].rolling(3, min_periods=1).mean(), window=3, min_periods=1, fn="mean")

# 5. Group-by, concat, merge, reshape
case("groupby_sum", "groupby", [base], lambda: base.groupby("key").sum(numeric_only=True), keys=["key"], fn="sum")
case("groupby_mean", "groupby", [base], lambda: base.groupby("key").mean(numeric_only=True), keys=["key"], fn="mean")
case("groupby_count", "groupby", [base], lambda: base.groupby("key").count(), keys=["key"], fn="count")
case("groupby_min", "groupby", [base], lambda: base.groupby("key").min(), keys=["key"], fn="min")
case("groupby_max", "groupby", [base], lambda: base.groupby("key").max(), keys=["key"], fn="max")
case("groupby_first", "groupby", [base], lambda: base.groupby("key").first(), keys=["key"], fn="first")
case("groupby_last", "groupby", [base], lambda: base.groupby("key").last(), keys=["key"], fn="last")
case("groupby_std", "groupby", [base], lambda: base.groupby("key").std(numeric_only=True), keys=["key"], fn="std")
case("groupby_var", "groupby", [base], lambda: base.groupby("key").var(numeric_only=True), keys=["key"], fn="var")
case("groupby_median", "groupby", [base], lambda: base.groupby("key").median(numeric_only=True), keys=["key"], fn="median")
case("groupby_nunique", "groupby", [base], lambda: base.groupby("key").nunique(), keys=["key"], fn="nunique")
case("groupby_size", "groupby_size", [base], lambda: base.groupby("key").size(), keys=["key"])
case("groupby_multi", "groupby", [base], lambda: base.groupby(["s", "k2"]).sum(numeric_only=True), keys=["s", "k2"], fn="sum")
case("groupby_int_key", "groupby", [base], lambda: base.groupby("y").sum(numeric_only=True), keys=["y"], fn="sum")
case("groupby_agg", "groupby_agg", [base], lambda: base.groupby("s").agg(x=("x", "mean"), y_sum=("y", "sum"), y_max=("y", "max")),
     keys=["s"], specs=[["x", ["mean"]], ["y", ["sum", "max"]]])
case("groupby_transform", "groupby_transform", [base], lambda: base.groupby("s")["y"].transform("mean"), keys=["s"], column="y", fn="mean")
c1 = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
c2 = pd.DataFrame({"b": ["z"], "c": [1.5]}, index=[7])
case("concat_rows", "concat", [c1, c2], lambda: pd.concat([c1, c2]), axis=0, ignore_index=False, inner=False)
case("concat_rows_ignore", "concat", [c1, c2], lambda: pd.concat([c1, c2], ignore_index=True), axis=0, ignore_index=True, inner=False)
case("concat_rows_inner", "concat", [c1, c2], lambda: pd.concat([c1, c2], join="inner"), axis=0, ignore_index=False, inner=True)
c3 = pd.DataFrame({"d": [9, 8, 7]}, index=[1, 2, 3])
case("concat_cols", "concat", [c1, c3], lambda: pd.concat([c1, c3], axis=1), axis=1, ignore_index=False, inner=False)
case("concat_cols_inner", "concat", [c1, c3], lambda: pd.concat([c1, c3], axis=1, join="inner"), axis=1, ignore_index=False, inner=True)
L = pd.DataFrame({"k": ["b", "a", "c", "a", None], "k2": [1, 1, 2, 2, 1], "lv": [1.0, 2.0, 3.0, 4.0, 5.0]})
R = pd.DataFrame({"k": ["a", "c", "d", "a", None], "k2": [1, 2, 2, 2, 1], "rv": [10, 20, 30, 40, 50]})
for how in ["inner", "left", "right", "outer"]:
    case("merge_" + how, "merge", [L, R], lambda how=how: pd.merge(L, R, on="k", how=how), on=["k"], how=how)
    case("merge2_" + how, "merge", [L, R], lambda how=how: pd.merge(L, R, on=["k", "k2"], how=how), on=["k", "k2"], how=how)
R2 = R.rename(columns={"k": "key"})
case("merge_left_right_on", "merge", [L, R2], lambda: pd.merge(L, R2, left_on="k", right_on="key"), left_on=["k"], right_on=["key"], how="inner")
case("merge_indicator", "merge", [L, R], lambda: pd.merge(L, R, on="k", how="outer", indicator=True), on=["k"], how="outer", indicator=True)
case("merge_cross", "merge", [L.head(2), R.head(2)], lambda: pd.merge(L.head(2), R.head(2), how="cross"), how="cross")
J1 = pd.DataFrame({"v": [1, 2, 3]}, index=["a", "b", "c"])
J2 = pd.DataFrame({"w": [10.0, 20.0]}, index=["c", "a"])
case("join_left", "join", [J1, J2], lambda: J1.join(J2), how="left")
case("join_inner", "join", [J1, J2], lambda: J1.join(J2, how="inner"), how="inner")
case("join_outer", "join", [J1, J2], lambda: J1.join(J2, how="outer"), how="outer")
long = pd.DataFrame({"date": ["d1", "d1", "d2", "d2", "d3"], "city": ["lyon", "nice", "lyon", "nice", "lyon"], "temp": [10.0, 15.0, 11.0, 16.0, 9.0]})
case("pivot", "pivot", [long], lambda: long.pivot(index="date", columns="city", values="temp"), index="date", columns="city", values="temp")
dup = pd.concat([long, long.assign(temp=long.temp + 1)])
case("pivot_table", "pivot_table", [dup], lambda: dup.pivot_table(index="date", columns="city", values="temp", aggfunc="mean"), index="date", columns="city", values="temp", aggfunc="mean")
case("pivot_table_sum", "pivot_table", [dup], lambda: dup.pivot_table(index="date", columns="city", values="temp", aggfunc="sum"), index="date", columns="city", values="temp", aggfunc="sum")
wide = pd.DataFrame({"p": [1.0, np.nan], "q": [3.0, 4.0]}, index=["r1", "r2"])
case("stack", "stack", [wide], lambda: wide.stack(future_stack=True).dropna())
case("unstack", "unstack", [long.set_index(["date", "city"])], lambda: long.set_index(["date", "city"])["temp"].unstack(), column="temp")
case("melt", "melt", [long], lambda: long.melt(id_vars=["date"], value_vars=["temp"]), id_vars=["date"], value_vars=["temp"])

# 6. Time series
times = pd.DataFrame({"when": ["2024-01-01 00:30", "2024-01-01 05:10", "2024-01-02 23:59", "2024-01-05 12:00",
                               "2024-02-10 08:00", "2024-03-31 23:00", "2024-04-01 00:00"], "v": [1.0, 2.0, 3.0, 4.0, np.nan, 6.0, 7.0], "n": [1, 2, 3, 4, 5, 6, 7]})
case("to_datetime", "to_datetime", [times], lambda: pd.to_datetime(times["when"]).astype("datetime64[ns]"), column="when")
case("to_datetime_fmt", "to_datetime", [pd.DataFrame({"d": ["31/12/2023", "01/02/2024"]})], lambda: pd.to_datetime(pd.Series(["31/12/2023", "01/02/2024"], name="d"), format="%d/%m/%Y").astype("datetime64[ns]"), column="d", format="%d/%m/%Y")
ts = times.assign(when=pd.to_datetime(times["when"]).astype("datetime64[ns]")).set_index("when")
for rule, fn in [("D", "sum"), ("D", "mean"), ("h", "count"), ("6h", "sum"), ("W", "sum"), ("ME", "sum"), ("MS", "mean"), ("QE", "max"), ("YE", "sum"), ("15min", "count")]:
    case(f"resample_{rule}_{fn}", "resample", [ts], lambda rule=rule, fn=fn: getattr(ts.resample(rule), fn)(), rule=rule, fn=fn)
for freq in ["M", "Q", "Y", "D", "W"]:
    case(f"to_period_{freq}", "to_period", [ts], lambda freq=freq: ts.index.to_period(freq).astype(str).to_series(index=range(len(ts))).rename(None), freq=freq)
case("dt_fields", "dt_fields", [ts.reset_index()], lambda: pd.DataFrame({
    "year": ts.index.year, "month": ts.index.month, "day": ts.index.day, "hour": ts.index.hour,
    "dayofweek": ts.index.dayofweek, "dayofyear": ts.index.dayofyear}).astype("int64"), column="when")

# 8. Display
case("repr_small", "repr", [small], lambda: repr(small))
case("repr_labelled", "repr", [labelled], lambda: repr(labelled))
case("repr_base", "repr", [base], lambda: repr(base))
case("repr_dates", "repr", [ts], lambda: repr(ts))
case("repr_series", "repr_series", [small], lambda: repr(small["a"]), column="a")
big = pd.DataFrame({"i": np.arange(100), "f": np.arange(100) / 8.0})
case("repr_truncated", "repr", [big], lambda: repr(big))
case("repr_sci", "repr", [pd.DataFrame({"tiny": [1e-8, 2.5e-7], "v": [1.0, 2.0]})], lambda: repr(pd.DataFrame({"tiny": [1e-8, 2.5e-7], "v": [1.0, 2.0]})))
case("repr_multi", "repr", [base.groupby(["s", "k2"]).sum(numeric_only=True)], lambda: repr(base.groupby(["s", "k2"]).sum(numeric_only=True)))

mixed = pd.DataFrame({"when": pd.to_datetime(["2024-01-01 00:00", "2024-01-02 10:30"]).astype("datetime64[ns]"), "kwh": [3.5, -0.25],
                      "longname": ["ab", "c"], "b": [True, False], "n": [-1, 20], "big": [123456789.0, 1.5]})
case("repr_mixed", "repr", [mixed], lambda: repr(mixed))
case("repr_series_named", "repr_series", [labelled], lambda: repr(labelled["c"]), column="c")

out = {"pandas": pd.__version__, "numpy": np.__version__, "cases": cases}
with open(sys.argv[1] if len(sys.argv) > 1 else "testdata/fixtures.json", "w") as f:
    json.dump(out, f, sort_keys=True, separators=(",", ":"))
print(len(cases), "cases")
