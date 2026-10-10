"""Write the CSV corpus and record how pandas reads each file.

Run from the repository root:

    ~/.local/bin/uv run --with pandas --with numpy --with pyarrow \
        python testdata/csv/gen_fixtures.py

It (re)writes testdata/csv/*.csv and testdata/csv/fixtures.json. The Go
test csv_corpus_test.go reads each file with the same options and compares
the result with pandas', column by column. pandas is the oracle here only;
the deliberate differences are listed in docs/design.md (CSV section).
"""

import io
import json
import math
import os
import random

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

FILES = {
    "basic.csv": "id,x,name,flag\n1,0.5,alpha,True\n2,1.25,beta,False\n3,-3.0,gamma,True\n",
    "quotes.csv": (
        'a,b,c\n'
        '"plain","with ""doubled"" quotes","x,y"\n'
        '"ab"c,ab"c,"NA"\n'
        '"",  "lead",""""\n'
        '"a""b""",tail"q",\n'
    ),
    "newlines.csv": 'id,text\n1,"line one\nline two"\n2,"crlf\r\ninside"\n3,"cr\rinside"\n4,plain\n',
    "bom.csv": "﻿a,b\n1,x\n2,y\n",
    "crlf.csv": "a,b\r\n1,2\r\n\r\n3,4\r\n5,6",
    "cr.csv": "a,b\r1,x\r2,y\r",
    "ragged_short.csv": "a,b,c\n1,2,3\n4,5\n6\n7,8,9\n",
    "ragged_long.csv": 'a,b\n"x\ny",1\n\n2,3\n4,5,6\n',
    "ragged_long_late.csv": "a,b\n1,2\n3,4\n5,6,7\n",
    "trailing_sep.csv": "a,b\n1,2,\n3,4,\n",
    "implicit2.csv": "a,b\n1,2,3,4\n5,6\n",
    "bigints.csv": (
        "i64,u64,big,mixed,negbig,intna\n"
        "9223372036854775807,9223372036854775808,18446744073709551616,-1,-9223372036854775809,1\n"
        "-9223372036854775808,18446744073709551615,1,18446744073709551615,2,\n"
        "0,0,2,0,3,9007199254740993\n"
    ),
    "uintna.csv": "u\n18446744073709551615\nNA\n1\n",
    "bignum_float.csv": "a\n1.5\n18446744073709551616\n",
    "dates.csv": (
        "d,dt,eu,bad,mixed\n"
        "2020-01-01,2020-01-01 10:00:00,01/02/2020,2020-01-01,2020-01-01\n"
        "2020-02-29,2020-02-29 23:59:59,31/12/2021,not a date,2020-01-02 10:00\n"
        ",2021-06-30 00:00:01,,2020-01-03,\n"
    ),
    "dates_z.csv": "t\n2020-01-01T10:00:00Z\n2020-01-02T11:30:00Z\n",
    "empty_fields.csv": "a,b,c\n1,,x\n,,\n3,,z\n",
    "na_tokens.csv": (
        "tok,num\n"
        + "".join(f'"{t}",{i}\n' for i, t in enumerate([
            "", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan",
            "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN", "None",
            "n/a", "nan", "null"]))
    ),
    "na_lookalikes.csv": "a,b\nNA ,1\n na,2\nNan,3\nnone,4\nNULL,5\n",
    "whitespace.csv": "a,b,c\n 1 , 2.5 , x \n2,\t3.5\t,  \n   \n\t\n4,5, y\n",
    "dup_headers.csv": "a,a,a.1,,b,a\n1,2,3,4,5,6\n",
    "unicode.csv": "nom,ville,émoji\nZoë,Saint-Étienne,😀\n李,北京,\"a,😀\"\nŁódź,\"Kraków\nPL\",ß\n",
    "single_col.csv": "v\n1\n\n2\n3\n",
    "header_only.csv": "a,b,c\n",
    "header_only_nonl.csv": "a,b,c",
    "empty.csv": "",
    "blank_only.csv": "\n\n  \n",
    "comments.csv": "# a comment\na,b\n1,2 # trailing\n# full line\n3,4\n",
    "skiprows.csv": "title line\n\nx,y\n1,2\n3,4\n",
    "semicolon.csv": "a;b;c\n1;2,5;\"x;y\"\n3;4,5;z\n",
    "floats.csv": (
        "f,inf,mix,exp,neg0\n"
        "1.,inf,1,1e5,-0.0\n"
        ".5,-Infinity,2.5,1E-3,0\n"
        "0.1,+INF,3,-2.5e+300,-0\n"
        "123456789.123456789,iNf,4,1e400,5\n"
        "1e-400,Infinity,,6.02214076e23,6\n"
    ),
    "float_digits.csv": "x\n" + "".join(
        repr(v) + "\n" for v in [random.Random(42 + k).uniform(-1e6, 1e6) for k in range(200)]
        + [random.Random(7 + k).random() * 10.0 ** random.Random(k).randint(-30, 30) for k in range(200)]
    ),
    "float_text.csv": "x\n" + "".join(
        "%.*f\n" % (random.Random(k).randint(0, 9), random.Random(1000 + k).uniform(-1e4, 1e4)) for k in range(200)
    ) + "0.30000000000000004\n2.2250738585072011e-308\n4.9406564584124654e-324\n1.7976931348623157e308\n",
    "bools.csv": "a,b,c,d\nTrue,true,yes,True\nFALSE,false,no,1\ntrue,TRUE,yes,\n",
    "unclosed.csv": 'a,b\n1,2\n"3,4\n5,6\n',
    "no_newline.csv": "a,b\n1,2\n3,4",
    "single_quote.csv": "a,b\n'x,y',1\n'it''s',2\n",
    "options.csv": "id,name,score,when,skip\n1,ann,1.5,2020-01-01,a\n2,bob,-,2020-01-02,b\n3,cid,2.5,2020-01-03,c\n4,dan,3.5,2020-01-04,d\n",
    "dtypes.csv": "i8,f32,s,cat,b,i,f,bt\n1,1.5,007,x,1,1,2,True\n2,2.5,010,y,0,,3,false\n3,,,x,1,3,,TRUE\n",
    "names.csv": "1,2,3\n4,5,6\n",
    "trailing_empty_line.csv": "a,b\n1,2\n\n\n",
    "spaces_in_quotes.csv": 'a,b\n" 1 ",2\n"3"x,4\n',
    "only_quotes.csv": '"a"\n""\n"x"\n',
}

CASES = [
    ("basic", "basic.csv", {}),
    ("quotes", "quotes.csv", {}),
    ("newlines", "newlines.csv", {}),
    ("bom", "bom.csv", {}),
    ("crlf", "crlf.csv", {}),
    ("crlf_keep_blank", "crlf.csv", {"skip_blank_lines": False}),
    ("cr", "cr.csv", {}),
    ("ragged_short", "ragged_short.csv", {}),
    ("ragged_long", "ragged_long.csv", {}),
    ("ragged_long_late", "ragged_long_late.csv", {}),
    ("ragged_long_nrows", "ragged_long_late.csv", {"nrows": 2}),
    ("trailing_sep", "trailing_sep.csv", {}),
    ("implicit2", "implicit2.csv", {}),
    ("bigints", "bigints.csv", {}),
    ("uintna", "uintna.csv", {}),
    ("bignum_float", "bignum_float.csv", {}),
    ("dates_plain", "dates.csv", {}),
    ("dates_parsed", "dates.csv", {"parse_dates": ["d", "dt", "bad"]}),
    ("dates_mixed", "dates.csv", {"parse_dates": ["mixed"]}),
    ("dates_format", "dates.csv", {"parse_dates": ["eu"], "date_format": "%d/%m/%Y"}),
    ("dates_z", "dates_z.csv", {"parse_dates": ["t"]}),
    ("empty_fields", "empty_fields.csv", {}),
    ("na_tokens", "na_tokens.csv", {}),
    ("na_tokens_nodefault", "na_tokens.csv", {"keep_default_na": False}),
    ("na_tokens_custom", "na_tokens.csv", {"keep_default_na": False, "na_values": ["NA", "", "x"]}),
    ("na_lookalikes", "na_lookalikes.csv", {}),
    ("whitespace", "whitespace.csv", {}),
    ("whitespace_keep_blank", "whitespace.csv", {"skip_blank_lines": False}),
    ("dup_headers", "dup_headers.csv", {}),
    ("unicode", "unicode.csv", {}),
    ("single_col", "single_col.csv", {}),
    ("single_col_keep_blank", "single_col.csv", {"skip_blank_lines": False}),
    ("header_only", "header_only.csv", {}),
    ("header_only_nonl", "header_only_nonl.csv", {}),
    ("empty", "empty.csv", {}),
    ("blank_only", "blank_only.csv", {}),
    ("comments", "comments.csv", {"comment": "#"}),
    ("skiprows", "skiprows.csv", {"skiprows": 2}),
    ("semicolon", "semicolon.csv", {"sep": ";"}),
    ("floats", "floats.csv", {}),
    ("float_digits", "float_digits.csv", {}),
    ("float_text", "float_text.csv", {}),
    ("bools", "bools.csv", {}),
    ("bools_custom", "bools.csv", {"true_values": ["yes"], "false_values": ["no"]}),
    ("unclosed", "unclosed.csv", {}),
    ("no_newline", "no_newline.csv", {}),
    ("single_quote", "single_quote.csv", {"quotechar": "'"}),
    ("usecols", "options.csv", {"usecols": ["score", "id"]}),
    ("usecols_missing", "options.csv", {"usecols": ["id", "nope"]}),
    ("index_col", "options.csv", {"index_col": "name"}),
    ("index_col_usecols", "options.csv", {"index_col": "name", "usecols": ["name", "score"]}),
    ("index_pos", "options.csv", {"index_col": 0}),
    ("nrows", "options.csv", {"nrows": 2}),
    ("nrows0", "options.csv", {"nrows": 0}),
    ("na_values", "options.csv", {"na_values": ["-"]}),
    ("names_header0", "options.csv", {"names": ["A", "B", "C", "D", "E"], "header": 0}),
    ("names_short", "options.csv", {"names": ["C", "D", "E"], "header": 0}),
    ("dtypes", "dtypes.csv", {"dtype": {"i8": "int8", "f32": "float32", "s": "str", "cat": "category", "b": "bool", "f": "float64", "bt": "bool"}}),
    ("dtype_int", "dtypes.csv", {"dtype": {"i": "int64", "i8": "int64"}}),
    ("dtype_str_all", "basic.csv", {"dtype": {"id": "str", "x": "str", "flag": "str"}}),
    ("dtype_int_bad", "basic.csv", {"dtype": {"name": "int64"}}),
    ("names_noheader", "names.csv", {"names": ["x", "y", "z"]}),
    ("names_long", "names.csv", {"names": ["w", "x", "y", "z"]}),
    ("names_implicit", "names.csv", {"names": ["y", "z"]}),
    ("header_none", "names.csv", {"header": None}),
    ("trailing_empty_line", "trailing_empty_line.csv", {}),
    ("trailing_empty_line_keep", "trailing_empty_line.csv", {"skip_blank_lines": False}),
    ("spaces_in_quotes", "spaces_in_quotes.csv", {}),
    ("only_quotes", "only_quotes.csv", {}),
]


def encode(v):
    """A value as a string, None when missing."""
    if v is None or v is pd.NaT:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, (bool, np.bool_)):
        return "True" if v else "False"
    if isinstance(v, (float, np.floating)):
        return repr(float(v))
    if isinstance(v, pd.Timestamp):
        return "ns:%d" % v.as_unit("ns").value
    if v is pd.NA:
        return None
    return str(v)


def encode_column(s):
    return [encode(v) for v in s.tolist()] if not str(s.dtype).startswith("datetime64") else [
        None if pd.isna(v) else "ns:%d" % pd.Timestamp(v).as_unit("ns").value for v in s
    ]


def record(name, fname, kwargs):
    path = os.path.join(HERE, fname)
    out = {"name": name, "file": fname, "options": kwargs}
    try:
        df = pd.read_csv(path, **kwargs)
    except Exception as e:  # noqa: BLE001 - every pandas error is an expected outcome
        out["error"] = "%s: %s" % (type(e).__name__, str(e).strip())
        return out
    out["columns"] = [str(c) for c in df.columns]
    out["dtypes"] = [str(t) for t in df.dtypes]
    out["values"] = [encode_column(df[c]) for c in df.columns]
    # For parse_dates columns pandas left as text, what the instants are.
    alt = {}
    for c in kwargs.get("parse_dates", []):
        if c in df.columns and not str(df[c].dtype).startswith("datetime64"):
            try:
                alt[c] = encode_column(pd.to_datetime(df[c], format="ISO8601"))
            except Exception:  # noqa: BLE001
                pass
    if alt:
        out["alt"] = alt
    # pandas' default float parser is not correctly rounded; what the
    # correctly rounded one ("round_trip") reads, where it differs.
    rt = pd.read_csv(path, float_precision="round_trip", **kwargs)
    exact = {}
    for c in df.columns:
        if str(df[c].dtype) == "float64":
            v = encode_column(rt[c])
            if v != out["values"][list(df.columns).index(c)]:
                exact[str(c)] = v
    if exact:
        out["round_trip"] = exact
    # The writer's oracle: how pandas writes the frame it read.
    out["to_csv"] = df.to_csv(lineterminator="\n")
    ix = df.index
    levels = [ix.get_level_values(k) for k in range(ix.nlevels)]
    out["index"] = {
        "names": [None if n is None else str(n) for n in ix.names],
        "dtypes": [str(l.dtype) for l in levels],
        "values": [encode_column(pd.Series(l)) for l in levels],
    }
    return out


def main():
    for fname, text in FILES.items():
        with open(os.path.join(HERE, fname), "w", encoding="utf-8", newline="") as f:
            f.write(text)
    fixtures = {"pandas": pd.__version__, "cases": [record(*c) for c in CASES]}
    with open(os.path.join(HERE, "fixtures.json"), "w", encoding="utf-8") as f:
        json.dump(fixtures, f, indent=1, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    main()
