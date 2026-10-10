"""Compare ReadCSV with pandas.read_csv and polars.read_csv.

Run from the repository root:

    ~/.local/bin/uv run --with pandas --with numpy --with pyarrow --with polars \
        python bench/csv/bench.py [--rows 1000000 10000000] [--runs 5] [--dir DIR]

It writes the data files into DIR (default: a new temporary directory
outside the repository; nothing is written inside it), builds the Go test
binary, and then, for each size, runs the readers in turn (ours, pandas C,
pandas pyarrow, polars, ours, ...) as separate processes under
/usr/bin/time -l, so that the peak RSS of each is its own. Each process
times only the read call. The table gives medians over the runs; the load
average is printed before and after, because a loaded machine slows every
reader.
"""

import argparse
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PANDAS = r"""
import sys, time, pandas as pd
kw = {"engine": sys.argv[2]}
if sys.argv[3] == "1":
    kw["parse_dates"] = ["date"]
t = time.perf_counter()
df = pd.read_csv(sys.argv[1], **kw)
dt = time.perf_counter() - t
print("csvbench seconds=%.4f rows=%d mem=%d" % (dt, len(df), df.memory_usage(deep=True).sum()))
"""

POLARS = r"""
import sys, time, polars as pl
t = time.perf_counter()
df = pl.read_csv(sys.argv[1], try_parse_dates=sys.argv[2] == "1")
dt = time.perf_counter() - t
print("csvbench seconds=%.4f rows=%d mem=%d" % (dt, df.height, df.estimated_size()))
"""

BASELINE = r"""
import pandas, polars, pyarrow
print("csvbench seconds=0 rows=0 mem=0")
"""


def generate(path, n):
    """n rows: int id, a full-precision float, a two-decimal float, a short
    category and a date, the shape csv_bench_test.go uses."""
    rng = np.random.default_rng(1)
    days = rng.integers(0, 3650, n)
    df = pd.DataFrame({
        "id": np.arange(n),
        "x": rng.standard_normal(n),
        "y": rng.integers(0, 1000000, n) / 100,
        "cat": pd.Series(rng.integers(0, 50, n)).map(lambda k: "cat%d" % k),
        "date": (np.datetime64("2015-01-01") + days.astype("timedelta64[D]")).astype(str),
    })
    df.to_csv(path, index=False)


def run(cmd, env=None):
    p = subprocess.run(["/usr/bin/time", "-l"] + cmd, capture_output=True, text=True,
                       env={**os.environ, **(env or {})})
    if p.returncode != 0:
        raise RuntimeError("%s failed:\n%s\n%s" % (cmd, p.stdout, p.stderr))
    m = re.search(r"csvbench seconds=([\d.]+)", p.stdout)
    mem = re.search(r"(?:mem|heap_live)=(\d+)", p.stdout)
    rss = re.search(r"(\d+)\s+maximum resident set size", p.stderr)
    return float(m.group(1)), int(mem.group(1)), int(rss.group(1))


def uptime():
    return subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=[1_000_000, 10_000_000])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--dir", default=None)
    ap.add_argument("--dates", action="store_true", help="also parse the date column")
    args = ap.parse_args()
    d = args.dir or tempfile.mkdtemp(prefix="csvbench-")
    if os.path.realpath(d).startswith(os.path.realpath(ROOT)):
        sys.exit("the data directory must be outside the repository")
    gobin = os.path.join(d, "dataframe.test")
    subprocess.run(["go", "test", "-c", "-o", gobin, "."], cwd=ROOT, check=True,
                   env={**os.environ, "GOWORK": "off"})
    import polars
    print("machine:", platform.machine(), platform.platform(), os.cpu_count(), "CPUs")
    print("pandas", pd.__version__, "polars", polars.__version__, "python", platform.python_version())
    print("data in", d)
    base = run([sys.executable, "-c", BASELINE])[2]
    print("python+pandas+polars+pyarrow import alone: peak RSS %.0f MB" % (base / 2**20))
    dates = "1" if args.dates else "0"
    readers = {
        "go ReadCSV": lambda f: run([gobin, "-test.run", "^TestCSVBenchFile$"],
                                    {"CSV_BENCH_FILE": f, "CSV_BENCH_DATES": dates}),
        "pandas (c)": lambda f: run([sys.executable, "-c", PANDAS, f, "c", dates]),
        "pandas (pyarrow)": lambda f: run([sys.executable, "-c", PANDAS, f, "pyarrow", dates]),
        "polars": lambda f: run([sys.executable, "-c", POLARS, f, dates]),
    }
    for n in args.rows:
        f = os.path.join(d, "mixed_%d.csv" % n)
        if not os.path.exists(f):
            generate(f, n)
        size = os.path.getsize(f)
        print("\n%d rows, %.0f MB, parse dates: %s" % (n, size / 2**20, args.dates))
        print("before:", uptime())
        res = {k: [] for k in readers}
        for r in range(args.runs):
            for name, fn in readers.items():
                res[name].append(fn(f))
        print("after: ", uptime())
        print("| reader | median s | min s | max s | MB/s | data in memory MB | peak RSS MB |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for name, rs in res.items():
            t = [x[0] for x in rs]
            med = statistics.median(t)
            print("| %s | %.3f | %.3f | %.3f | %.0f | %.0f | %.0f |" % (
                name, med, min(t), max(t), size / 2**20 / med,
                statistics.median(x[1] for x in rs) / 2**20,
                statistics.median(x[2] for x in rs) / 2**20))
            print("raw", name, " ".join("%.3f" % x for x in t))


if __name__ == "__main__":
    main()
