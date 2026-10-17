#!/bin/sh
# Interleaved runs of the Go and pandas/polars operation benchmarks on the
# same table: ROUNDS rounds, each running both sides once; report medians
# with bench/medians.py. The host's load is printed before and after.
# THREADS=1 holds every side to one thread (GOMAXPROCS=1, pyarrow's pools
# sized 1, POLARS_MAX_THREADS=1).
#   sh bench/run.sh 1000000 5 > /tmp/bench-1e6.txt
#   THREADS=1 sh bench/run.sh 1000000 5 > /tmp/bench-1e6-1t.txt
set -e
if [ "$THREADS" = 1 ]; then export GOMAXPROCS=1 POLARS_MAX_THREADS=1; fi
N=${1:-1000000}
ROUNDS=${2:-5}
uptime
BIN=$(mktemp -d)/dataframe.bench
go test -c -o "$BIN" . >/dev/null
for r in $(seq "$ROUNDS"); do
  DATAFRAME_BENCH_N=$N "$BIN" -test.run XXX -test.bench 'GroupBySum|SortValues|Filter|Merge|ToDatetime|WriteCSVTable|StringEq|StringIsIn|Memory' -test.benchtime 3x | grep Benchmark
  uv run -q --with pandas --with numpy --with polars --with pyarrow python bench/bench_pandas.py "$N" 3
done
uptime
