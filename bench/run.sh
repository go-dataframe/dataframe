#!/bin/sh
# Interleaved runs of the Go and pandas/polars operation benchmarks on the
# same table: ROUNDS rounds, each running both sides once; report medians
# with bench/medians.py. The host's load is printed before and after.
#   sh bench/run.sh 1000000 5 > /tmp/bench-1e6.txt
set -e
N=${1:-1000000}
ROUNDS=${2:-5}
uptime
go test -c -o /tmp/dataframe.bench . >/dev/null
for r in $(seq "$ROUNDS"); do
  DATAFRAME_BENCH_N=$N /tmp/dataframe.bench -test.run XXX -test.bench 'GroupBySum|SortValues|Filter|Merge|Memory' -test.benchtime 3x | grep Benchmark
  uv run -q --with pandas --with numpy --with polars --with pyarrow python bench/bench_pandas.py "$N" 3
done
uptime
