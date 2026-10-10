"""Medians of bench/run.sh output, one line per (library, operation)."""
import collections, re, statistics, sys

go = {"BenchmarkGroupBySum": "groupby_sum", "BenchmarkGroupBySumString": "groupby_sum_string",
      "BenchmarkSortValues": "sort_values", "BenchmarkFilter": "filter", "BenchmarkMerge": "merge"}
vals = collections.defaultdict(list)
for line in open(sys.argv[1]):
    m = re.match(r"(Benchmark\w+)-\d+\s+\d+\s+([\d.]+) ns/op(?:\s+([\d.]+) MB)?", line)
    if m:
        if m.group(1) == "BenchmarkMemory":
            vals[("go", "memory MB")].append(float(m.group(3)))
        else:
            vals[("go", go[m.group(1)])].append(float(m.group(2)) / 1e6)
        continue
    m = re.match(r"(pandas|polars) (\w+) ([\d.]+) (ms|MB)", line)
    if m:
        key = m.group(2) if m.group(4) == "ms" else "memory MB"
        vals[(m.group(1), key)].append(float(m.group(3)))
ops = ["groupby_sum", "groupby_sum_string", "sort_values", "filter", "merge", "memory MB"]
print("| operation | go-dataframe | pandas | polars |")
print("|---|---:|---:|---:|")
for op in ops:
    row = [op]
    for lib in ["go", "pandas", "polars"]:
        v = vals.get((lib, op))
        row.append(f"{statistics.median(v):.1f}" if v else "-")
    print("| " + " | ".join(row) + " |")
