"""Medians of bench/run.sh output, one line per (library, operation)."""
import collections, re, statistics, sys

go = {"BenchmarkGroupBySum": "groupby_sum", "BenchmarkGroupBySumString": "groupby_sum_string",
      "BenchmarkSortValues": "sort_values", "BenchmarkFilter": "filter", "BenchmarkMerge": "merge",
      "BenchmarkToDatetime": "to_datetime", "BenchmarkToDatetimeFormat": "to_datetime_format",
      "BenchmarkWriteCSVTable": "write_csv", "BenchmarkStringEq": "string_eq",
      "BenchmarkStringIsIn": "string_isin"}
vals = collections.defaultdict(list)
for line in open(sys.argv[1]):
    m = re.match(r"(Benchmark\w+?)(?:-\d+)?\s+\d+\s+([\d.]+) ns/op(?:\s+([\d.]+) MB)?", line)
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
ops = ["groupby_sum", "groupby_sum_string", "sort_values", "filter", "merge",
       "to_datetime", "to_datetime_format", "write_csv", "string_eq", "string_isin", "memory MB"]
print("| operation | go-dataframe | pandas | go / pandas | polars |")
print("|---|---:|---:|---:|---:|")
for op in ops:
    row = [op]
    med = {}
    for lib in ["go", "pandas", "polars"]:
        v = vals.get((lib, op))
        med[lib] = statistics.median(v) if v else None
    for lib in ["go", "pandas"]:
        row.append(f"{med[lib]:.1f}" if med[lib] is not None else "-")
    ratio = med["go"] / med["pandas"] if med["go"] and med["pandas"] else None
    row.append(f"{ratio:.2f}" if ratio is not None and op != "memory MB" else "-")
    row.append(f"{med['polars']:.1f}" if med["polars"] is not None else "-")
    print("| " + " | ".join(row) + " |")
