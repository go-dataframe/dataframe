// Command harness measures go-dataframe against pandas and pyarrow, and
// two builds of go-dataframe against each other, in interleaved rounds:
// every side runs once per round, in turn, so that a host whose load
// drifts slows every side alike. It prints the median of each operation
// over the rounds, in milliseconds, and each side's ratio to pandas.
//
// Build the test binaries of each tree first, then run from the
// repository root:
//
//	go test -c -o /tmp/new/df.test . && go test -c -o /tmp/new/parquetio.test ./parquetio &&
//	    go test -c -o /tmp/new/arrowio.test ./arrowio
//	(the same into /tmp/main from a checkout of main)
//	cd bench/harness && go run . measure -rounds 7 -threads 1 main=/tmp/main new=/tmp/new
//
// With golden instead of measure, it compares the files two builds wrote
// with TestGoldenFiles (parquetio), see golden.go:
//
//	cd bench/harness && go run . golden /tmp/golden-main /tmp/golden-new
//
// -threads 1 holds every side to one thread: GOMAXPROCS=1 for Go, and
// THREADS=1 for the pandas scripts (pyarrow's pools sized 1,
// use_threads=False on reads, POLARS_MAX_THREADS=1). -threads 0 leaves
// each side's default.
package main

import (
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
)

// ops are the operations reported, in order, with the Go benchmark that
// measures each (in the root package unless io) and its pandas name.
var ops = []struct {
	name, bench string
	io          int // position in the I/O line of arrowio/testdata/bench_pandas.py, or -1
}{
	{"write Parquet", "BenchmarkWriteParquet1e6", 0},
	{"read Parquet", "BenchmarkReadParquet1e6", 1},
	{"write IPC file", "BenchmarkWriteIPC1e6", 2},
	{"read IPC file", "BenchmarkReadIPC1e6", 3},
	{"groupby_sum", "BenchmarkGroupBySum", -1},
	{"groupby_sum_string", "BenchmarkGroupBySumString", -1},
	{"sort_values", "BenchmarkSortValues", -1},
	{"filter", "BenchmarkFilter", -1},
	{"merge", "BenchmarkMerge", -1},
	{"to_datetime", "BenchmarkToDatetime", -1},
	{"to_datetime_format", "BenchmarkToDatetimeFormat", -1},
	{"write_csv", "BenchmarkWriteCSVTable", -1},
	{"string_eq", "BenchmarkStringEq", -1},
	{"string_isin", "BenchmarkStringIsIn", -1},
}

const opBenchmarks = "GroupBySum|SortValues|Filter|Merge|ToDatetime|WriteCSVTable|StringEq|StringIsIn|Memory"

func main() {
	if len(os.Args) < 2 {
		fail("usage: harness measure [flags] name=dir... | harness golden dirA dirB")
	}
	switch os.Args[1] {
	case "measure":
		measure(os.Args[2:])
	case "snappy":
		snappyCompare(5)
	case "golden":
		if len(os.Args) != 4 {
			fail("usage: harness golden dirA dirB")
		}
		if !golden(os.Args[2], os.Args[3]) {
			os.Exit(1)
		}
	default:
		fail("unknown command %q", os.Args[1])
	}
}

func measure(args []string) {
	flag := flag.NewFlagSet("measure", flag.ExitOnError)
	rounds := flag.Int("rounds", 7, "interleaved rounds")
	threads := flag.Int("threads", 1, "1: every side on one thread; 0: defaults")
	root := flag.String("root", "../..", "the repository, for the pandas scripts")
	uv := flag.String("uv", "uv", "the uv command that runs the pandas scripts")
	flag.Parse(args)
	type side struct{ name, dir string }
	var sides []side
	for _, a := range flag.Args() {
		name, dir, ok := strings.Cut(a, "=")
		if !ok {
			fail("side %q is not name=dir", a)
		}
		sides = append(sides, side{name, dir})
	}
	if len(sides) == 0 {
		fail("no side to measure: give name=dir of test binaries")
	}
	env := os.Environ()
	if *threads == 1 {
		env = append(env, "GOMAXPROCS=1", "THREADS=1", "POLARS_MAX_THREADS=1")
	}
	samples := map[string]map[string][]float64{} // side → op → ms
	add := func(side, op string, ms float64) {
		if samples[side] == nil {
			samples[side] = map[string][]float64{}
		}
		samples[side][op] = append(samples[side][op], ms)
	}
	fmt.Fprintln(os.Stderr, "load before:", load())
	for r := range *rounds {
		fmt.Fprintf(os.Stderr, "round %d/%d\n", r+1, *rounds)
		for _, s := range sides {
			for _, run := range []struct{ bin, pattern string }{
				{"parquetio.test", "."}, {"arrowio.test", "."}, {"df.test", opBenchmarks},
			} {
				out := runCmd(env, "", filepath.Join(s.dir, run.bin), "-test.run", "^$", "-test.bench", run.pattern, "-test.benchtime", "3x")
				for name, ms := range goBenchmarks(out) {
					add(s.name, name, ms)
				}
			}
		}
		io := runCmd(env, filepath.Join(*root, "arrowio", "testdata"), *uv, "run", "-q", "--with", "pandas", "--with", "numpy", "--with", "pyarrow", "python", "bench_pandas.py")
		for i, f := range strings.Fields(io) {
			v, _ := strconv.ParseFloat(f, 64)
			add("pandas", fmt.Sprint("io", i), v)
		}
		pd := runCmd(env, *root, *uv, "run", "-q", "--with", "pandas", "--with", "numpy", "--with", "polars", "--with", "pyarrow", "python", "bench/bench_pandas.py", "1000000", "3")
		for _, line := range strings.Split(pd, "\n") {
			f := strings.Fields(line)
			if len(f) == 4 && f[0] == "pandas" && f[3] == "ms" {
				v, _ := strconv.ParseFloat(f[2], 64)
				add("pandas", f[1], v)
			}
		}
	}
	fmt.Fprintln(os.Stderr, "load after:", load())
	head := "| operation |"
	rule := "|---|"
	for _, s := range sides {
		head += " " + s.name + " |"
		rule += "---:|"
	}
	head += " pandas |"
	rule += "---:|"
	for _, s := range sides {
		head += " " + s.name + "/pandas |"
		rule += "---:|"
	}
	fmt.Println(head)
	fmt.Println(rule)
	for _, op := range ops {
		key := op.name
		if op.io >= 0 {
			key = fmt.Sprint("io", op.io)
		}
		py := median(samples["pandas"][key])
		row := "| " + op.name + " |"
		var meds []float64
		for _, s := range sides {
			m := median(samples[s.name][op.bench])
			meds = append(meds, m)
			row += fmt.Sprintf(" %.1f |", m)
		}
		row += fmt.Sprintf(" %.1f |", py)
		for _, m := range meds {
			row += fmt.Sprintf(" %.2f |", m/py)
		}
		fmt.Println(row)
	}
	for _, s := range sides {
		fmt.Printf("\n%s: memory of the operations table %.1f MB\n", s.name, median(samples[s.name]["memory"]))
	}
}

// goBenchmarks reads `go test -bench` output: milliseconds per operation
// by benchmark name (without the -N suffix), and the memory benchmark's
// MB as "memory".
func goBenchmarks(out string) map[string]float64 {
	res := map[string]float64{}
	sc := bufio.NewScanner(strings.NewReader(out))
	for sc.Scan() {
		f := strings.Fields(sc.Text())
		if len(f) < 4 || !strings.HasPrefix(f[0], "Benchmark") || f[3] != "ns/op" {
			continue
		}
		name, _, _ := strings.Cut(f[0], "-")
		ns, _ := strconv.ParseFloat(f[2], 64)
		res[name] = ns / 1e6
		if name == "BenchmarkMemory" && len(f) >= 6 {
			mb, _ := strconv.ParseFloat(f[4], 64)
			res["memory"] = mb
		}
	}
	return res
}

func median(v []float64) float64 {
	if len(v) == 0 {
		return 0
	}
	s := slices.Sorted(slices.Values(v))
	if len(s)%2 == 1 {
		return s[len(s)/2]
	}
	return (s[len(s)/2-1] + s[len(s)/2]) / 2
}

func runCmd(env []string, dir, name string, args ...string) string {
	cmd := exec.Command(name, args...)
	cmd.Env, cmd.Dir = env, dir
	var out, errb bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &errb
	if err := cmd.Run(); err != nil {
		fail("%s %s: %v\n%s", name, strings.Join(args, " "), err, errb.String())
	}
	return out.String()
}

func load() string {
	out, err := exec.Command("uptime").Output()
	if err != nil {
		return "?"
	}
	return strings.TrimSpace(string(out))
}

func fail(format string, a ...any) {
	fmt.Fprintf(os.Stderr, "harness: "+format+"\n", a...)
	os.Exit(1)
}
