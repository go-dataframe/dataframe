package main

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// loadAvg returns the one- and five-minute load averages, from
// /proc/loadavg on Linux and `sysctl -n vm.loadavg` elsewhere.
func loadAvg() (one, five float64, err error) {
	var s string
	if b, err := os.ReadFile("/proc/loadavg"); err == nil {
		s = string(b)
	} else {
		out, err := exec.Command("sysctl", "-n", "vm.loadavg").Output()
		if err != nil {
			return 0, 0, err
		}
		s = strings.Trim(strings.TrimSpace(string(out)), "{}")
	}
	f := strings.Fields(s)
	if len(f) < 2 {
		return 0, 0, fmt.Errorf("load average %q", s)
	}
	one, err = strconv.ParseFloat(f[0], 64)
	if err == nil {
		five, err = strconv.ParseFloat(f[1], 64)
	}
	return one, five, err
}

// waitLoad blocks until the one-minute load average is under max and the
// five-minute one under 1.25*max, polling every 30 s, or gives up after
// timeout; it reports whether the host became quiet.
func waitLoad(max float64, timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	for {
		one, five, err := loadAvg()
		if err != nil {
			fail("%v", err)
		}
		if one < max && five < 1.25*max {
			return true
		}
		if time.Now().After(deadline) {
			fmt.Fprintf(os.Stderr, "load %.1f %.1f still above %.1f after %v\n", one, five, max, timeout)
			return false
		}
		fmt.Fprintf(os.Stderr, "%s load %.1f %.1f, waiting for < %.1f\n", time.Now().Format(time.TimeOnly), one, five, max)
		time.Sleep(30 * time.Second)
	}
}

// waitCmd is `harness waitload -max 8 -timeout 2h`.
func waitCmd(args []string) {
	fs := flag.NewFlagSet("waitload", flag.ExitOnError)
	max := fs.Float64("max", 8, "the load average to wait for")
	timeout := fs.Duration("timeout", 2*time.Hour, "how long to wait")
	_ = fs.Parse(args)
	if !waitLoad(*max, *timeout) {
		os.Exit(1)
	}
	fmt.Println("load", load())
}

// buildCmd is `harness build -o dir tree`: the three benchmark test
// binaries of the go-dataframe tree, as measure runs them.
func buildCmd(args []string) {
	fs := flag.NewFlagSet("build", flag.ExitOnError)
	out := fs.String("o", "", "the directory to write df.test, parquetio.test and arrowio.test into")
	_ = fs.Parse(args)
	if *out == "" || fs.NArg() != 1 {
		fail("usage: harness build -o dir tree")
	}
	if err := os.MkdirAll(*out, 0o755); err != nil {
		fail("%v", err)
	}
	env := append(os.Environ(), "GOWORK=off")
	for _, p := range []struct{ bin, pkg string }{{"df.test", "."}, {"parquetio.test", "./parquetio"}, {"arrowio.test", "./arrowio"}} {
		abs, _ := filepath.Abs(filepath.Join(*out, p.bin))
		runCmd(env, fs.Arg(0), "go", "test", "-c", "-o", abs, p.pkg)
	}
	fmt.Println("built", *out)
}
