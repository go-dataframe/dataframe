package main

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
)

// gate runs, on the module at dir, every step of .github/workflows/ci.yml:
// gofmt, vet, build, the race-enabled tests with coverage over every
// package held to 100% (and no block left at zero), vet for the qemu
// architectures, the test binaries of every target, govulncheck. Tests run
// with -p 4. It stops at the first failing step and reports false.
func gate(dir string) bool {
	tmp, err := os.MkdirTemp("", "dataframe-gate-")
	if err != nil {
		fail("%v", err)
	}
	defer os.RemoveAll(tmp)
	cover := filepath.Join(tmp, "cover.out")
	env := append(os.Environ(), "GOWORK=off")
	step := func(name string, extra []string, args ...string) (string, bool) {
		fmt.Println("==", name)
		cmd := exec.Command(args[0], args[1:]...)
		cmd.Dir, cmd.Env = dir, append(env, extra...)
		out, err := cmd.CombinedOutput()
		if err != nil {
			fmt.Printf("%s\nFAILED: %v\n", out, err)
			return string(out), false
		}
		return string(out), true
	}
	if out, ok := step("gofmt -l", nil, "gofmt", "-l", "."); !ok || strings.TrimSpace(out) != "" {
		fmt.Print(out)
		return false
	}
	for _, s := range [][]string{{"go", "vet", "./..."}, {"go", "build", "./..."}} {
		if _, ok := step(strings.Join(s, " "), nil, s...); !ok {
			return false
		}
	}
	pkgs, ok := step("go list", nil, "go", "list", "./...")
	if !ok {
		return false
	}
	list := strings.Fields(pkgs)
	if _, ok := step("go test -p 4 -race -coverpkg", nil, "go", "test", "-p", "4", "-race", "-coverpkg="+strings.Join(list, ","), "-coverprofile="+cover, "./..."); !ok {
		return false
	}
	total, zero := coverage(cover)
	fmt.Printf("coverage: %.1f%%\n", total)
	if len(zero) > 0 {
		fmt.Println("uncovered:\n" + strings.Join(zero, "\n"))
		return false
	}
	for _, arch := range []string{"riscv64", "loong64", "ppc64le", "s390x"} {
		if _, ok := step("go vet "+arch, []string{"GOOS=linux", "GOARCH=" + arch}, "go", "vet", "./..."); !ok {
			return false
		}
	}
	fmt.Println("== go test -c every target")
	for _, t := range []string{"linux/amd64", "linux/arm64", "linux/riscv64", "linux/loong64", "linux/ppc64le", "linux/s390x",
		"linux/386", "linux/arm", "darwin/amd64", "darwin/arm64", "windows/amd64", "windows/arm64", "freebsd/amd64"} {
		goos, goarch, _ := strings.Cut(t, "/")
		for _, p := range list {
			cmd := exec.Command("go", "test", "-c", "-o", os.DevNull, p)
			cmd.Dir, cmd.Env = dir, append(env, "CGO_ENABLED=0", "GOOS="+goos, "GOARCH="+goarch)
			if out, err := cmd.CombinedOutput(); err != nil {
				fmt.Printf("%s %s: %v\n%s", t, p, err, out)
				return false
			}
		}
	}
	vuln := "govulncheck"
	if _, err := exec.LookPath(vuln); err != nil {
		home, _ := os.UserHomeDir()
		vuln = filepath.Join(home, "go", "bin", "govulncheck")
	}
	if _, ok := step("govulncheck", nil, vuln, "./..."); !ok {
		return false
	}
	fmt.Println("GATE OK")
	return true
}

// coverage reads a coverage profile written by several package runs: the
// statement coverage over every block, and the blocks no run executed.
func coverage(path string) (float64, []string) {
	f, err := os.Open(path)
	if err != nil {
		fail("%v", err)
	}
	defer f.Close()
	count := map[string]int{}
	stmts := map[string]int{}
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := sc.Text()
		if strings.HasPrefix(line, "mode:") {
			continue
		}
		var block string
		var n, c int
		if _, err := fmt.Sscanf(line, "%s %d %d", &block, &n, &c); err != nil {
			continue
		}
		count[block] += c
		stmts[block] = n
	}
	all, hit := 0, 0
	var zero []string
	for b, n := range stmts {
		all += n
		if count[b] > 0 {
			hit += n
		} else {
			zero = append(zero, b)
		}
	}
	sort.Strings(zero)
	if all == 0 {
		return 0, zero
	}
	return 100 * float64(hit) / float64(all), zero
}
