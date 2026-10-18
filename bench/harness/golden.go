package main

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"

	"github.com/apache/arrow-go/v18/parquet/file"
	"github.com/go-dataframe/dataframe/parquetio"
)

// golden compares the files two builds wrote with TestGoldenFiles. A file
// with the same SHA-256 in both is identical. A Parquet file that differs
// must still hold the same data, read back, and the same statistics in
// every column chunk; any other difference is a failure. It reports each
// file and whether all held.
func golden(a, b string) bool {
	sa, sb := sums(a), sums(b)
	if !slices.Equal(slices.Sorted(mapKeys(sa)), slices.Sorted(mapKeys(sb))) {
		fmt.Println("the two directories hold different files")
		return false
	}
	ok, same, equiv := true, 0, 0
	for _, f := range slices.Sorted(mapKeys(sa)) {
		if sa[f] == sb[f] {
			same++
			continue
		}
		if !strings.Contains(f, ".pq") {
			fmt.Printf("%s: bytes differ\n", f)
			ok = false
			continue
		}
		if err := sameParquet(filepath.Join(a, f), filepath.Join(b, f)); err != nil {
			fmt.Printf("%s: %v\n", f, err)
			ok = false
			continue
		}
		equiv++
		fmt.Printf("%s: bytes differ; same data and statistics\n", f)
	}
	fmt.Printf("%d files: %d byte-identical, %d Parquet files equivalent, all held: %v\n", len(sa), same, equiv, ok)
	return ok
}

func sums(dir string) map[string]string {
	f, err := os.Open(filepath.Join(dir, "SHA256SUMS"))
	if err != nil {
		fail("%v", err)
	}
	defer f.Close()
	m := map[string]string{}
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		if fs := strings.Fields(sc.Text()); len(fs) == 3 {
			m[fs[2]] = fs[0]
		}
	}
	return m
}

func mapKeys(m map[string]string) func(func(string) bool) {
	return func(yield func(string) bool) {
		for k := range m {
			if !yield(k) {
				return
			}
		}
	}
}

// sameParquet reports how two Parquet files differ in what they hold: the
// frames read back (columns, dtypes, every value, validity, index), the
// row groups, and each column chunk's statistics.
func sameParquet(a, b string) error {
	fa, err := parquetio.ReadParquetFile(a)
	if err != nil {
		return err
	}
	fb, err := parquetio.ReadParquetFile(b)
	if err != nil {
		return err
	}
	if !slices.Equal(fa.Columns(), fb.Columns()) || !slices.Equal(fa.DTypes(), fb.DTypes()) {
		return fmt.Errorf("columns or dtypes differ")
	}
	for i, s := range fa.Series() {
		t := fb.Series()[i]
		if !reflect.DeepEqual(s.Items(), t.Items()) && fmt.Sprint(s.Items()) != fmt.Sprint(t.Items()) {
			return fmt.Errorf("column %q differs", s.Name())
		}
		if !slices.Equal(s.ValidityBitmap(), t.ValidityBitmap()) {
			return fmt.Errorf("column %q: validity differs", s.Name())
		}
	}
	if fmt.Sprint(fa.Index().Labels()) != fmt.Sprint(fb.Index().Labels()) {
		return fmt.Errorf("index differs")
	}
	ra, err := file.OpenParquetFile(a, false)
	if err != nil {
		return err
	}
	defer ra.Close()
	rb, err := file.OpenParquetFile(b, false)
	if err != nil {
		return err
	}
	defer rb.Close()
	ma, mb := ra.MetaData(), rb.MetaData()
	if ma.NumRowGroups() != mb.NumRowGroups() || ma.NumRows != mb.NumRows {
		return fmt.Errorf("row groups differ")
	}
	for g := range ma.NumRowGroups() {
		for c := range ma.Schema.NumColumns() {
			ca, _ := ma.RowGroup(g).ColumnChunk(c)
			cb, _ := mb.RowGroup(g).ColumnChunk(c)
			if ca.NumValues() != cb.NumValues() {
				return fmt.Errorf("row group %d column %d: value counts differ", g, c)
			}
			sa, ea := ca.Statistics()
			sb, eb := cb.Statistics()
			if (ea == nil) != (eb == nil) || (sa == nil) != (sb == nil) {
				return fmt.Errorf("row group %d column %d: statistics present in one only", g, c)
			}
			if sa == nil {
				continue
			}
			if sa.NullCount() != sb.NullCount() || sa.NumValues() != sb.NumValues() || sa.HasMinMax() != sb.HasMinMax() ||
				string(sa.EncodeMin()) != string(sb.EncodeMin()) || string(sa.EncodeMax()) != string(sb.EncodeMax()) {
				return fmt.Errorf("row group %d column %d: statistics differ", g, c)
			}
		}
	}
	return nil
}
