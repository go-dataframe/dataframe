package dataframe

import (
	"fmt"
	"math"
	"strings"
	"time"
)

// timeOf returns nanoseconds since the epoch as a UTC time.Time.
func timeOf(ns int64) time.Time { return time.Unix(0, ns).UTC() }

// daysFromCivil returns the days since 1970-01-01 of a proleptic Gregorian
// date (Howard Hinnant's algorithm), without time.Date's normalisation cost.
func daysFromCivil(y, m, d int) int64 {
	if m <= 2 {
		y--
	}
	era := y
	if era < 0 {
		era -= 399
	}
	era /= 400
	yoe := y - era*400
	mp := (m + 9) % 12
	doy := (153*mp+2)/5 + d - 1
	doe := yoe*365 + yoe/4 - yoe/100 + doy
	return int64(era)*146097 + int64(doe) - 719468
}

// civilFromDays is the inverse of daysFromCivil.
func civilFromDays(z int64) (y, m, d int) {
	z += 719468
	era := z
	if era < 0 {
		era -= 146096
	}
	era /= 146097
	doe := z - era*146097
	yoe := (doe - doe/1460 + doe/36524 - doe/146096) / 365
	y = int(yoe + era*400)
	doy := doe - (365*yoe + yoe/4 - yoe/100)
	mp := (5*doy + 2) / 153
	d = int(doy - (153*mp+2)/5 + 1)
	m = int(mp + 3)
	if m > 12 {
		m -= 12
	}
	if m <= 2 {
		y++
	}
	return y, m, d
}

var daysIn = [13]int{0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}

func isLeap(y int) bool { return y%4 == 0 && (y%100 != 0 || y%400 == 0) }

// fallbackLayouts are tried, in order, for strings the ISO reader does not
// take. Month-first comes before day-first, as pandas assumes by default.
var fallbackLayouts = []string{
	"2006/01/02", "2006/01/02 15:04:05", "2006/01/02 15:04",
	"01/02/2006", "01/02/2006 15:04:05", "01/02/2006 15:04", "1/2/2006",
	"02.01.2006", "20060102", "Jan 2, 2006", "2 Jan 2006", "January 2, 2006",
	"02-Jan-2006", "Mon, 02 Jan 2006 15:04:05 MST", time.RFC3339Nano,
}

// parseTime parses an instant to nanoseconds since the epoch, UTC. With an
// empty layout it reads ISO 8601 (date, optional time with fractional
// seconds, optional Z or offset, which is applied) and a list of common
// layouts; otherwise layout is a Go time layout, or a strftime pattern such
// as "%d/%m/%Y" (anything with a %).
func parseTime(s, layout string) (int64, error) {
	return newTimeParser(layout).parse(s)
}

// timeParser is parseTime for one layout, translated (from strftime) and
// planned once, to parse many strings.
type timeParser struct {
	layout string     // a Go layout, or "" for ISO 8601 and the fallbacks
	plan   []layoutOp // the fixed-width numeric reading of layout, or nil
}

// layoutOp is one step of a fixed-width plan: a field of width digits
// (field > 0), or a literal byte.
type layoutOp struct {
	field   layoutField
	width   int
	literal byte
}

type layoutField uint8

const (
	literal layoutField = iota
	fieldYear
	fieldMonth
	fieldDay
	fieldHour
	fieldMinute
	fieldSecond
)

// layoutTokens are the Go layout elements a plan reads, longest first.
var layoutTokens = []struct {
	tok   string
	field layoutField
}{
	{"2006", fieldYear}, {"01", fieldMonth}, {"02", fieldDay},
	{"15", fieldHour}, {"04", fieldMinute}, {"05", fieldSecond},
}

func newTimeParser(layout string) *timeParser {
	if strings.Contains(layout, "%") {
		layout = strftimeLayout(layout)
	}
	return &timeParser{layout: layout, plan: planLayout(layout)}
}

// planLayout returns the plan of a layout made only of the year, month,
// day, hour, minute and second elements and the separators - / : space T,
// each element at most once, year month and day all there; nil otherwise.
// A string that fits the plan exactly (each field all digits, at its fixed
// width) is one time.Parse reads the same way; any other is left to it.
func planLayout(layout string) []layoutOp {
	var plan []layoutOp
	var seen [fieldSecond + 1]bool
next:
	for i := 0; i < len(layout); {
		for _, t := range layoutTokens {
			if strings.HasPrefix(layout[i:], t.tok) {
				if seen[t.field] {
					return nil
				}
				seen[t.field] = true
				plan = append(plan, layoutOp{field: t.field, width: len(t.tok)})
				i += len(t.tok)
				continue next
			}
		}
		switch c := layout[i]; c {
		case '-', '/', ':', ' ', 'T':
			plan = append(plan, layoutOp{literal: c})
			i++
		default:
			return nil
		}
	}
	if !seen[fieldYear] || !seen[fieldMonth] || !seen[fieldDay] {
		return nil
	}
	return plan
}

// fast parses s by the plan, or reports false: a string of another shape,
// a field out of range, or a year outside 1678-2261 (the nanosecond range,
// with room to spare), which time.Parse then handles, errors included.
func (p *timeParser) fast(s string) (int64, bool) {
	var v [fieldSecond + 1]int
	i := 0
	for _, op := range p.plan {
		if op.field == literal {
			if i >= len(s) || s[i] != op.literal {
				return 0, false
			}
			i++
			continue
		}
		n, ok := num(s, i, op.width)
		if !ok {
			return 0, false
		}
		v[op.field] = n
		i += op.width
	}
	y, mo, d := v[fieldYear], v[fieldMonth], v[fieldDay]
	switch {
	case i != len(s), y < 1678, y > 2261, mo < 1, mo > 12, d < 1, d > daysIn[mo],
		mo == 2 && d == 29 && !isLeap(y), v[fieldHour] > 23, v[fieldMinute] > 59, v[fieldSecond] > 59:
		return 0, false
	}
	return daysFromCivil(y, mo, d)*86400e9 + int64(v[fieldHour])*3600e9 +
		int64(v[fieldMinute])*60e9 + int64(v[fieldSecond])*1e9, true
}

func (p *timeParser) parse(s string) (int64, error) {
	layout := p.layout
	if layout != "" {
		if p.plan != nil {
			if ns, ok := p.fast(s); ok {
				return ns, nil
			}
		}
		t, err := time.Parse(layout, s)
		if err != nil {
			return 0, fmt.Errorf("%w: %q does not match %q", ErrParse, s, layout)
		}
		return nanosOf(t, s)
	}
	if ns, ok := parseISO(s); ok {
		return ns, nil
	}
	for _, l := range fallbackLayouts {
		if t, err := time.Parse(l, s); err == nil {
			return nanosOf(t, s)
		}
	}
	return 0, fmt.Errorf("%w: %q is not a date", ErrParse, s)
}

// Instants are int64 nanoseconds: about 1677-09-21 to 2262-04-11.
var (
	minTime = time.Unix(0, math.MinInt64)
	maxTime = time.Unix(0, math.MaxInt64)
)

// nanosOf returns a time as nanoseconds since the epoch, or an error when it
// is out of their range (pandas' OutOfBoundsDatetime).
func nanosOf(t time.Time, s string) (int64, error) {
	if t.Before(minTime) || t.After(maxTime) {
		return 0, fmt.Errorf("%w: %q is out of the nanosecond range", ErrParse, s)
	}
	return t.UnixNano(), nil
}

// num reads n decimal digits of s at i.
func num(s string, i, n int) (int, bool) {
	if i+n > len(s) {
		return 0, false
	}
	v := 0
	for _, c := range []byte(s[i : i+n]) {
		if c < '0' || c > '9' {
			return 0, false
		}
		v = v*10 + int(c-'0')
	}
	return v, true
}

// parseISO reads YYYY, YYYY-MM, YYYY-MM-DD, then optionally [T ]HH:MM,
// :SS, .fraction and Z or ±HH[:]MM.
func parseISO(s string) (int64, bool) {
	y, ok := num(s, 0, 4)
	if !ok {
		return 0, false
	}
	mo, d := 1, 1
	i := 4
	if i < len(s) && s[i] == '-' {
		if mo, ok = num(s, 5, 2); !ok {
			return 0, false
		}
		i = 7
		if i < len(s) && s[i] == '-' {
			if d, ok = num(s, 8, 2); !ok {
				return 0, false
			}
			i = 10
		}
	}
	if mo < 1 || mo > 12 || d < 1 || d > daysIn[mo] || (mo == 2 && d == 29 && !isLeap(y)) {
		return 0, false
	}
	days := daysFromCivil(y, mo, d)
	if days < -106751 || days > 106750 {
		return 0, false
	}
	ns := days * 86400e9
	if i == len(s) {
		return ns, true
	}
	if i != 10 || (s[i] != 'T' && s[i] != ' ') {
		return 0, false
	}
	h, ok1 := num(s, 11, 2)
	mi, ok2 := num(s, 14, 2)
	if !ok1 || !ok2 || s[13] != ':' || h > 23 || mi > 59 {
		return 0, false
	}
	ns += int64(h)*3600e9 + int64(mi)*60e9
	i = 16
	if i < len(s) && s[i] == ':' {
		sec, ok := num(s, 17, 2)
		if !ok || sec > 59 {
			return 0, false
		}
		ns += int64(sec) * 1e9
		i = 19
		if i < len(s) && (s[i] == '.' || s[i] == ',') {
			i++
			scale := int64(1e8)
			start := i
			for i < len(s) && s[i] >= '0' && s[i] <= '9' {
				ns += int64(s[i]-'0') * scale
				scale /= 10
				i++
			}
			if i == start {
				return 0, false
			}
		}
	}
	return applyZone(s[i:], ns)
}

// applyZone applies a trailing Z, ±HH, ±HHMM or ±HH:MM (or nothing).
func applyZone(z string, ns int64) (int64, bool) {
	switch {
	case z == "" || z == "Z":
		return ns, true
	case z[0] != '+' && z[0] != '-':
		return 0, false
	}
	sign := int64(1)
	if z[0] == '-' {
		sign = -1
	}
	z = strings.ReplaceAll(z[1:], ":", "")
	h, ok := num(z, 0, 2)
	m := 0
	if ok && len(z) == 4 {
		m, ok = num(z, 2, 2)
	} else if len(z) != 2 {
		ok = false
	}
	if !ok {
		return 0, false
	}
	return ns - sign*(int64(h)*3600e9+int64(m)*60e9), true
}

// strftimeLayout translates the common strftime directives to a Go layout.
func strftimeLayout(f string) string { return strftimeReplacer.Replace(f) }

var strftimeReplacer = strings.NewReplacer(
	"%Y", "2006", "%y", "06", "%m", "01", "%d", "02", "%H", "15",
	"%I", "03", "%M", "04", "%S", "05", "%f", "000000", "%p", "PM",
	"%b", "Jan", "%B", "January", "%a", "Mon", "%A", "Monday",
	"%z", "-0700", "%Z", "MST", "%j", "002", "%%", "%",
)

// ToDatetime converts a series to datetime. A string series is parsed (ISO
// 8601 and common layouts, or the given layout: a Go layout or a strftime
// pattern such as "%d/%m/%Y"); a string that does not parse is an error. An
// integer series is read as nanoseconds since the epoch; a datetime series
// is returned as is.
func ToDatetime(s *Series, layout ...string) (*Series, error) {
	l := ""
	if len(layout) > 0 {
		l = layout[0]
	}
	switch c := s.c.(type) {
	case stringCol:
		out := makeCol[int64](Datetime, c.len())
		out.valid = c.validity().clone()
		p := newTimeParser(l)
		for i := range c.len() {
			if c.isNA(i) {
				continue
			}
			ns, err := p.parse(strings.TrimSpace(c.at(i)))
			if err != nil {
				return nil, err
			}
			out.data[i] = ns
		}
		return s.with(out), nil
	}
	if !s.DType().isReal() && s.DType() != Datetime {
		return nil, fmt.Errorf("%w: cannot convert %v to datetime", ErrDType, s.DType())
	}
	c, _ := cast(s.c, Int64)
	c, _ = cast(c, Datetime)
	return s.with(c), nil
}

// Timestamp parses one instant as ToDatetime parses a string.
func Timestamp(s string, layout ...string) (time.Time, error) {
	l := ""
	if len(layout) > 0 {
		l = layout[0]
	}
	ns, err := parseTime(s, l)
	return timeOf(ns), err
}
