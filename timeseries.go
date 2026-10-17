package dataframe

import (
	"cmp"
	"fmt"
	"strconv"
	"strings"
	"time"
)

const (
	nsSecond = int64(time.Second)
	nsMinute = int64(time.Minute)
	nsHour   = int64(time.Hour)
	nsDay    = 24 * nsHour
)

// floorDiv64 divides rounding toward minus infinity.
func floorDiv64(a, b int64) int64 {
	q := a / b
	if a%b != 0 && (a < 0) != (b < 0) {
		q--
	}
	return q
}

// Period frequencies: the calendar unit a period spans.
var periodFreqs = map[string]string{
	"Y": "Y", "A": "Y", "YE": "Y", "Q": "Q", "QE": "Q", "M": "M", "ME": "M",
	"W": "W", "D": "D", "h": "h", "H": "h", "min": "min", "T": "min", "s": "s", "S": "s",
}

// periodOrdinal returns the ordinal of the period of frequency freq holding
// the instant ns: periods counted from the one holding the Unix epoch.
func periodOrdinal(ns int64, freq string) int64 {
	days := floorDiv64(ns, nsDay)
	y, m, _ := civilFromDays(days)
	switch freq {
	case "Y":
		return int64(y - 1970)
	case "Q":
		return int64(y-1970)*4 + int64(m-1)/3
	case "M":
		return int64(y-1970)*12 + int64(m-1)
	case "W":
		// Weeks run Monday to Sunday; 1970-01-01 was a Thursday.
		return floorDiv64(days+3, 7)
	case "D":
		return days
	case "h":
		return floorDiv64(ns, nsHour)
	case "min":
		return floorDiv64(ns, nsMinute)
	}
	return floorDiv64(ns, nsSecond)
}

// periodStart returns the first instant of a period.
func periodStart(ord int64, freq string) int64 {
	switch freq {
	case "Y":
		return daysFromCivil(int(ord)+1970, 1, 1) * nsDay
	case "Q":
		y, q := floorDiv64(ord, 4), ord-4*floorDiv64(ord, 4)
		return daysFromCivil(int(y)+1970, int(q)*3+1, 1) * nsDay
	case "M":
		y, m := floorDiv64(ord, 12), ord-12*floorDiv64(ord, 12)
		return daysFromCivil(int(y)+1970, int(m)+1, 1) * nsDay
	case "W":
		return (ord*7 - 3) * nsDay
	case "D":
		return ord * nsDay
	case "h":
		return ord * nsHour
	case "min":
		return ord * nsMinute
	}
	return ord * nsSecond
}

// formatPeriod formats a period as pandas prints one.
func formatPeriod(ord int64, freq string) string {
	t := timeOf(periodStart(ord, freq))
	switch freq {
	case "Y":
		return t.Format("2006")
	case "Q":
		return t.Format("2006") + "Q" + strconv.Itoa(int(t.Month()+2)/3)
	case "M":
		return t.Format("2006-01")
	case "W":
		return t.Format("2006-01-02") + "/" + t.AddDate(0, 0, 6).Format("2006-01-02")
	case "D":
		return t.Format("2006-01-02")
	case "h", "min":
		return t.Format("2006-01-02 15:04")
	}
	return t.Format("2006-01-02 15:04:05")
}

// periodOps returns the ops of a Period column of a frequency.
func periodOps(freq string) *ops[int64] {
	return &ops[int64]{
		cmp: cmp.Compare[int64],
		str: func(o int64) string { return formatPeriod(o, freq) },
		box: func(o int64) any { return formatPeriod(o, freq) },
		unbox: func(v any) (int64, bool) {
			ns, ok := toNanos(v)
			return periodOrdinal(ns, freq), ok
		},
	}
}

// normFreq returns the canonical period frequency of an alias.
func normFreq(freq string) (string, error) {
	f, ok := periodFreqs[freq]
	if !ok {
		return "", fmt.Errorf("%w: unknown period frequency %q", ErrArgument, freq)
	}
	return f, nil
}

// periodCol converts a datetime column to periods.
func periodCol(c column, freq string) (column, error) {
	p, ok := c.(*col[int64])
	if !ok || p.dt != Datetime {
		return nil, fmt.Errorf("%w: periods come from datetimes, not %v", ErrDType, c.dtype())
	}
	f, err := normFreq(freq)
	if err != nil {
		return nil, err
	}
	out := makeCol[int64](Period, len(p.data))
	out.freq = f
	out.o = periodOps(f)
	out.valid = p.valid
	for i, ns := range p.data {
		out.data[i] = periodOrdinal(ns, f)
	}
	return out, nil
}

// ToPeriod converts a datetime series to periods of a frequency: "Y", "Q",
// "M", "W", "D", "h", "min" or "s" (pandas' dt.to_period).
func (s *Series) ToPeriod(freq string) (*Series, error) {
	c, err := periodCol(s.c, freq)
	if err != nil {
		return nil, err
	}
	return s.with(c), nil
}

// ToPeriod converts the frame's datetime index to periods (pandas'
// to_period).
func (df *DataFrame) ToPeriod(freq string) (*DataFrame, error) {
	if df.index.NLevels() != 1 {
		return nil, fmt.Errorf("%w: ToPeriod needs a one-level index", ErrArgument)
	}
	c, err := periodCol(df.index.level(0), freq)
	if err != nil {
		return nil, err
	}
	return df.WithIndex(labelIndex(df.index.names, []column{c}))
}

// ToTimestamp converts a period series to the instants its periods start.
func (s *Series) ToTimestamp() (*Series, error) {
	p, ok := s.c.(*col[int64])
	if !ok || p.dt != Period {
		return nil, fmt.Errorf("%w: ToTimestamp needs periods, not %v", ErrDType, s.DType())
	}
	out := makeCol[int64](Datetime, len(p.data))
	out.valid = p.valid
	for i, o := range p.data {
		out.data[i] = periodStart(o, p.freq)
	}
	return s.with(out), nil
}

// DatetimeFields reads parts of a datetime series, as pandas' .dt accessor.
type DatetimeFields struct{ s *Series }

// Dt returns the datetime accessor; it fails on a series of another dtype.
func (s *Series) Dt() (DatetimeFields, error) {
	if s.DType() != Datetime {
		return DatetimeFields{}, fmt.Errorf("%w: .Dt needs datetimes, not %v", ErrDType, s.DType())
	}
	return DatetimeFields{s}, nil
}

func (d DatetimeFields) field(f func(t time.Time) int64) *Series {
	p := d.s.c.(*col[int64])
	out := makeCol[int64](Int64, len(p.data))
	out.valid = p.valid
	for i, ns := range p.data {
		out.data[i] = f(timeOf(ns))
	}
	return d.s.with(out)
}

// Year returns the year of each instant.
func (d DatetimeFields) Year() *Series {
	return d.field(func(t time.Time) int64 { return int64(t.Year()) })
}

// Month returns the month, 1 to 12.
func (d DatetimeFields) Month() *Series {
	return d.field(func(t time.Time) int64 { return int64(t.Month()) })
}

// Day returns the day of the month.
func (d DatetimeFields) Day() *Series {
	return d.field(func(t time.Time) int64 { return int64(t.Day()) })
}

// Hour returns the hour.
func (d DatetimeFields) Hour() *Series {
	return d.field(func(t time.Time) int64 { return int64(t.Hour()) })
}

// Minute returns the minute.
func (d DatetimeFields) Minute() *Series {
	return d.field(func(t time.Time) int64 { return int64(t.Minute()) })
}

// Second returns the second.
func (d DatetimeFields) Second() *Series {
	return d.field(func(t time.Time) int64 { return int64(t.Second()) })
}

// DayOfWeek returns the day of the week, Monday 0 to Sunday 6, as pandas.
func (d DatetimeFields) DayOfWeek() *Series {
	return d.field(func(t time.Time) int64 { return int64(t.Weekday()+6) % 7 })
}

// DayOfYear returns the day of the year, from 1.
func (d DatetimeFields) DayOfYear() *Series {
	return d.field(func(t time.Time) int64 { return int64(t.YearDay()) })
}

// Normalize returns each instant at midnight of its day.
func (d DatetimeFields) Normalize() *Series {
	p := d.s.c.(*col[int64])
	out := makeCol[int64](Datetime, len(p.data))
	out.valid = p.valid
	for i, ns := range p.data {
		out.data[i] = floorDiv64(ns, nsDay) * nsDay
	}
	return d.s.with(out)
}

// Strftime formats each instant with a Go layout or a strftime pattern.
func (d DatetimeFields) Strftime(layout string) *Series {
	if strings.Contains(layout, "%") {
		layout = strftimeLayout(layout)
	}
	p := d.s.c.(*col[int64])
	out := newStrBuilder(len(p.data), 0)
	for _, ns := range p.data {
		out.buf = timeOf(ns).AppendFormat(out.buf, layout)
		out.offs = append(out.offs, int64(len(out.buf)))
	}
	return d.s.with(out.column(p.valid))
}

// resampleRule is a parsed resampling frequency.
type resampleRule struct {
	width  int64  // a fixed bin width in nanoseconds, or 0
	period string // a calendar period ("W", "M", "Q", "Y"), or ""
	end    bool   // label calendar bins by their last day
}

// parseRule parses a resampling frequency: an optional count and a unit,
// one of s, min (or T), h (or H), D for fixed bins; W (weeks ending
// Sunday), ME (or M), QE (or Q), YE (or Y, A) labelled by their last day;
// MS, QS, YS (or AS) labelled by their first day.
func parseRule(rule string) (resampleRule, error) {
	i := 0
	for i < len(rule) && isDigit(rule[i]) {
		i++
	}
	n := int64(1)
	if i > 0 {
		n, _ = strconv.ParseInt(rule[:i], 10, 64)
	}
	unit := rule[i:]
	fixed := map[string]int64{"s": nsSecond, "S": nsSecond, "min": nsMinute, "T": nsMinute,
		"h": nsHour, "H": nsHour, "D": nsDay}
	if w, ok := fixed[unit]; ok && n > 0 {
		return resampleRule{width: n * w}, nil
	}
	cal := map[string]resampleRule{
		"W": {period: "W", end: true}, "W-SUN": {period: "W", end: true},
		"ME": {period: "M", end: true}, "M": {period: "M", end: true},
		"QE": {period: "Q", end: true}, "Q": {period: "Q", end: true},
		"YE": {period: "Y", end: true}, "Y": {period: "Y", end: true}, "A": {period: "Y", end: true},
		"MS": {period: "M"}, "QS": {period: "Q"}, "YS": {period: "Y"}, "AS": {period: "Y"},
	}
	if r, ok := cal[unit]; ok && n == 1 {
		return r, nil
	}
	return resampleRule{}, fmt.Errorf("%w: unsupported resampling rule %q", ErrArgument, rule)
}

// Resampler is a frame's rows binned by time, ready to aggregate.
type Resampler struct {
	g *GroupBy
}

// Resample bins the rows of a frame with a datetime index by time (pandas'
// resample): fixed bins (e.g. "15min", "h", "D", "7D") start at midnight of
// the first day and are labelled by their start; calendar bins ("W", "ME",
// "QE", "YE", "MS", "QS", "YS") hold whole weeks, months, quarters or
// years. Every bin from the first to the last is in the result, empty ones
// included (their sum is 0, their mean missing).
func (df *DataFrame) Resample(rule string) (*Resampler, error) {
	if df.index.NLevels() != 1 || df.index.DType() != Datetime {
		return nil, fmt.Errorf("%w: Resample needs a datetime index", ErrArgument)
	}
	return df.resample(df.index.level(0).(*col[int64]), df.index.Name(), rule)
}

// ResampleOn bins by a datetime column instead of the index.
func (df *DataFrame) ResampleOn(column, rule string) (*Resampler, error) {
	i, err := df.pos(column)
	if err != nil {
		return nil, err
	}
	p, ok := df.cols[i].(*col[int64])
	if !ok || p.dt != Datetime {
		return nil, fmt.Errorf("%w: column %q is not datetime", ErrDType, column)
	}
	rest, _ := df.Drop(column)
	return rest.resample(p, column, rule)
}

func (df *DataFrame) resample(t *col[int64], name, rule string) (*Resampler, error) {
	r, err := parseRule(rule)
	if err != nil {
		return nil, err
	}
	n := len(t.data)
	var origin int64
	found := false
	for i, ns := range t.data {
		if !t.isNA(i) {
			o := floorDiv64(ns, nsDay) * nsDay
			if !found || o < origin {
				origin = o
			}
			found = true
		}
	}
	bins := make([]int64, n)
	lo, hi := int64(0), int64(-1)
	for i, ns := range t.data {
		if t.isNA(i) {
			continue
		}
		b := periodOrdinal(ns, r.period)
		if r.width > 0 {
			b = floorDiv64(ns-origin, r.width)
		}
		if hi < lo || b < lo {
			lo = b
		}
		if b > hi {
			hi = b
		}
		bins[i] = b
	}
	nb := int(hi - lo + 1)
	codes := make([]int, n)
	for i := range codes {
		codes[i] = -1
		if !t.isNA(i) {
			codes[i] = int(bins[i] - lo)
		}
	}
	labels := makeCol[int64](Datetime, nb)
	for k := range nb {
		b := lo + int64(k)
		switch {
		case r.width > 0:
			labels.data[k] = origin + b*r.width
		case r.end:
			labels.data[k] = periodStart(b+1, r.period) - nsDay
		default:
			labels.data[k] = periodStart(b, r.period)
		}
	}
	g := &GroupBy{df: df, codes: codes, ngroups: nb, values: df.names,
		bins: labelIndex([]string{name}, []column{labels})}
	return &Resampler{g: g}, nil
}

// Sum returns the sum of each numeric column per bin.
func (r *Resampler) Sum() (*DataFrame, error) { return r.g.aggregate("sum") }

// Mean returns the mean per bin.
func (r *Resampler) Mean() (*DataFrame, error) { return r.g.aggregate("mean") }

// Min returns the minimum per bin.
func (r *Resampler) Min() (*DataFrame, error) { return r.g.aggregate("min") }

// Max returns the maximum per bin.
func (r *Resampler) Max() (*DataFrame, error) { return r.g.aggregate("max") }

// Count returns the number of present values per bin.
func (r *Resampler) Count() (*DataFrame, error) { return r.g.aggregate("count") }

// First returns the first present value per bin.
func (r *Resampler) First() (*DataFrame, error) { return r.g.aggregate("first") }

// Last returns the last present value per bin.
func (r *Resampler) Last() (*DataFrame, error) { return r.g.aggregate("last") }

// Std returns the sample standard deviation per bin.
func (r *Resampler) Std() (*DataFrame, error) { return r.g.aggregate("std") }

// Median returns the median per bin.
func (r *Resampler) Median() (*DataFrame, error) { return r.g.aggregate("median") }

// Agg applies several aggregations, as GroupBy.Agg.
func (r *Resampler) Agg(specs ...AggSpec) (*DataFrame, error) { return r.g.Agg(specs...) }

// Size returns the number of rows per bin.
func (r *Resampler) Size() *Series { return r.g.Size() }
