package dataframe

import (
	"fmt"
	"math"
	"math/bits"
	"strconv"
	"sync"
)

// Converting fields to columns. Each column is converted chunk by chunk in
// parallel. Inference takes one pass per chunk that parses integers (or,
// once a float shows up, floats) straight into the column's final storage,
// born inside an ndarray, and records which kinds of value the chunk holds;
// the kinds of all the chunks decide the column's dtype, as pandas decides
// it from the whole column, and only chunks whose pass did not already
// produce that dtype are converted again.

// convMode says how a column is converted.
type convMode uint8

const (
	convInfer convMode = iota // infer the dtype
	convDate                  // parse_dates: datetime, else string
	convDType                 // an explicit dtype
)

// colSpec is one column to convert: field j of each record.
type colSpec struct {
	j    int
	name string
	mode convMode
	dt   DType
}

// The classes of a field's value.
const (
	clsInt   = iota // an integer that fits int64
	clsUint         // an integer above int64 that fits uint64
	clsBig          // an integer beyond 64 bits
	clsFloat        // any other number
	clsOther        // not a number
)

// Flags of the values a chunk of a column holds.
const (
	fNA uint16 = 1 << iota
	fInt
	fNeg
	fUint
	fBig
	fFloat
	fBool
	fOther
	fFail    // a date that does not parse, or a value the dtype rejects
	fNegZero // "-0", which is -0.0 if the column turns out float
)

// What a chunk's first pass left in the column's storage.
const (
	modeInt   = iota // int64 values
	modeFloat        // float64 values
	modeNone         // nothing usable
)

type converter struct {
	t      *tokenizer
	cfg    *csvConfig
	groups []*chunkOut
	offs   []int // the first row of each chunk
	n      int

	na        map[string]bool
	naFirst   [256]bool
	naMaxLen  int
	naNumeric bool // a missing-value token reads as a number
	bools     map[string]bool
}

func newConverter(t *tokenizer, cfg *csvConfig, groups []*chunkOut) *converter {
	cv := &converter{t: t, cfg: cfg, groups: groups, na: make(map[string]bool), bools: make(map[string]bool)}
	for _, g := range groups {
		cv.offs = append(cv.offs, cv.n)
		cv.n += g.nrec
	}
	var tokens []string
	if cfg.keepDefaultNA {
		tokens = defaultNA
	}
	for _, s := range append(tokens, cfg.naValues...) {
		cv.na[s] = true
		cv.naMaxLen = max(cv.naMaxLen, len(s))
		if s != "" {
			cv.naFirst[s[0]] = true
		}
		if c, _, _, _ := parseNumber(s); c != clsOther {
			cv.naNumeric = true
		}
	}
	for _, s := range append([]string{"True", "TRUE", "true"}, cfg.trueValues...) {
		cv.bools[s] = true
	}
	for _, s := range append([]string{"False", "FALSE", "false"}, cfg.falseValues...) {
		cv.bools[s] = false
	}
	return cv
}

// isNA reports a missing-value token.
func (cv *converter) isNA(s string) bool {
	if s == "" {
		return cv.na[""]
	}
	if len(s) > cv.naMaxLen || !cv.naFirst[s[0]] {
		return false
	}
	return cv.na[s]
}

// colState is the conversion of one column.
type colState struct {
	sp      colSpec
	cv      *converter
	intOnce sync.Once
	ints    *col[int64]
	fltOnce sync.Once
	flts    *col[float64]
	flags   []uint16 // per chunk
	mode    []uint8  // per chunk
	valid   []bitmap // per chunk, rows relative to it; nil when none missing
	errs    []error  // per chunk
	kind    DType
	strs    []string // headers into the input until finish lays them out
	bools   *col[bool]
	uints   *col[uint64]
}

func (st *colState) intBuf() []int64 {
	st.intOnce.Do(func() {
		dt := Int64
		if st.sp.mode == convDate || (st.sp.mode == convDType && st.sp.dt == Datetime) {
			dt = Datetime
		}
		st.ints = makeCol[int64](dt, st.cv.n)
	})
	return st.ints.data
}

func (st *colState) floatBuf() []float64 {
	st.fltOnce.Do(func() { st.flts = makeCol[float64](Float64, st.cv.n) })
	return st.flts.data
}

// missing marks row r of chunk g missing.
func (st *colState) missing(g, r int) {
	if st.valid[g] == nil {
		st.valid[g] = allValid(st.cv.groups[g].nrec)
	}
	st.valid[g].clear(r)
	st.flags[g] |= fNA
}

// run converts the columns.
func (cv *converter) run(specs []colSpec) ([]column, error) {
	ng := len(cv.groups)
	sts := make([]*colState, len(specs))
	for k, sp := range specs {
		sts[k] = &colState{sp: sp, cv: cv, flags: make([]uint16, ng), mode: make([]uint8, ng),
			valid: make([]bitmap, ng), errs: make([]error, ng)}
		if sp.mode == convDType {
			sts[k].kind = sp.dt
			if !directDType(sp.dt) {
				sts[k].kind = String
			}
			sts[k].alloc()
		}
	}
	parallelEach(len(specs)*ng, func(i int) { sts[i/ng].first(i % ng) })
	for _, st := range sts {
		if err := st.decide(); err != nil {
			return nil, err
		}
	}
	parallelEach(len(specs)*ng, func(i int) { sts[i/ng].second(i % ng) })
	out := make([]column, len(specs))
	for k, st := range sts {
		c, err := st.finish()
		if err != nil {
			return nil, err
		}
		out[k] = c
	}
	return out, nil
}

// first is the first pass over chunk g.
func (st *colState) first(g int) {
	switch st.sp.mode {
	case convInfer:
		st.inferChunk(g)
	case convDate:
		st.dateChunk(g, false)
	default:
		st.dtypeChunk(g)
	}
}

func (st *colState) inferChunk(g int) {
	cv, o, off := st.cv, st.cv.groups[g], st.cv.offs[g]
	j := st.sp.j
	var ints []int64
	var flts []float64
	mode := uint8(modeInt)
	flags := uint16(0)
	defer func() { st.mode[g] = mode; st.flags[g] |= flags }()
	for r := range o.nrec {
		s, ok := cv.t.field(o, r, j)
		if !ok || (cv.naNumeric && cv.isNA(s)) {
			st.missing(g, r)
			continue
		}
		cls, iv, fv, neg := parseNumber(s)
		switch cls {
		case clsInt:
			flags |= fInt
			if neg {
				flags |= fNeg
				if iv == 0 {
					flags |= fNegZero
				}
			}
			switch mode {
			case modeInt:
				if ints == nil {
					ints = st.intBuf()
				}
				ints[off+r] = iv
			case modeFloat:
				flts[off+r] = fv
			}
			continue
		case clsUint:
			flags |= fUint
		case clsBig:
			flags |= fBig
		case clsFloat:
			flags |= fFloat
		default:
			if cv.isNA(s) {
				st.missing(g, r)
				continue
			}
			if _, ok := cv.bools[s]; ok {
				flags |= fBool
				mode = modeNone
				continue
			}
			flags |= fOther
			mode = modeNone
			return
		}
		// A uint, big integer or float: floats from now on, if ints so far.
		if mode == modeInt && cls != clsUint {
			flts = st.floatBuf()
			st.intsToFloats(g, r, flags, ints)
			mode = modeFloat
		}
		if mode == modeFloat {
			flts[off+r] = fv
		} else {
			mode = modeNone
		}
	}
}

// intsToFloats turns the first rows of chunk g, read as integers, into
// floats. A float64 rounds an int64 as parsing its digits as a float does,
// except for the sign of "-0", which takes reading the text again. ints is
// nil when no row holds an integer.
func (st *colState) intsToFloats(g, rows int, flags uint16, ints []int64) {
	off := st.cv.offs[g]
	switch {
	case flags&fNegZero != 0:
		st.reparseFloats(g, rows)
	case ints != nil:
		flts := st.flts.data
		for r := range rows {
			flts[off+r] = float64(ints[off+r])
		}
	}
}

// reparseFloats reads the first rows of chunk g as floats.
func (st *colState) reparseFloats(g, rows int) {
	cv, o, off := st.cv, st.cv.groups[g], st.cv.offs[g]
	flts := st.flts.data
	for r := range rows {
		if s, ok := cv.t.field(o, r, st.sp.j); ok {
			_, _, flts[off+r], _ = parseNumber(s)
		}
	}
}

// dateChunk parses chunk g as instants; strict makes a bad value an error.
func (st *colState) dateChunk(g int, strict bool) {
	cv, o, off := st.cv, st.cv.groups[g], st.cv.offs[g]
	p := newTimeParser(cv.cfg.dateFormat)
	for r := range o.nrec {
		s, ok := cv.t.field(o, r, st.sp.j)
		if !ok || cv.isNA(s) {
			st.missing(g, r)
			continue
		}
		ns, err := p.parse(s)
		if err != nil {
			st.flags[g] |= fFail
			if strict {
				st.errs[g] = st.badValue(s, Datetime)
			}
			return
		}
		st.intBuf()[off+r] = ns
	}
}

func (st *colState) badValue(s string, dt DType) error {
	return fmt.Errorf("%w: column %q: cannot read %q as %v", ErrParse, st.sp.name, s, dt)
}

// dtypeChunk converts chunk g to an explicit dtype: int64, float64, bool and
// datetime directly; the others become strings, converted at the end.
func (st *colState) dtypeChunk(g int) {
	switch st.sp.dt {
	case Int64, Float64:
		st.numberChunk(g)
	case Bool:
		st.boolChunk(g, true)
	case Datetime:
		st.dateChunk(g, true)
	default:
		st.stringChunk(g)
	}
}

// numberChunk parses chunk g as int64 or float64, a bad value an error.
func (st *colState) numberChunk(g int) {
	cv, o, off := st.cv, st.cv.groups[g], st.cv.offs[g]
	isInt := st.sp.dt == Int64
	if !isInt {
		st.mode[g] = modeFloat
	}
	for r := range o.nrec {
		s, ok := cv.t.field(o, r, st.sp.j)
		if !ok || cv.isNA(s) {
			st.missing(g, r)
			continue
		}
		cls, iv, fv, _ := parseNumber(s)
		switch {
		case isInt && cls == clsInt:
			st.intBuf()[off+r] = iv
		case !isInt && cls != clsOther:
			st.floatBuf()[off+r] = fv
		default:
			st.errs[g] = st.badValue(s, st.sp.dt)
			return
		}
	}
}

// boolChunk reads chunk g as bools; with ones, "1" and "0" are bools too.
func (st *colState) boolChunk(g int, ones bool) {
	cv, o, off := st.cv, st.cv.groups[g], st.cv.offs[g]
	for r := range o.nrec {
		s, ok := cv.t.field(o, r, st.sp.j)
		if !ok || cv.isNA(s) {
			st.missing(g, r)
			continue
		}
		b, ok := cv.bools[s]
		if !ok && ones && (s == "1" || s == "0") {
			b, ok = s == "1", true
		}
		if !ok {
			st.errs[g] = st.badValue(s, Bool)
			return
		}
		st.bools.data[off+r] = b
	}
}

func (st *colState) stringChunk(g int) {
	cv, o, off := st.cv, st.cv.groups[g], st.cv.offs[g]
	for r := range o.nrec {
		s, ok := cv.t.field(o, r, st.sp.j)
		if !ok || cv.isNA(s) {
			st.missing(g, r)
			continue
		}
		st.strs[off+r] = s
	}
}

func (st *colState) uintChunk(g int) {
	cv, o, off := st.cv, st.cv.groups[g], st.cv.offs[g]
	for r := range o.nrec {
		if s, ok := cv.t.field(o, r, st.sp.j); ok && (st.valid[g] == nil || st.valid[g].get(r)) {
			st.uints.data[off+r] = parseUint(s)
		}
	}
}

func (st *colState) floatChunk(g int) {
	o, off := st.cv.groups[g], st.cv.offs[g]
	flts := st.floatBuf()
	switch st.mode[g] {
	case modeInt:
		var ints []int64
		if st.ints != nil {
			ints = st.ints.data
		}
		st.intsToFloats(g, o.nrec, st.flags[g], ints)
	case modeNone:
		st.reparseFloats(g, o.nrec)
	}
	if v := st.valid[g]; v != nil {
		for r := range o.nrec {
			if !v.get(r) {
				flts[off+r] = math.NaN()
			}
		}
	}
}

// decide picks the column's dtype from what the first pass found, and
// allocates what the second pass fills.
func (st *colState) decide() error {
	var flags uint16
	for g, f := range st.flags {
		if st.errs[g] != nil {
			return st.errs[g]
		}
		flags |= f
	}
	n := st.cv.n
	switch st.sp.mode {
	case convDate:
		st.kind = Datetime
		if flags&fFail != 0 {
			st.kind = String
		}
	case convDType:
		// run set the dtype.
	default:
		num := flags&(fInt|fUint|fBig|fFloat) != 0
		switch {
		case flags&fOther != 0 || (flags&fBool != 0 && num):
			st.kind = String
		case flags&fBool != 0:
			st.kind = Bool
		case flags&fFloat != 0:
			st.kind = Float64
		case flags&fBig != 0:
			st.kind = String
		case flags&fUint != 0 && flags&fNeg != 0:
			st.kind = String
		case flags&fUint != 0:
			st.kind = Uint64
		case flags&fInt != 0:
			st.kind = Int64
		case n > 0:
			st.kind = Float64 // every value missing
		default:
			st.kind = String
		}
	}
	st.alloc()
	switch st.kind {
	case Uint64:
		st.uints = makeCol[uint64](Uint64, n)
	case Float64:
		st.floatBuf()
	case Int64, Datetime:
		st.intBuf()
	}
	return nil
}

// directDType reports the explicit dtypes the passes convert to directly.
func directDType(dt DType) bool {
	return dt == Int64 || dt == Float64 || dt == Bool || dt == Datetime
}

// alloc allocates the string or bool storage the column's dtype needs.
func (st *colState) alloc() {
	switch {
	case st.strs == nil && st.kind == String:
		st.strs = make([]string, st.cv.n)
	case st.bools == nil && st.kind == Bool:
		st.bools = makeCol[bool](Bool, st.cv.n)
	}
}

// second is the second pass over chunk g, for the chunks the first pass
// did not leave in the column's dtype.
func (st *colState) second(g int) {
	switch {
	case st.kind == Float64:
		st.floatChunk(g)
	case st.sp.mode == convDType:
	case st.kind == String:
		st.stringChunk(g)
	case st.kind == Bool:
		st.boolChunk(g, false)
	case st.kind == Uint64:
		st.uintChunk(g)
	}
}

// finish assembles the column.
func (st *colState) finish() (column, error) {
	n := st.cv.n
	var valid bitmap
	if st.kind != Float64 {
		for g, v := range st.valid {
			if v == nil {
				continue
			}
			if valid == nil {
				valid = allValid(n)
			}
			off := st.cv.offs[g]
			for r := range st.cv.groups[g].nrec {
				if !v.get(r) {
					valid.clear(off + r)
				}
			}
		}
	}
	var c column
	switch st.kind {
	case String:
		c = newStrCol(st.strs, valid)
		st.strs = nil
		if st.sp.mode == convDType && st.sp.dt != String {
			return st.cast(c)
		}
	case Bool:
		st.bools.valid = valid
		c = st.bools
	case Uint64:
		st.uints.valid = valid
		c = st.uints
	case Float64:
		c = st.flts
	default:
		st.ints.valid = valid
		c = st.ints
	}
	return c, nil
}

// cast converts a string column to an explicit dtype the passes do not
// handle directly.
func (st *colState) cast(c column) (column, error) {
	out, err := cast(c, st.sp.dt)
	if err != nil {
		return nil, fmt.Errorf("column %q: %w", st.sp.name, err)
	}
	return out, nil
}

// isSpace reports the bytes pandas skips around a number.
func isSpace(c byte) bool {
	return c == ' ' || c == '\t' || c == '\v' || c == '\f' || c == '\r' || c == '\n'
}

// pow10 are the powers of ten a float64 holds exactly.
var pow10 = [...]float64{1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10,
	1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22}

// parseNumber classifies a field as pandas' parser reads numbers: an integer
// or a decimal float with an optional exponent, spaces around it allowed,
// or inf / infinity in any case with an optional sign and no spaces. It
// returns the class, the value as an int64 (clsInt) and as a float64 (every
// number class), and whether a sign was negative.
func parseNumber(s string) (cls int, iv int64, fv float64, neg bool) {
	i, j := 0, len(s)
	for i < j && isSpace(s[i]) {
		i++
	}
	for j > i && isSpace(s[j-1]) {
		j--
	}
	if i == j {
		return clsOther, 0, 0, false
	}
	p := i
	if s[p] == '+' || s[p] == '-' {
		neg = s[p] == '-'
		p++
	}
	var m uint64
	nd, over := 0, false
	for ; p < j && s[p] >= '0' && s[p] <= '9'; p++ {
		m, over = addDigit(m, s[p], nd, over)
		nd++
	}
	if p == j && nd > 0 {
		switch {
		case over || (neg && m > 1<<63):
			f, _ := strconv.ParseFloat(s[i:j], 64)
			return clsBig, 0, f, neg
		case neg:
			return clsInt, int64(-m), -float64(m), true
		case m > math.MaxInt64:
			return clsUint, 0, float64(m), false
		}
		return clsInt, int64(m), float64(m), false
	}
	frac := 0
	if p < j && s[p] == '.' {
		for p++; p < j && s[p] >= '0' && s[p] <= '9'; p++ {
			m, over = addDigit(m, s[p], nd+frac, over)
			frac++
		}
	}
	if nd+frac == 0 {
		if i == 0 && j == len(s) && isInf(s[p:]) {
			sign := 1
			if neg {
				sign = -1
			}
			return clsFloat, 0, math.Inf(sign), neg
		}
		return clsOther, 0, 0, false
	}
	exp := 0
	if p < j && (s[p] == 'e' || s[p] == 'E') {
		p++
		eneg := false
		if p < j && (s[p] == '+' || s[p] == '-') {
			eneg = s[p] == '-'
			p++
		}
		start := p
		for ; p < j && s[p] >= '0' && s[p] <= '9'; p++ {
			if exp < 10000 {
				exp = exp*10 + int(s[p]-'0')
			}
		}
		if p == start {
			return clsOther, 0, 0, false
		}
		if eneg {
			exp = -exp
		}
	}
	if p != j {
		return clsOther, 0, 0, false
	}
	// Clinger's fast path: an exact mantissa scaled by an exact power of ten
	// rounds once, correctly.
	if e := exp - frac; !over && m < 1<<53 && e >= -22 && e <= 22 {
		f := float64(m)
		if e < 0 {
			f /= pow10[-e]
		} else {
			f *= pow10[e]
		}
		if neg {
			f = -f
		}
		return clsFloat, 0, f, neg
	}
	f, _ := strconv.ParseFloat(s[i:j], 64)
	return clsFloat, 0, f, neg
}

// addDigit appends decimal digit c to m, which holds nd digits; over
// reports that m no longer holds the number. Nineteen digits always fit in
// a uint64, so only the twentieth and later can overflow it.
func addDigit(m uint64, c byte, nd int, over bool) (uint64, bool) {
	d := uint64(c - '0')
	if nd >= 19 {
		hi, lo := bits.Mul64(m, 10)
		over = over || hi != 0 || lo+d < lo
	}
	return m*10 + d, over
}

// isInf reports "inf" or "infinity" in any case.
func isInf(s string) bool {
	const word = "infinity"
	if len(s) != 3 && len(s) != 8 {
		return false
	}
	for k := range len(s) {
		if s[k]|0x20 != word[k] {
			return false
		}
	}
	return true
}

// parseUint reads a field parseNumber classed as an integer that fits
// uint64.
func parseUint(s string) uint64 {
	var m uint64
	for k := range len(s) {
		if c := s[k]; c >= '0' && c <= '9' {
			m = m*10 + uint64(c-'0')
		}
	}
	return m
}
