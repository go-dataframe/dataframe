package dataframe

import (
	"fmt"
	"strconv"
	"strings"
	"unicode"
)

// Query returns the rows for which a boolean expression holds (pandas'
// query). The expression language is small and is parsed, never executed as
// code:
//
//   - column names, bare (price) or in backquotes (`unit price`); "index"
//     names the index when no column is called that;
//   - @name, a variable from vars;
//   - numbers, 'strings' or "strings", true/false (or True/False), lists
//     [1, 2, 3];
//   - arithmetic + - * / // % ** and unary minus;
//   - comparisons == != < <= > >=, chained as in 1 < x < 5;
//   - x in [...] and x not in [...];
//   - and, or, not (also &, |, ~), and parentheses;
//   - the functions isna(x), notna(x) and abs(x), and the methods
//     x.isna() and x.notna().
//
// Comparisons against a missing value are false (!= is true), as for NaN in
// pandas.
func (df *DataFrame) Query(expr string, vars ...map[string]any) (*DataFrame, error) {
	v, err := df.Eval(expr, vars...)
	if err != nil {
		return nil, err
	}
	if v.DType() != Bool {
		return nil, fmt.Errorf("%w: %q is not a condition (it gives %v)", ErrQuery, expr, v.DType())
	}
	return df.Filter(v)
}

// Eval evaluates an expression of the Query language to a series, one value
// per row (pandas' eval).
func (df *DataFrame) Eval(expr string, vars ...map[string]any) (*Series, error) {
	toks, err := tokenize(expr)
	if err != nil {
		return nil, err
	}
	p := &parser{toks: toks, df: df, vars: map[string]any{}}
	for _, m := range vars {
		for k, v := range m {
			p.vars[k] = v
		}
	}
	v, err := p.or()
	if err != nil {
		return nil, err
	}
	if p.pos < len(p.toks) {
		return nil, fmt.Errorf("%w: unexpected %q", ErrQuery, p.toks[p.pos].text)
	}
	return p.series(v)
}

type tokKind uint8

const (
	tkNum tokKind = iota
	tkStr
	tkName
	tkVar
	tkOp
)

type token struct {
	kind tokKind
	text string
}

// tokenize splits an expression into tokens.
func tokenize(s string) ([]token, error) {
	var out []token
	i := 0
	for i < len(s) {
		c := s[i]
		switch {
		case c == ' ' || c == '\t' || c == '\n' || c == '\r':
			i++
		case c >= '0' && c <= '9' || (c == '.' && i+1 < len(s) && s[i+1] >= '0' && s[i+1] <= '9'):
			j := i
			for j < len(s) && (isDigit(s[j]) || s[j] == '.' || s[j] == '_' ||
				((s[j] == 'e' || s[j] == 'E') && j+1 < len(s)) ||
				((s[j] == '+' || s[j] == '-') && (s[j-1] == 'e' || s[j-1] == 'E'))) {
				j++
			}
			out = append(out, token{tkNum, s[i:j]})
			i = j
		case c == '\'' || c == '"':
			var b strings.Builder
			j := i + 1
			for ; j < len(s) && s[j] != c; j++ {
				if s[j] == '\\' && j+1 < len(s) {
					j++
				}
				b.WriteByte(s[j])
			}
			if j >= len(s) {
				return nil, fmt.Errorf("%w: unterminated string", ErrQuery)
			}
			out = append(out, token{tkStr, b.String()})
			i = j + 1
		case c == '`':
			j := strings.IndexByte(s[i+1:], '`')
			if j < 0 {
				return nil, fmt.Errorf("%w: unterminated `name`", ErrQuery)
			}
			out = append(out, token{tkName, s[i+1 : i+1+j]})
			i += j + 2
		case c == '@' || c == '_' || unicode.IsLetter(rune(c)):
			j := i + 1
			for j < len(s) && (s[j] == '_' || isDigit(s[j]) || unicode.IsLetter(rune(s[j]))) {
				j++
			}
			if c == '@' {
				out = append(out, token{tkVar, s[i+1 : j]})
			} else {
				out = append(out, token{tkName, s[i:j]})
			}
			i = j
		default:
			op := ""
			for _, o := range []string{"**", "//", "==", "!=", "<=", ">=", "+", "-", "*", "/", "%", "<", ">", "(", ")", "[", "]", ",", "&", "|", "~", "."} {
				if strings.HasPrefix(s[i:], o) {
					op = o
					break
				}
			}
			if op == "" {
				return nil, fmt.Errorf("%w: unexpected character %q", ErrQuery, c)
			}
			out = append(out, token{tkOp, op})
			i += len(op)
		}
	}
	return out, nil
}

func isDigit(c byte) bool { return c >= '0' && c <= '9' }

// parser is a recursive-descent evaluator: it computes as it parses. A
// value is a *Series (one per row), a scalar Go value, or a []any list.
type parser struct {
	toks []token
	pos  int
	df   *DataFrame
	vars map[string]any
}

func (p *parser) peek(texts ...string) bool {
	if p.pos >= len(p.toks) {
		return false
	}
	t := p.toks[p.pos]
	for _, x := range texts {
		if (t.kind == tkOp || t.kind == tkName) && t.text == x {
			return true
		}
	}
	return false
}

func (p *parser) expect(text string) error {
	if !p.peek(text) {
		return fmt.Errorf("%w: expected %q", ErrQuery, text)
	}
	p.pos++
	return nil
}

// series turns a value into a series over the frame's rows.
func (p *parser) series(v any) (*Series, error) {
	if s, ok := v.(*Series); ok {
		return s, nil
	}
	c, err := fullCol(v, p.df.Len())
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrQuery, err)
	}
	return &Series{index: p.df.index, c: c}, nil
}

func (p *parser) or() (any, error) {
	return p.binary(p.and, []string{"or", "|"}, func(a, b *Series) (*Series, error) { return a.Or(b) })
}

func (p *parser) and() (any, error) {
	return p.binary(p.not, []string{"and", "&"}, func(a, b *Series) (*Series, error) { return a.And(b) })
}

func (p *parser) binary(next func() (any, error), ops []string, f func(a, b *Series) (*Series, error)) (any, error) {
	v, err := next()
	if err != nil {
		return nil, err
	}
	for p.peek(ops...) {
		p.pos++
		w, err := next()
		if err != nil {
			return nil, err
		}
		a, err := p.series(v)
		if err != nil {
			return nil, err
		}
		b, err := p.series(w)
		if err != nil {
			return nil, err
		}
		if v, err = f(a, b); err != nil {
			return nil, err
		}
	}
	return v, nil
}

func (p *parser) not() (any, error) {
	if p.peek("not", "~") {
		p.pos++
		v, err := p.not()
		if err != nil {
			return nil, err
		}
		s, err := p.series(v)
		if err != nil {
			return nil, err
		}
		return s.Not()
	}
	return p.comparison()
}

var cmpOps = map[string]cmpOp{"==": cmpEq, "!=": cmpNe, "<": cmpLt, "<=": cmpLe, ">": cmpGt, ">=": cmpGe}

func (p *parser) comparison() (any, error) {
	left, err := p.sum()
	if err != nil {
		return nil, err
	}
	var result *Series
	for {
		negate := false
		var op string
		switch {
		case p.peek("==", "!=", "<", "<=", ">", ">="):
			op = p.toks[p.pos].text
			p.pos++
		case p.peek("in"):
			op = "in"
			p.pos++
		case p.peek("not") && p.pos+1 < len(p.toks) && p.toks[p.pos+1].text == "in":
			op, negate = "in", true
			p.pos += 2
		default:
			if result == nil {
				return left, nil
			}
			return result, nil
		}
		right, err := p.sum()
		if err != nil {
			return nil, err
		}
		a, err := p.series(left)
		if err != nil {
			return nil, err
		}
		var r *Series
		if op == "in" {
			list, ok := right.([]any)
			if !ok {
				list = []any{right}
			}
			if r, err = a.IsIn(list...); err == nil && negate {
				r, err = r.Not()
			}
		} else {
			r, err = a.compareWith(cmpOps[op], right)
		}
		if err != nil {
			return nil, fmt.Errorf("%w: %v", ErrQuery, err)
		}
		if result == nil {
			result = r
		} else {
			result, _ = result.And(r)
		}
		left = right
	}
}

func (p *parser) sum() (any, error) {
	return p.arith(p.term, map[string]arithOp{"+": opAdd, "-": opSub})
}

func (p *parser) term() (any, error) {
	return p.arith(p.unary, map[string]arithOp{"*": opMul, "/": opDiv, "//": opFloorDiv, "%": opMod})
}

func (p *parser) arith(next func() (any, error), ops map[string]arithOp) (any, error) {
	v, err := next()
	if err != nil {
		return nil, err
	}
	for p.pos < len(p.toks) && p.toks[p.pos].kind == tkOp {
		op, ok := ops[p.toks[p.pos].text]
		if !ok {
			break
		}
		p.pos++
		w, err := next()
		if err != nil {
			return nil, err
		}
		if v, err = p.apply(op, v, w); err != nil {
			return nil, err
		}
	}
	return v, nil
}

// apply computes a binary arithmetic operation on two values.
func (p *parser) apply(op arithOp, v, w any) (any, error) {
	a, err := p.series(v)
	if err != nil {
		return nil, err
	}
	var r *Series
	if s, ok := w.(*Series); ok {
		r, err = a.arith(op, s)
	} else {
		r, err = a.arith(op, w)
	}
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrQuery, err)
	}
	return r, nil
}

func (p *parser) unary() (any, error) {
	if p.peek("-") {
		p.pos++
		v, err := p.unary()
		if err != nil {
			return nil, err
		}
		return p.apply(opMul, v, -1)
	}
	if p.peek("+") {
		p.pos++
		return p.unary()
	}
	return p.power()
}

func (p *parser) power() (any, error) {
	v, err := p.postfix()
	if err != nil {
		return nil, err
	}
	if p.peek("**") {
		p.pos++
		w, err := p.unary()
		if err != nil {
			return nil, err
		}
		return p.apply(opPow, v, w)
	}
	return v, nil
}

// postfix parses an atom followed by .isna() / .notna() calls.
func (p *parser) postfix() (any, error) {
	v, err := p.atom()
	if err != nil {
		return nil, err
	}
	for p.peek(".") {
		p.pos++
		if p.pos >= len(p.toks) || p.toks[p.pos].kind != tkName {
			return nil, fmt.Errorf("%w: expected a method name after '.'", ErrQuery)
		}
		name := p.toks[p.pos].text
		p.pos++
		if err := p.expect("("); err != nil {
			return nil, err
		}
		if err := p.expect(")"); err != nil {
			return nil, err
		}
		if v, err = p.call(name, v); err != nil {
			return nil, err
		}
	}
	return v, nil
}

// call applies a function of the language.
func (p *parser) call(name string, arg any) (any, error) {
	s, err := p.series(arg)
	if err != nil {
		return nil, err
	}
	switch name {
	case "isna", "isnull":
		return s.IsNA(), nil
	case "notna", "notnull":
		return s.NotNA(), nil
	case "abs":
		r, err := s.Abs()
		if err != nil {
			return nil, fmt.Errorf("%w: %v", ErrQuery, err)
		}
		return r, nil
	}
	return nil, fmt.Errorf("%w: unknown function %q", ErrQuery, name)
}

func (p *parser) atom() (any, error) {
	if p.pos >= len(p.toks) {
		return nil, fmt.Errorf("%w: unexpected end of expression", ErrQuery)
	}
	t := p.toks[p.pos]
	p.pos++
	switch t.kind {
	case tkNum:
		s := strings.ReplaceAll(t.text, "_", "")
		if i, err := strconv.ParseInt(s, 10, 64); err == nil {
			return i, nil
		}
		f, err := strconv.ParseFloat(s, 64)
		if err != nil {
			return nil, fmt.Errorf("%w: bad number %q", ErrQuery, t.text)
		}
		return f, nil
	case tkStr:
		return t.text, nil
	case tkVar:
		v, ok := p.vars[t.text]
		if !ok {
			return nil, fmt.Errorf("%w: undefined variable @%s", ErrQuery, t.text)
		}
		if s, ok := v.(*Series); ok {
			return s, nil
		}
		if l, ok := listOf(v); ok {
			return l, nil
		}
		return v, nil
	case tkName:
		return p.name(t)
	}
	switch t.text {
	case "(":
		v, err := p.or()
		if err != nil {
			return nil, err
		}
		return v, p.expect(")")
	case "[":
		var list []any
		for !p.peek("]") {
			v, err := p.sum()
			if err != nil {
				return nil, err
			}
			list = append(list, v)
			if !p.peek(",") {
				break
			}
			p.pos++
		}
		return list, p.expect("]")
	}
	return nil, fmt.Errorf("%w: unexpected %q", ErrQuery, t.text)
}

// name resolves an identifier: a function call, a keyword constant, a
// column, or the index.
func (p *parser) name(t token) (any, error) {
	if p.peek("(") {
		p.pos++
		v, err := p.or()
		if err != nil {
			return nil, err
		}
		if err := p.expect(")"); err != nil {
			return nil, err
		}
		return p.call(t.text, v)
	}
	switch t.text {
	case "true", "True":
		return true, nil
	case "false", "False":
		return false, nil
	}
	if s, err := p.df.Column(t.text); err == nil {
		return s, nil
	}
	if t.text == "index" || t.text == p.df.index.Name() && t.text != "" {
		lv, _ := p.df.index.Level(0)
		return &Series{name: lv.name, index: p.df.index, c: lv.c}, nil
	}
	return nil, fmt.Errorf("%w: unknown name %q", ErrQuery, t.text)
}

// listOf converts a Go slice variable to a []any list.
func listOf(v any) ([]any, bool) {
	switch l := v.(type) {
	case []any:
		return l, true
	case []string:
		out := make([]any, len(l))
		for i, x := range l {
			out[i] = x
		}
		return out, true
	case []int:
		out := make([]any, len(l))
		for i, x := range l {
			out[i] = x
		}
		return out, true
	case []float64:
		out := make([]any, len(l))
		for i, x := range l {
			out[i] = x
		}
		return out, true
	}
	return nil, false
}
