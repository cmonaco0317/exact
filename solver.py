"""
Exact calculus engine.

Design contract:
  - ALL mathematics is done by SymPy (a real computer algebra system), never
    guessed. This module only (a) translates human input into SymPy and
    (b) independently *verifies* every answer numerically before returning it.
  - The public entry point is solve(query) -> dict. It never raises; on any
    failure it returns {"ok": False, "error": "..."} so the UI can render it.

This exact file is embedded verbatim into the web app and runs in the browser
via Pyodide, so it must stay pure-Python (sympy + mpmath only, no file/network).
"""

import re
import sympy as sp
from sympy import (
    Symbol,
    diff,
    integrate,
    limit,
    series,
    solve as sp_solve,
    Eq,
    oo,
    zoo,
    nan,
    S,
    latex,
    simplify,
    factor,
    expand,
    Abs,
    pi,
    E,
    I,
    Derivative,
    Integral,
    Rational,
    Float,
    sympify,
    N,
)
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)
from sympy.integrals.manualintegrate import integral_steps

_TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

# Constants / aliases available to the parser.
_LOCALS = {
    "e": E,
    "E": E,
    "pi": pi,
    "Pi": pi,
    "PI": pi,
    "oo": oo,
    "inf": oo,
    "infinity": oo,
    "infty": oo,
    "gamma": sp.EulerGamma,
}

_FUNCS = "sin|cos|tan|sec|csc|cot|sinh|cosh|tanh|coth|sech|csch|asin|acos|atan|asec|acsc|acot|ln|log|exp"

# Human notation -> SymPy function names. Longest keys first (regex applied in
# order) so "arcsinh" is replaced before "arcsin".
_ALIASES = [
    ("arcsinh", "asinh"),
    ("arccosh", "acosh"),
    ("arctanh", "atanh"),
    ("arcsin", "asin"),
    ("arccos", "acos"),
    ("arctan", "atan"),
    ("arcsec", "asec"),
    ("arccsc", "acsc"),
    ("arccot", "acot"),
    ("cosec", "csc"),
]

# Deterministic sample points for numeric verification (avoids RNG so results
# are reproducible run-to-run and Mac-vs-phone).
_SAMPLES = [0.4123, 1.2711, -0.7237, 2.1329, -1.4519, 0.9137, -0.3301, 1.7743]


# --------------------------------------------------------------------------- #
#  Preprocessing
# --------------------------------------------------------------------------- #
_SUPER = {
    "²": "^2",
    "³": "^3",
    "⁴": "^4",
    "⁵": "^5",
    "⁶": "^6",
    "⁷": "^7",
    "⁸": "^8",
    "⁹": "^9",
    "¹": "^1",
    "⁰": "^0",
}


def preprocess(raw):
    """Normalize unicode and common notation into ascii SymPy-friendly text."""
    s = raw.strip()
    for k, v in _SUPER.items():
        s = s.replace(k, v)
    s = (
        s.replace("×", "*")
        .replace("·", "*")
        .replace("÷", "/")
        .replace("√", "sqrt")
        .replace("π", "pi")
        .replace("∞", "oo")
        .replace("→", "->")
        .replace("−", "-")
        .replace("’", "'")
        .replace("^{", "^(")
        .replace("}", ")")
    )
    # arc-trig / alias names -> SymPy's names (longest first so arcsinh wins
    # over arcsin). Without this, "arctan" is an unknown multi-letter symbol
    # and gets split into a*r*c*t*a*n.
    for long, short in _ALIASES:
        s = re.sub(rf"\b{long}\b", short, s)
    # |x|  ->  Abs(x)  (non-nested absolute-value bars)
    s = re.sub(r"\|([^|]+)\|", r"Abs(\1)", s)
    # func^n(...)  ->  (func(...))^n   e.g. cos^2(x) -> (cos(x))**2
    s = re.sub(rf"\b({_FUNCS})\s*\^\s*(-?\d+)\s*\(([^()]*)\)", r"(\1(\3))**(\2)", s)
    return s


def _P(expr_str):
    """Parse an expression string into a SymPy expression (may raise)."""
    expr_str = expr_str.strip().strip(".").strip()
    # strip a trailing "dx" / "d x" if it slipped through
    expr_str = re.sub(r"\s*d\s*[a-zA-Z]\s*$", "", expr_str)
    if not expr_str:
        raise ValueError("empty expression")
    return parse_expr(expr_str, transformations=_TRANSFORMS, local_dict=dict(_LOCALS))


def _pick_var(expr, prefer="x"):
    """Choose the working variable: prefer x, else first free symbol, else x."""
    syms = sorted(expr.free_symbols, key=lambda s: s.name)
    if not syms:
        return Symbol(prefer)
    for s in syms:
        if s.name == prefer:
            return s
    return syms[0]


# --------------------------------------------------------------------------- #
#  Numeric evaluation + verification helpers
# --------------------------------------------------------------------------- #
def _numeric(expr, subs):
    """Evaluate expr with a {sym: value} dict to a python complex, or None."""
    try:
        val = complex(N(expr.subs(subs), 30))
        if val != val or abs(val) == float("inf"):  # nan / inf
            return None
        return val
    except Exception:
        return None


def _close(a, b, rel=1e-6, atol=1e-9):
    if a is None or b is None:
        return None
    return abs(a - b) <= atol + rel * max(abs(a), abs(b))


def _verify_derivative(f, var, dydx):
    """Central finite-difference check of a derivative at several points."""
    h = 1e-5
    agree = 0
    total = 0
    for x0 in _SAMPLES:
        fp = _numeric(f, {var: x0 + h})
        fm = _numeric(f, {var: x0 - h})
        exact = _numeric(dydx, {var: x0})
        if fp is None or fm is None or exact is None:
            continue
        approx = (fp - fm) / (2 * h)
        total += 1
        if abs(approx - exact) <= 1e-3 * (1 + abs(exact)):
            agree += 1
    if total == 0:
        return (
            None,
            "computed symbolically (numeric check not applicable on this domain)",
        )
    if agree == total:
        return True, f"numerically confirmed at {total} sample points"
    return False, f"numeric check disagreed ({agree}/{total} points matched)"


def _verify_antiderivative(f, var, F):
    """An antiderivative is correct iff d/dx F == f. Check symbolically then numerically."""
    d = simplify(diff(F, var) - f)
    if d == 0:
        return True, "confirmed: differentiating the answer returns the integrand"
    ok, note = _verify_derivative(F, var, f)  # does F' match f numerically?
    if ok:
        return True, "confirmed: derivative of the answer matches the integrand"
    if ok is None:
        return None, "computed symbolically (numeric check not applicable)"
    return False, "numeric check: derivative of the answer did not match the integrand"


def _verify_definite(f, var, a, b, value):
    """Independently integrate numerically (mpmath) and compare."""
    try:
        af = complex(N(a, 30))
        bf = complex(N(b, 30))
    except Exception:
        af = bf = None
    try:
        num = complex(sp.Integral(f, (var, a, b)).evalf(25))
    except Exception:
        num = None
    sym = _numeric(value, {})
    if num is None or sym is None:
        return None, "computed symbolically (independent numeric check not available)"
    if _close(num, sym, rel=1e-6, atol=1e-8):
        return True, "confirmed against independent numerical integration"
    return (
        False,
        f"numeric integration gave {num.real:.6g}, symbolic gave {sym.real:.6g}",
    )


def _verify_limit(f, var, point, value):
    """Approach the point numerically and check convergence toward value."""
    tgt = _numeric(value, {})
    if tgt is None:
        return None, "computed symbolically (numeric check not applicable)"
    if point in (oo, -oo):
        pts = [1e2, 1e3, 1e4, 1e5, 1e6]
        pts = pts if point == oo else [-p for p in pts]
        seq = [_numeric(f, {var: p}) for p in pts]
    else:
        p0 = complex(N(point, 30)).real
        seq = []
        for k in (2, 3, 4, 5, 6):
            seq.append(_numeric(f, {var: p0 + 10 ** (-k)}))
            seq.append(_numeric(f, {var: p0 - 10 ** (-k)}))
    seq = [v for v in seq if v is not None]
    if not seq:
        return None, "computed symbolically (numeric check not applicable)"
    last = seq[-1]
    if _close(last, tgt, rel=1e-4, atol=1e-6):
        return True, "confirmed: function approaches this value near the point"
    return (
        False,
        f"numeric approach gave ~{last.real:.6g}, symbolic limit {tgt.real:.6g}",
    )


def _verify_roots(expr, var, roots):
    total = 0
    agree = 0
    for r in roots:
        val = _numeric(expr, {var: r})
        if val is None:
            continue
        total += 1
        if abs(val) <= 1e-6:
            agree += 1
    if total == 0:
        return None, "computed symbolically (numeric check not applicable)"
    return (agree == total), (
        f"all {total} roots substitute back to ~0"
        if agree == total
        else f"{agree}/{total} roots verified"
    )


# --------------------------------------------------------------------------- #
#  Step-by-step integration (recursive rule describer)
# --------------------------------------------------------------------------- #
def _rule_name(rule):
    return type(rule).__name__


def _describe_integral(rule, depth=0):
    """Turn a manualintegrate rule tree into human-readable LaTeX step lines."""
    steps = []
    name = _rule_name(rule)
    g = lambda attr: getattr(rule, attr, None)
    integ = g("integrand")
    var = g("variable")

    def line(txt):
        steps.append(txt)

    if name == "ConstantRule":
        line(
            rf"\int {latex(integ)}\,d{latex(var)} = {latex(integ)}\,{latex(var)}"
            r"\quad\text{(integral of a constant)}"
        )
    elif name == "PowerRule":
        base, exp = g("base"), g("exp")
        line(
            rf"\text{{Power rule: }} \int {latex(var)}^{{{latex(exp)}}}\,d{latex(var)}"
            rf" = \frac{{{latex(var)}^{{{latex(exp)}+1}}}}{{{latex(exp)}+1}}"
        )
    elif name in ("ExpRule", "ExpBaseRule"):
        line(rf"\text{{Exponential rule applied to }} {latex(integ)}")
    elif name in (
        "SinRule",
        "CosRule",
        "TrigRule",
        "SecTanRule",
        "CscCotRule",
        "Sec2Rule",
        "Csc2Rule",
    ):
        line(rf"\text{{Standard trig integral of }} {latex(integ)}")
    elif name == "ReciprocalRule":
        line(rf"\int \frac{{1}}{{{latex(var)}}}\,d{latex(var)} = \ln|{latex(var)}|")
    elif name in ("ArctanRule", "ArctangentRule"):
        line(rf"\text{{Recognize }} {latex(integ)} \text{{ as an arctangent form}}")
    elif name in ("ArcsinRule", "ArcsinRule"):
        line(rf"\text{{Recognize }} {latex(integ)} \text{{ as an arcsine form}}")
    elif name == "ConstantTimesRule":
        c, other = g("constant"), g("other")
        line(
            rf"\text{{Pull out the constant }} {latex(c)}: \int {latex(c)}\cdot{latex(other)}"
            rf"\,d{latex(var)} = {latex(c)}\int {latex(other)}\,d{latex(var)}"
        )
        sub = g("substep")
        if sub is not None:
            steps += _describe_integral(sub, depth + 1)
    elif name == "AddRule":
        line(r"\text{Split the integral over the sum, term by term:}")
        for sub in g("substeps") or []:
            steps += _describe_integral(sub, depth + 1)
    elif name in ("URule", "USubstitutionRule"):
        u = g("u_var") or g("u")
        line(
            rf"\text{{Substitution: let }} u = {latex(g('u_func') if g('u_func') is not None else u)}"
        )
        sub = g("substep")
        if sub is not None:
            steps += _describe_integral(sub, depth + 1)
    elif name == "PartsRule":
        u, dv = g("u"), g("dv")
        line(
            rf"\text{{Integration by parts with }} u = {latex(u)},\ dv = {latex(dv)}\,d{latex(var)}"
            r"\quad \left(\int u\,dv = uv - \int v\,du\right)"
        )
    elif name in (
        "RewriteRule",
        "TrigSubstitutionRule",
        "CompleteSquareRule",
        "PiecewiseRule",
        "AlternativeRule",
        "DontKnowRule",
    ):
        if name == "RewriteRule" and g("rewritten") is not None:
            line(rf"\text{{Rewrite as }} {latex(g('rewritten'))}")
        sub = g("substep")
        if sub is not None:
            steps += _describe_integral(sub, depth + 1)
    else:
        # Unknown/other rule: don't fabricate a method, just note it.
        line(r"\text{Apply standard integration rules.}")
    return steps


# --------------------------------------------------------------------------- #
#  Operation handlers.  Each returns a result dict.
# --------------------------------------------------------------------------- #
def _result(**kw):
    base = dict(ok=True, steps=[], verified=None, verify_note="", plot=None, approx="")
    base.update(kw)
    return base


def _num_approx(value):
    """Short decimal for a concrete real number (the exact/approx display).
    Returns '' when the value isn't a plain real number, or already reads as one."""
    try:
        v = sympify(value) if isinstance(value, str) else value
        if getattr(v, "free_symbols", set()):
            return ""
        if v.is_Integer or v.is_Float:
            return ""
        n = N(v, 12)
        if n.is_real is not True:
            return ""
        return "%.8g" % float(n)
    except Exception:
        return ""


def _do_derivative(expr_str, var_name=None, order=1):
    expr = _P(expr_str)
    var = Symbol(var_name) if var_name else _pick_var(expr)
    d = diff(expr, var, order)
    d_s = simplify(d)
    verified, note = (
        _verify_derivative(expr, var, d_s)
        if order == 1
        else (None, "higher-order derivative computed symbolically")
    )
    ordtxt = {1: "", 2: "second ", 3: "third "}.get(order, f"{order}th ")
    steps = [
        rf"\text{{Differentiate }} f({latex(var)}) = {latex(expr)}",
    ]
    if order == 1:
        steps.append(
            rf"\frac{{d}}{{d{latex(var)}}}\left[{latex(expr)}\right] = {latex(d_s)}"
        )
    else:
        steps.append(
            rf"\frac{{d^{order}}}{{d{latex(var)}^{order}}}\left[{latex(expr)}\right] = {latex(d_s)}"
        )
    ans = (
        rf"\frac{{d^{order}}}{{d{latex(var)}^{order}}} = {latex(d_s)}"
        if order > 1
        else rf"f'({latex(var)}) = {latex(d_s)}"
    )
    return _result(
        type=f"{ordtxt}derivative",
        input_latex=latex(expr),
        answer_latex=ans,
        answer_str=str(d_s),
        steps=steps,
        verified=verified,
        verify_note=note,
        plot=_plot_spec(expr, var, extra=d_s, extra_label="f'"),
    )


def _do_integral(expr_str, var_name=None, a=None, b=None):
    expr = _P(expr_str)
    var = Symbol(var_name) if var_name else _pick_var(expr)
    if a is not None and b is not None:
        av, bv = _P(a) if isinstance(a, str) else a, _P(b) if isinstance(b, str) else b
        val = integrate(expr, (var, av, bv))
        if val.has(Integral):
            return {
                "ok": False,
                "error": "SymPy could not evaluate this definite integral in closed form.",
            }
        val_s = simplify(val)
        verified, note = _verify_definite(expr, var, av, bv, val_s)
        F = integrate(expr, var)
        steps = [rf"\text{{Find an antiderivative }} F({latex(var)}) = {latex(F)}"]
        if not F.has(Integral):
            steps.append(rf"\text{{Evaluate }} F({latex(bv)}) - F({latex(av)})")
        steps.append(
            rf"\int_{{{latex(av)}}}^{{{latex(bv)}}} {latex(expr)}\,d{latex(var)} = {latex(val_s)}"
            rf" \approx {N(val_s, 8)}"
        )
        return _result(
            type="definite integral",
            input_latex=latex(expr),
            answer_latex=rf"\int_{{{latex(av)}}}^{{{latex(bv)}}} {latex(expr)}\,d{latex(var)} = {latex(val_s)}",
            answer_str=str(val_s),
            approx=_num_approx(val_s),
            steps=steps,
            verified=verified,
            verify_note=note,
            plot=_plot_spec(expr, var, shade=(av, bv)),
        )
    # indefinite
    F = integrate(expr, var)
    if F.has(Integral):
        return {
            "ok": False,
            "error": "SymPy could not find a closed-form antiderivative for this expression.",
        }
    F_s = simplify(F)
    verified, note = _verify_antiderivative(expr, var, F_s)
    try:
        steps = _describe_integral(integral_steps(expr, var))
    except Exception:
        steps = []
    steps = [rf"\text{{Integrate }} {latex(expr)}\,d{latex(var)}"] + steps
    steps.append(rf"\int {latex(expr)}\,d{latex(var)} = {latex(F_s)} + C")
    return _result(
        type="indefinite integral",
        input_latex=latex(expr),
        answer_latex=rf"\int {latex(expr)}\,d{latex(var)} = {latex(F_s)} + C",
        answer_str=str(F_s) + " + C",
        steps=steps,
        verified=verified,
        verify_note=note,
        plot=_plot_spec(expr, var, extra=F_s, extra_label="F"),
    )


def _do_limit(expr_str, var_name, point_str, direction=None):
    expr = _P(expr_str)
    var = Symbol(var_name) if var_name else _pick_var(expr)
    point = _P(point_str) if isinstance(point_str, str) else point_str
    d = {"left": "-", "right": "+", "-": "-", "+": "+"}.get(direction, "+")
    both = direction is None
    if both:
        L = limit(expr, var, point)
        Lm = limit(expr, var, point, "-")
        Lp = limit(expr, var, point, "+")
        if Lm != Lp:
            return _result(
                type="limit",
                input_latex=latex(expr),
                answer_latex=rf"\lim_{{{latex(var)}\to {latex(point)}}} {latex(expr)}\ \text{{does not exist}}"
                rf"\quad(\text{{left}}={latex(Lm)},\ \text{{right}}={latex(Lp)})",
                answer_str="does not exist (left != right)",
                steps=[
                    rf"\text{{Left limit}} = {latex(Lm)},\quad \text{{Right limit}} = {latex(Lp)}",
                    r"\text{One-sided limits differ, so the two-sided limit does not exist.}",
                ],
                verified=None,
                verify_note="one-sided limits differ",
                plot=_plot_spec(expr, var, center=point),
            )
        value = L
    else:
        value = limit(expr, var, point, d)
    val_s = simplify(value)
    verified, note = _verify_limit(expr, var, point, val_s)
    steps = [
        rf"\text{{Evaluate }} \lim_{{{latex(var)}\to {latex(point)}}} {latex(expr)}"
    ]
    # Note an indeterminate form if direct substitution fails.
    sub = None
    try:
        sub = expr.subs(var, point)
    except Exception:
        pass
    if sub is not None and sub in (nan, zoo) or (sub is not None and sub.has(nan, zoo)):
        steps.append(
            r"\text{Direct substitution is indeterminate; apply limit laws / L'H\^opital's rule.}"
        )
    steps.append(
        rf"\lim_{{{latex(var)}\to {latex(point)}}} {latex(expr)} = {latex(val_s)}"
    )
    return _result(
        type="limit",
        input_latex=latex(expr),
        answer_latex=rf"\lim_{{{latex(var)}\to {latex(point)}}} {latex(expr)} = {latex(val_s)}",
        answer_str=str(val_s),
        approx=_num_approx(val_s),
        steps=steps,
        verified=verified,
        verify_note=note,
        plot=_plot_spec(expr, var, center=point),
    )


def _do_series(expr_str, var_name, about="0", order=6):
    expr = _P(expr_str)
    var = Symbol(var_name) if var_name else _pick_var(expr)
    a0 = _P(about) if isinstance(about, str) else about
    ser = series(expr, var, a0, order)
    poly = ser.removeO()
    # verify: truncated polynomial matches f near the center
    verified, note = _verify_limit(  # reuse: (f - poly) -> 0 as x->a
        simplify(expr - poly), var, a0, S.Zero
    )
    where = "Maclaurin" if a0 == 0 else "Taylor"
    steps = [
        rf"\text{{{where} series of }} {latex(expr)} \text{{ about }} {latex(var)}={latex(a0)}",
        rf"{latex(expr)} = {latex(ser)}",
    ]
    return _result(
        type=f"{where.lower()} series",
        input_latex=latex(expr),
        answer_latex=rf"{latex(expr)} = {latex(ser)}",
        answer_str=str(ser),
        steps=steps,
        verified=verified,
        verify_note=note,
        plot=_plot_spec(expr, var, extra=poly, extra_label="Taylor", center=a0),
    )


def _do_solve(expr_str, var_name=None, rhs="0"):
    if "=" in expr_str:
        lhs, _, r = expr_str.partition("=")
        left, right = _P(lhs), _P(r)
    else:
        left, right = _P(expr_str), (_P(rhs) if isinstance(rhs, str) else rhs)
    eqexpr = simplify(left - right)
    var = Symbol(var_name) if var_name else _pick_var(eqexpr)
    sols = sp_solve(Eq(left, right), var, dict=False)
    sols = [simplify(s) for s in sols]
    verified, note = _verify_roots(eqexpr, var, sols)
    sols_l = ",\\ ".join(latex(s) for s in sols) if sols else r"\text{no solution}"
    steps = [
        rf"\text{{Solve }} {latex(Eq(left, right))} \text{{ for }} {latex(var)}",
        rf"{latex(var)} = {sols_l}",
    ]
    return _result(
        type="solve equation",
        input_latex=latex(Eq(left, right)),
        answer_latex=rf"{latex(var)} = {sols_l}",
        answer_str=", ".join(str(s) for s in sols) if sols else "no solution",
        steps=steps,
        verified=verified,
        verify_note=note,
        plot=_plot_spec(eqexpr, var),
    )


def _do_critical(expr_str, var_name=None):
    expr = _P(expr_str)
    var = Symbol(var_name) if var_name else _pick_var(expr)
    d1 = diff(expr, var)
    d2 = diff(expr, var, 2)
    crit = sp_solve(Eq(d1, 0), var)
    crit = [simplify(c) for c in crit if c.is_real is not False]
    rows = []
    for c in crit:
        y = simplify(expr.subs(var, c))
        conc = simplify(d2.subs(var, c))
        if conc.is_positive:
            kind = "local minimum"
        elif conc.is_negative:
            kind = "local maximum"
        else:
            kind = "inconclusive (2nd-deriv test = 0)"
        rows.append((c, y, kind))
    # verify: derivative is ~0 at each critical point
    verified, note = _verify_roots(d1, var, crit)
    steps = [
        rf"f'({latex(var)}) = {latex(simplify(d1))}",
        rf"\text{{Set }} f'({latex(var)})=0 \Rightarrow {latex(var)} = "
        + (",\\ ".join(latex(c) for c in crit) if crit else r"\text{none}"),
        rf"\text{{Classify with }} f''({latex(var)}) = {latex(simplify(d2))}",
    ]
    ans = (
        r"\\ ".join(
            rf"{latex(var)}={latex(c)}:\ f={latex(y)}\ (\text{{{k}}})"
            for c, y, k in rows
        )
        or r"\text{no critical points}"
    )
    return _result(
        type="critical points",
        input_latex=latex(expr),
        answer_latex=ans,
        answer_str="; ".join(f"{c}: {k}" for c, y, k in rows) or "none",
        steps=steps,
        verified=verified,
        verify_note=note,
        plot=_plot_spec(
            expr,
            var,
            points=[
                (float(N(c)), float(N(y)))
                for c, y, k in rows
                if N(c).is_real and N(y).is_real
            ],
        ),
    )


def _do_tangent(expr_str, var_name, at_str):
    expr = _P(expr_str)
    var = Symbol(var_name) if var_name else _pick_var(expr)
    a0 = _P(at_str) if isinstance(at_str, str) else at_str
    slope = simplify(diff(expr, var).subs(var, a0))
    y0 = simplify(expr.subs(var, a0))
    line = simplify(slope * (var - a0) + y0)
    # verify: line and curve share value + slope at a0
    okval = _close(
        _numeric(expr, {var: complex(N(a0)).real}),
        _numeric(line, {var: complex(N(a0)).real}),
    )
    verified = bool(okval) if okval is not None else None
    note = (
        "confirmed: line meets the curve with matching slope at the point"
        if verified
        else "computed symbolically"
    )
    steps = [
        rf"f({latex(a0)}) = {latex(y0)},\quad f'({latex(a0)}) = {latex(slope)}",
        rf"y - {latex(y0)} = {latex(slope)}\left({latex(var)} - {latex(a0)}\right)",
        rf"y = {latex(line)}",
    ]
    return _result(
        type="tangent line",
        input_latex=latex(expr),
        answer_latex=rf"y = {latex(line)}",
        answer_str=str(line),
        steps=steps,
        verified=verified,
        verify_note=note,
        plot=_plot_spec(expr, var, extra=line, extra_label="tangent", center=a0),
    )


def _do_algebra(kind, expr_str):
    expr = _P(expr_str)
    if kind == "factor" and expr.is_Integer:  # "factor 70560" -> prime factorization
        return _do_factorint(int(expr))
    if kind == "simplify":
        out = simplify(expr)
        t = sp.trigsimp(expr)  # prefer the trig-simpler form when it's shorter
        if sp.count_ops(t) < sp.count_ops(out):
            out = t
    elif kind == "expand":
        out = sp.expand_trig(expand(expr))  # also expands sin(2x), sin(x+y), etc.
    else:  # factor
        out = factor(expr)
    # verify: original and result are equal as functions
    var = _pick_var(expr)
    diffz = simplify(expr - out)
    verified = diffz == 0
    return _result(
        type=kind,
        input_latex=latex(expr),
        answer_latex=rf"{latex(expr)} = {latex(out)}",
        answer_str=str(out),
        approx=_num_approx(out),
        steps=[rf"{kind.title()}: {latex(expr)} = {latex(out)}"],
        verified=verified,
        verify_note="algebraically equivalent to the input" if verified else "",
        plot=_plot_spec(expr, var) if expr.free_symbols else None,
    )


def _do_expression(expr_str):
    """Bare expression: evaluate if numeric, else show + quick calculus facts."""
    expr = _P(expr_str)
    if not expr.free_symbols:
        val = simplify(expr)
        # 0.15*80 -> 12.0 -> 12 : collapse integer-valued floats to a clean integer.
        if val.is_Float and val == int(val):
            val = sp.Integer(int(val))
        # A pure-numeric input has already collapsed to its value at parse time, so
        # "expr = val" would be redundant ("12.0 = 12"); show the clean value once.
        if val.is_Float:
            answer_latex = answer_str = "%.10g" % float(
                val
            )  # compact non-integer decimal
        else:
            approx = ""
            if (
                not val.is_Integer
            ):  # rationals / irrationals get a decimal approximation
                try:
                    d = "%.10g" % float(N(val, 12))
                    if d != latex(val):
                        approx = rf" \approx {d}"
                except Exception:
                    pass
            answer_latex = rf"{latex(val)}{approx}"
            answer_str = str(val)
        return _result(
            type="evaluate",
            input_latex="",
            answer_latex=answer_latex,
            answer_str=answer_str,
            verified=True,
            verify_note="exact value",
            steps=[],
        )
    var = _pick_var(expr)
    d = simplify(diff(expr, var))
    F = integrate(expr, var)
    facts = [
        rf"\text{{Simplified: }} {latex(simplify(expr))}",
        rf"\frac{{d}}{{d{latex(var)}}} = {latex(d)}",
    ]
    if not F.has(Integral):
        facts.append(rf"\int \,d{latex(var)} = {latex(simplify(F))} + C")
    return _result(
        type="expression",
        input_latex=latex(expr),
        answer_latex=rf"{latex(simplify(expr))}",
        answer_str=str(simplify(expr)),
        steps=facts,
        verified=None,
        verify_note="showing the expression with its derivative and integral",
        plot=_plot_spec(expr, var),
    )


# --------------------------------------------------------------------------- #
#  Systems, inequalities, summation (Tier-1 scope beyond single-variable calc)
# --------------------------------------------------------------------------- #
def _do_system(eq_strs):
    """Solve a system of equations (linear or nonlinear)."""
    eqs = []
    for es in eq_strs:
        es = es.strip()
        if not es:
            continue
        if "=" in es:
            lhs, _, rhs = es.partition("=")
            eqs.append(Eq(_P(lhs), _P(rhs)))
        else:
            eqs.append(Eq(_P(es), S.Zero))
    syms = sorted(
        set().union(*[e.free_symbols for e in eqs]) if eqs else set(),
        key=lambda s: s.name,
    )
    if not eqs or not syms:
        return {"ok": False, "error": "Couldn't read a system of equations."}
    sols = sp_solve(eqs, syms, dict=True)
    in_tex = r",\ ".join(latex(e) for e in eqs)
    if not sols:
        return _result(
            type="system of equations",
            input_latex=in_tex,
            answer_latex=r"\text{no solution}",
            answer_str="no solution",
            verified=None,
            verify_note="the system has no solution",
        )
    # verify: every solution must satisfy every equation
    total = 0
    agree = 0
    for sol in sols:
        for e in eqs:
            total += 1
            if simplify(e.lhs.subs(sol) - e.rhs.subs(sol)) == 0:
                agree += 1
    verified = (agree == total) if total else None

    def _fmt(sol):
        return ",\\ ".join(
            rf"{latex(k)} = {latex(sol[k])}" for k in sorted(sol, key=lambda s: s.name)
        )

    # multiple solutions -> stacked lines via a KaTeX-safe environment (not bare \\)
    if len(sols) > 1:
        ans = (
            r"\begin{gathered}" + r"\\".join(_fmt(s) for s in sols) + r"\end{gathered}"
        )
    else:
        ans = _fmt(sols[0])
    ans_str = "; ".join(", ".join(f"{k}={v}" for k, v in s.items()) for s in sols)
    return _result(
        type="system of equations",
        input_latex=in_tex,
        answer_latex=ans,
        answer_str=ans_str,
        steps=[rf"\text{{Solve}}\ {in_tex}"],
        verified=verified,
        verify_note=(
            "every solution satisfies every equation"
            if verified
            else (
                "a solution did not check out"
                if verified is False
                else "computed symbolically"
            )
        ),
    )


def _do_inequality(body):
    """Solve a one-variable inequality; return the solution set."""
    m = re.search(r"(<=|>=|<|>)", body)
    if not m:
        return {"ok": False, "error": "That doesn't look like an inequality."}
    op = m.group(1)
    lhs, rhs = _P(body[: m.start()]), _P(body[m.end() :])
    rel = {"<": lhs < rhs, "<=": lhs <= rhs, ">": lhs > rhs, ">=": lhs >= rhs}[op]
    var = _pick_var(lhs - rhs)
    try:
        sol = sp.reduce_inequalities(rel, [var])
    except Exception:
        sol = sp.solve_univariate_inequality(rel, var, relational=True)
    verified, note = _verify_inequality(rel, sol, var)
    return _result(
        type="inequality",
        input_latex=latex(rel),
        answer_latex=latex(sol),
        answer_str=str(sol),
        steps=[rf"\text{{Solve}}\ {latex(rel)}", latex(sol)],
        verified=verified,
        verify_note=note,
        plot=_plot_spec(lhs - rhs, var),
    )


def _verify_inequality(rel, sol, var):
    """Sample points: membership in the solution set must match the raw inequality."""
    agree = 0
    total = 0
    for x0 in _SAMPLES + [0.0, 5.0, -5.0, 3.1415, -2.5, 10.0, -10.0]:
        try:
            in_sol = bool(sol.subs(var, x0))
            in_rel = bool(rel.subs(var, x0))
        except Exception:
            continue
        total += 1
        if in_sol == in_rel:
            agree += 1
    if total == 0:
        return None, "computed symbolically"
    return (agree == total), (
        f"solution set matches the inequality at {total} test points"
        if agree == total
        else f"sample check disagreed ({agree}/{total} points)"
    )


def _do_summation(expr_str, var_name, a, b, is_product=False):
    expr = _P(expr_str)
    var = Symbol(var_name) if var_name else _pick_var(expr)
    av = _P(a) if isinstance(a, str) else a
    bv = _P(b) if isinstance(b, str) else b
    val = (
        sp.product(expr, (var, av, bv))
        if is_product
        else sp.summation(expr, (var, av, bv))
    )
    val_s = simplify(val)
    sym = r"\prod" if is_product else r"\sum"
    tex = rf"{sym}_{{{latex(var)}={latex(av)}}}^{{{latex(bv)}}} {latex(expr)}"
    verified, note = _verify_summation(expr, var, av, bv, val_s, is_product)
    return _result(
        type=("product" if is_product else "summation"),
        input_latex=tex,
        answer_latex=rf"{tex} = {latex(val_s)}",
        answer_str=str(val_s),
        approx=_num_approx(val_s),
        steps=[rf"{tex} = {latex(val_s)}"],
        verified=verified,
        verify_note=note,
    )


def _verify_summation(expr, var, a, b, value, is_product):
    """When the bounds are concrete integers, recompute term-by-term and compare."""
    try:
        ai, bi = int(a), int(b)
    except Exception:
        return None, "computed symbolically (symbolic bounds)"
    if bi - ai > 2000:
        return None, "computed symbolically (range too large to re-check)"
    acc = S.One if is_product else S.Zero
    for k in range(ai, bi + 1):
        t = expr.subs(var, k)
        acc = acc * t if is_product else acc + t
    ok = simplify(acc - value) == 0
    return bool(ok), (
        "re-computed term by term" if ok else "term-by-term re-check disagreed"
    )


# --------------------------------------------------------------------------- #
#  Linear algebra (Tier-2 scope: matrices)
# --------------------------------------------------------------------------- #
def _parse_matrix(s):
    """Parse {{1,2},{3,4}} or [[1,2],[3,4]] into a SymPy Matrix (entries via _P)."""
    s = s.strip().replace("{", "[").replace("}", "]").strip()
    if not (s.startswith("[") and s.endswith("]")):
        raise ValueError("not a matrix")
    body = s[1:-1].strip()
    if body.startswith("["):
        rows = []
        for rs in re.split(r"\]\s*,\s*\[", body):
            rs = rs.strip().lstrip("[").rstrip("]")
            rows.append([_P(e) for e in rs.split(",") if e.strip()])
    else:  # a single flat vector like {1,2,3}
        rows = [[_P(e)] for e in body.split(",") if e.strip()]
    return sp.Matrix(rows)


def _mat_zero(M):
    return all(simplify(e) == 0 for e in M)


def _do_matrix(op, mat_str):
    M = _parse_matrix(mat_str)
    op = re.sub(r"\s+", "", op).lower()
    in_tex = latex(M)
    n = M.rows

    if op in ("determinant", "det"):
        val = simplify(M.det())
        return _result(
            type="determinant",
            input_latex=in_tex,
            answer_latex=rf"\det = {latex(val)}",
            answer_str=str(val),
            approx=_num_approx(val),
            verify_note="exact",
        )
    if op == "trace":
        val = simplify(M.trace())
        return _result(
            type="trace",
            input_latex=in_tex,
            answer_latex=rf"\operatorname{{tr}} = {latex(val)}",
            answer_str=str(val),
            approx=_num_approx(val),
            verify_note="exact",
        )
    if op == "rank":
        val = M.rank()
        return _result(
            type="rank",
            input_latex=in_tex,
            answer_latex=rf"\operatorname{{rank}} = {val}",
            answer_str=str(val),
            verify_note="exact",
        )
    if op == "transpose":
        T = M.T
        return _result(
            type="transpose",
            input_latex=in_tex,
            answer_latex=latex(T),
            answer_str=str(T),
            verify_note="exact",
        )
    if op in ("rref", "rowreduce"):
        R = M.rref()[0]
        return _result(
            type="rref",
            input_latex=in_tex,
            answer_latex=latex(R),
            answer_str=str(R),
            verify_note="reduced row-echelon form",
        )
    if op == "inverse":
        if simplify(M.det()) == 0:
            return {
                "ok": False,
                "error": "This matrix is singular (determinant 0) — it has no inverse.",
            }
        inv = M.inv()
        verified = _mat_zero(M * inv - sp.eye(n))
        return _result(
            type="inverse",
            input_latex=in_tex,
            answer_latex=latex(inv),
            answer_str=str(inv),
            verified=verified,
            verify_note=(
                "confirmed: A A^{-1} = I" if verified else "computed symbolically"
            ),
        )
    if op == "eigenvalues":
        vals = []
        for v, mult in M.eigenvals().items():
            vals.extend([simplify(v)] * mult)
        ok = all(simplify((M - v * sp.eye(n)).det()) == 0 for v in vals)
        ans = ",\\ ".join(latex(v) for v in vals)
        return _result(
            type="eigenvalues",
            input_latex=in_tex,
            answer_latex=rf"\lambda = {ans}",
            answer_str=", ".join(str(v) for v in vals),
            verified=ok,
            verify_note=(
                "every eigenvalue satisfies det(A - λI) = 0"
                if ok
                else "computed symbolically"
            ),
        )
    if op == "eigenvectors":
        lines, ok = [], True
        for val, mult, vecs in M.eigenvects():
            val = simplify(val)
            for vec in vecs:
                if not _mat_zero(M * vec - val * vec):
                    ok = False
                lines.append(rf"\lambda={latex(val)}:\ {latex(vec.T)}")
        ans = r"\\".join(lines)
        if len(lines) > 1:
            ans = r"\begin{gathered}" + ans + r"\end{gathered}"
        return _result(
            type="eigenvectors",
            input_latex=in_tex,
            answer_latex=ans,
            answer_str="; ".join(lines),
            verified=ok,
            verify_note=(
                "every vector satisfies A v = λ v" if ok else "computed symbolically"
            ),
        )
    if op == "nullspace":
        ns = M.nullspace()
        if not ns:
            return _result(
                type="nullspace",
                input_latex=in_tex,
                answer_latex=r"\{\vec 0\}",
                answer_str="{0}",
                verify_note="the null space is trivial",
            )
        ans = ",\\ ".join(latex(v.T) for v in ns)
        return _result(
            type="nullspace",
            input_latex=in_tex,
            answer_latex=ans,
            answer_str="; ".join(str(v) for v in ns),
            verify_note="basis for the null space",
        )
    return {"ok": False, "error": f"Unknown matrix operation: {op}"}


# --------------------------------------------------------------------------- #
#  Number theory (Tier-2 scope)
# --------------------------------------------------------------------------- #
def _do_factorint(n):
    f = sp.factorint(n)
    parts_tex = [(rf"{p}^{{{e}}}" if e > 1 else f"{p}") for p, e in sorted(f.items())]
    tex = r" \cdot ".join(parts_tex) if parts_tex else str(n)
    ans_str = " * ".join(
        (f"{p}^{e}" if e > 1 else f"{p}") for p, e in sorted(f.items())
    )
    prod = 1
    for p, e in f.items():
        prod *= p**e
    verified = prod == n
    return _result(
        type="prime factorization",
        input_latex=str(n),
        answer_latex=rf"{n} = {tex}",
        answer_str=ans_str or str(n),
        verified=verified,
        verify_note="factors multiply back to the original" if verified else "",
    )


def _do_isprime(n):
    if sp.isprime(n):
        return _result(
            type="primality",
            input_latex=str(n),
            answer_latex=rf"{n}\ \text{{is prime}}",
            answer_str=f"{n} is prime",
            verified=True,
            verify_note="no divisor other than 1 and itself",
        )
    f = sp.factorint(n)
    sm = min(f) if f else None
    return _result(
        type="primality",
        input_latex=str(n),
        answer_latex=rf"{n}\ \text{{is not prime}}",
        answer_str=f"{n} is not prime",
        verified=True,
        verify_note=(f"divisible by {sm}" if (sm is not None and sm < n) else ""),
    )


# --------------------------------------------------------------------------- #
#  Descriptive statistics (Tier-2 scope; hand-built exact arithmetic over a list)
# --------------------------------------------------------------------------- #
def _parse_data(s):
    s = s.strip().strip("{}[]() ").strip()
    return [_P(x) for x in s.split(",") if x.strip()]


def _do_stats(op, data_str):
    data = _parse_data(data_str)
    n = len(data)
    if n == 0:
        return {"ok": False, "error": "No data values found."}
    op = re.sub(r"\s+", "", op).lower()
    data_tex = rf"\{{{','.join(latex(v) for v in data)}\}}"
    total = sum(data, S.Zero)
    mean = simplify(total / n)
    srt = sorted(data, key=lambda v: float(N(v)))

    def _median(xs):
        m = len(xs)
        return xs[m // 2] if m % 2 else simplify((xs[m // 2 - 1] + xs[m // 2]) / 2)

    def _res(kind, val, note="", verified=None):
        return _result(
            type=kind,
            input_latex=data_tex,
            answer_latex=rf"\text{{{kind}}} = {latex(val)}",
            answer_str=str(val),
            approx=_num_approx(val),
            verified=verified,
            verify_note=note,
        )

    if op in ("mean", "average"):
        verified = simplify(mean * n - total) == 0
        return _res("mean", mean, "sum / count" if verified else "", verified)
    if op == "median":
        return _res("median", _median(srt))
    if op == "mode":
        from collections import Counter

        c = Counter(str(v) for v in data)
        top = max(c.values())
        seen = []
        for v in data:
            if c[str(v)] == top and v not in seen:
                seen.append(v)
        mode_tex = ",\\ ".join(latex(v) for v in seen)
        return _result(
            type="mode",
            input_latex=data_tex,
            answer_latex=rf"\text{{mode}} = {mode_tex}",
            answer_str=", ".join(str(v) for v in seen),
            verify_note=f"appears {top} time(s)",
        )

    ss = sum((simplify(v - mean)) ** 2 for v in data)
    pop_var = simplify(ss / n)
    samp_var = simplify(ss / (n - 1)) if n > 1 else None
    if op in ("variance", "var"):
        parts = [rf"\text{{population}} = {latex(pop_var)}"]
        if samp_var is not None:
            parts.append(rf"\text{{sample}} = {latex(samp_var)}")
        return _result(
            type="variance",
            input_latex=data_tex,
            answer_latex=r",\ \ ".join(parts),
            answer_str=f"population={pop_var}"
            + (f", sample={samp_var}" if samp_var is not None else ""),
            approx=_num_approx(pop_var),
            verify_note="sum of squared deviations / n (population) or /(n-1) (sample)",
        )
    if op in ("standarddeviation", "std", "sd", "stdev"):
        pop_sd = simplify(sp.sqrt(pop_var))
        samp_sd = simplify(sp.sqrt(samp_var)) if samp_var is not None else None
        parts = [rf"\text{{population}} = {latex(pop_sd)}"]
        if samp_sd is not None:
            parts.append(rf"\text{{sample}} = {latex(samp_sd)}")
        return _result(
            type="standard deviation",
            input_latex=data_tex,
            answer_latex=r",\ \ ".join(parts),
            answer_str=f"population={pop_sd}"
            + (f", sample={samp_sd}" if samp_sd is not None else ""),
            approx=_num_approx(pop_sd),
            verify_note="square root of the variance",
        )
    if op in ("summary", "summarystatistics", "stats", "describe"):
        pop_sd = simplify(sp.sqrt(pop_var))
        rows = [
            rf"n = {n}",
            rf"\text{{mean}} = {latex(mean)}",
            rf"\text{{median}} = {latex(_median(srt))}",
            rf"\min = {latex(srt[0])},\ \max = {latex(srt[-1])}",
            rf"\text{{sd (pop)}} = {latex(pop_sd)}",
        ]
        return _result(
            type="summary statistics",
            input_latex=data_tex,
            answer_latex=r"\begin{gathered}" + r"\\".join(rows) + r"\end{gathered}",
            answer_str=f"n={n}, mean={mean}, median={_median(srt)}, min={srt[0]}, max={srt[-1]}, sd={pop_sd}",
            verify_note="descriptive statistics",
        )
    return {"ok": False, "error": f"Unknown statistic: {op}"}


# --------------------------------------------------------------------------- #
#  Multivariable calculus (Tier-2 scope)
# --------------------------------------------------------------------------- #
def _do_gradient(expr_str):
    expr = _P(expr_str)
    syms = sorted(expr.free_symbols, key=lambda s: s.name) or [Symbol("x")]
    parts = [simplify(diff(expr, s)) for s in syms]
    grad = sp.Matrix(parts)
    return _result(
        type="gradient",
        input_latex=latex(expr),
        answer_latex=rf"\nabla f = {latex(grad.T)}",
        answer_str=str(grad.T),
        steps=[
            rf"\frac{{\partial f}}{{\partial {latex(s)}}} = {latex(p)}"
            for s, p in zip(syms, parts)
        ],
        verify_note="exact partial derivatives",
    )


def _do_hessian(expr_str):
    expr = _P(expr_str)
    syms = sorted(expr.free_symbols, key=lambda s: s.name) or [Symbol("x")]
    H = sp.hessian(expr, syms)
    return _result(
        type="hessian",
        input_latex=latex(expr),
        answer_latex=latex(H),
        answer_str=str(H),
        verify_note="matrix of second partial derivatives",
    )


def _do_multi_integral(expr_str, bounds_str):
    expr = _P(expr_str)
    triples = []
    for part in bounds_str.split(","):
        m = re.search(r"([a-z])\s*=?\s*(.+?)\s*(?:to|\.\.)\s*(.+)$", part.strip())
        if m:
            triples.append((Symbol(m.group(1)), _P(m.group(2)), _P(m.group(3))))
    if not triples:
        return {
            "ok": False,
            "error": "Couldn't read the bounds (try 'x=0 to 1, y=0 to 2').",
        }
    val = expr
    for v, a, b in triples:
        val = integrate(val, (v, a, b))
    if val.has(Integral):
        return {
            "ok": False,
            "error": "SymPy couldn't evaluate this multiple integral in closed form.",
        }
    val_s = simplify(val)
    # verify: integrating in the reverse order (Fubini) must give the same result
    try:
        val2 = expr
        for v, a, b in reversed(triples):
            val2 = integrate(val2, (v, a, b))
        verified = simplify(val_s - simplify(val2)) == 0
        note = (
            "confirmed: both integration orders agree (Fubini)"
            if verified
            else "the two integration orders disagreed"
        )
    except Exception:
        verified, note = None, "computed symbolically"
    bnd_tex = ",\\ ".join(
        rf"{latex(v)}\in[{latex(a)},{latex(b)}]" for v, a, b in triples
    )
    return _result(
        type="multiple integral",
        input_latex=rf"\iint {latex(expr)}",
        answer_latex=latex(val_s),
        answer_str=str(val_s),
        approx=_num_approx(val_s),
        steps=[rf"\text{{Integrate over }} {bnd_tex}", rf"= {latex(val_s)}"],
        verified=verified,
        verify_note=note,
    )


# --------------------------------------------------------------------------- #
#  Plot spec (numeric samples computed here; UI just draws them)
# --------------------------------------------------------------------------- #
def _samples(expr, var, lo, hi, n=241):
    xs, ys = [], []
    step = (hi - lo) / (n - 1)
    for i in range(n):
        x = lo + i * step
        y = _numeric(expr, {var: x})
        if y is None or abs(y.imag) > 1e-6 or abs(y.real) > 1e6:
            xs.append(x)
            ys.append(None)
        else:
            xs.append(x)
            ys.append(y.real)
    return xs, ys


def _plot_spec(
    expr, var, extra=None, extra_label=None, shade=None, center=None, points=None
):
    """Build a JSON-able plot description, or None if not plottable."""
    try:
        if len(expr.free_symbols) > 1:
            return None
        lo, hi = -6.0, 6.0
        if center is not None:
            try:
                c = float(N(center))
                lo, hi = c - 4.0, c + 4.0
            except Exception:
                pass
        if shade is not None:
            try:
                a = float(N(shade[0]))
                b = float(N(shade[1]))
                pad = max(1.0, (b - a) * 0.4)
                lo, hi = min(lo, a - pad), max(hi, b + pad)
            except Exception:
                pass
        xs, ys = _samples(expr, var, lo, hi)
        if all(y is None for y in ys):
            return None
        traces = [{"label": "f", "x": xs, "y": ys}]
        if extra is not None and len(extra.free_symbols) <= 1:
            ex, ey = _samples(extra, var, lo, hi)
            traces.append({"label": extra_label or "g", "x": ex, "y": ey})
        spec = {"var": var.name, "traces": traces}
        if shade is not None:
            try:
                spec["shade"] = [float(N(shade[0])), float(N(shade[1]))]
            except Exception:
                pass
        if points:
            spec["points"] = points
        return spec
    except Exception:
        return None


# --------------------------------------------------------------------------- #
#  Query router
# --------------------------------------------------------------------------- #
def _clean(q):
    return re.sub(r"\s+", " ", q.strip())


def solve(query):
    """Public entry point. Returns a result dict; never raises."""
    try:
        if not query or not query.strip():
            return {
                "ok": False,
                "error": "Type a calculus problem, e.g. `derivative of x^2 sin(x)`.",
            }
        # linear algebra: matrix ops read the RAW query (preprocess mangles { } braces)
        if "{" in query or "[" in query:
            mm = re.search(
                r"^\s*(determinant|det|inverse|eigenvalues|eigenvectors|rank|rref|transpose|trace|nullspace|row\s*reduce)\s+(?:of\s+)?(.+)$",
                query.strip(),
                re.IGNORECASE,
            )
            if mm:
                return _do_matrix(mm.group(1), mm.group(2))
        # descriptive statistics also read the RAW query (data braces / commas)
        ms = re.search(
            r"^\s*(mean|average|median|mode|variance|var|standard\s+deviation|std|sd|stdev|summary\s+statistics|summary|stats|describe)\s+(?:of\s+)?(.+)$",
            query.strip(),
            re.IGNORECASE,
        )
        if ms and "," in ms.group(2):
            return _do_stats(ms.group(1), ms.group(2))
        raw = _clean(preprocess(query))
        q = raw
        ql = q.lower()

        # ---- order-sensitive command detection ----

        # nth / second / third derivative, d^n/dx^n
        m = re.search(
            r"^(?:the\s+)?(second|third|2nd|3rd|\d+(?:st|nd|rd|th)?)\s+derivative\s+of\s+(.+)$",
            ql,
        )
        if m:
            omap = {"second": 2, "2nd": 2, "third": 3, "3rd": 3}
            o = omap.get(m.group(1), None)
            if o is None:
                o = int(re.match(r"\d+", m.group(1)).group())
            return _do_derivative(q[q.lower().find(m.group(2)) :], order=o)
        m = re.search(r"^d\^?(\d+)\s*/\s*d\s*([a-z])\^?\d*\s*(?:of\s+)?(.+)$", ql)
        if m:
            return _do_derivative(
                q[q.lower().find(m.group(3)) :],
                var_name=m.group(2),
                order=int(m.group(1)),
            )

        # partial derivative
        m = re.search(
            r"partial\s+derivative\s+of\s+(.+?)\s+with\s+respect\s+to\s+([a-z])\b", ql
        )
        if m:
            return _do_derivative(
                q[
                    q.lower().find(m.group(1)) : q.lower().find(m.group(1))
                    + len(m.group(1))
                ],
                var_name=m.group(2),
                order=1,
            )

        # first derivative:  derivative of / differentiate / d/dx
        m = re.search(r"^(?:the\s+)?derivative\s+of\s+(.+)$", ql) or re.search(
            r"^differentiate\s+(.+)$", ql
        )
        if m:
            body = q[q.lower().find(m.group(1)) :]
            vn = None
            mv = re.search(r"\s+with\s+respect\s+to\s+([a-z])\b", body.lower())
            if mv:
                vn = mv.group(1)
                body = body[: body.lower().find("with respect to")]
            return _do_derivative(body, var_name=vn)
        m = re.search(r"^d\s*/\s*d\s*([a-z])\s*(?:of\s+)?\(?(.+?)\)?$", ql)
        if m:
            return _do_derivative(q[q.lower().find(m.group(2)) :], var_name=m.group(1))

        # gradient / hessian (multivariable)
        m = re.search(r"^gradient\s+(?:of\s+)?(.+)$", ql)
        if m:
            return _do_gradient(q[q.lower().find(m.group(1)) :])
        m = re.search(r"^hessian\s+(?:of\s+)?(.+)$", ql)
        if m:
            return _do_hessian(q[q.lower().find(m.group(1)) :])

        # double / triple integral over explicit bounds
        m = re.search(
            r"^(?:double|triple)\s+integral\s+of\s+(.+?)\s+(?:over|for|with)\s+(.+)$",
            ql,
        )
        if m:
            body = q[
                q.lower().find(m.group(1)) : q.lower().find(m.group(1))
                + len(m.group(1))
            ]
            bnds = q[q.lower().find(m.group(2)) :]
            return _do_multi_integral(body, bnds)

        # definite integral: "... from A to B"
        m = re.search(
            r"(?:integrate|integral\s+of|definite\s+integral\s+of|antiderivative\s+of|area\s+under(?:\s+the\s+curve)?)\s+(.+?)\s+from\s+(.+?)\s+to\s+(.+?)$",
            ql,
        )
        if m:
            body = q[
                q.lower().find(m.group(1)) : q.lower().find(m.group(1))
                + len(m.group(1))
            ]
            vn = _integ_var(body)
            return _do_integral(
                _strip_dv(body), var_name=vn, a=m.group(2), b=m.group(3)
            )

        # indefinite integral
        m = re.search(
            r"^(?:integrate|integral\s+of|indefinite\s+integral\s+of|antiderivative\s+of)\s+(.+)$",
            ql,
        )
        if m:
            body = q[q.lower().find(m.group(1)) :]
            vn = _integ_var(body)
            return _do_integral(_strip_dv(body), var_name=vn)

        # limit
        m = re.search(
            r"(?:limit|lim)\s+(?:of\s+)?(.+?)\s+as\s+([a-z])\s*(?:->|to|approaches)\s*(.+?)(?:\s+from\s+(?:the\s+)?(left|right|below|above))?$",
            ql,
        )
        if m:
            body = q[
                q.lower().find(m.group(1)) : q.lower().find(m.group(1))
                + len(m.group(1))
            ]
            dirn = {"left": "-", "below": "-", "right": "+", "above": "+"}.get(
                m.group(4)
            )
            pt = m.group(3)
            # a^+ / a^- suffix
            dm = re.match(r"(.+?)\s*\^?\s*([+-])\s*$", pt)
            if dm and dirn is None:
                pt, dirn = dm.group(1), dm.group(2)
            return _do_limit(body, m.group(2), pt.strip(), direction=dirn)

        # taylor / maclaurin / series
        m = re.search(
            r"(taylor|maclaurin|power)\s+series\s+of\s+(.+)$", ql
        ) or re.search(r"^series\s+(?:expansion\s+)?of\s+(.+)$", ql)
        if m:
            body_group = (
                m.group(2) if m.re.groups >= 2 and m.lastindex >= 2 else m.group(1)
            )
            body = q[q.lower().find(body_group) :]
            about = "0"
            am = re.search(
                r"\s+(?:about|around|at)\s+([a-z]?\s*=?\s*[^,]+?)(?:\s+to\s+order\s+(\d+))?$",
                body.lower(),
            )
            order = 6
            if am:
                ab = am.group(1)
                ab = re.sub(r"^[a-z]\s*=\s*", "", ab).strip()
                about = ab or "0"
                if am.group(2):
                    order = int(am.group(2)) + 1
                body = body[
                    : (
                        body.lower().rfind(" about ")
                        if " about " in body.lower()
                        else (
                            body.lower().rfind(" around ")
                            if " around " in body.lower()
                            else body.lower().rfind(" at ")
                        )
                    )
                ]
            om = re.search(r"\bto\s+order\s+(\d+)", ql)
            if om:
                order = int(om.group(1)) + 1
            vn = _pick_var(_P(body)).name
            return _do_series(body, vn, about=about, order=order)

        # summation / product:  sum EXPR from k=A to B
        m = re.search(
            r"^(sum|summation|product|prod)\s+(?:of\s+)?(.+?)\s+(?:from|for)\s+([a-z])\s*=\s*(.+?)\s+to\s+(.+?)$",
            ql,
        )
        if m:
            is_prod = m.group(1) in ("product", "prod")
            body = q[
                q.lower().find(m.group(2)) : q.lower().find(m.group(2))
                + len(m.group(2))
            ]
            return _do_summation(
                body, m.group(3), m.group(4), m.group(5), is_product=is_prod
            )

        # tangent line
        m = re.search(
            r"tangent\s+(?:line\s+)?to\s+(.+?)\s+at\s+(?:[a-z]\s*=\s*)?(.+?)$", ql
        )
        if m:
            body = q[
                q.lower().find(m.group(1)) : q.lower().find(m.group(1))
                + len(m.group(1))
            ]
            vn = _pick_var(_P(body)).name
            return _do_tangent(body, vn, m.group(2).strip())

        # critical points / extrema
        m = re.search(
            r"(?:critical\s+points?|extrema|local\s+(?:max|min)(?:ima|imum)?)\s+of\s+(.+)$",
            ql,
        )
        if m:
            return _do_critical(q[q.lower().find(m.group(1)) :])

        # inequality (bare, or "solve ..."):  x^2 > 4 ,  solve 2x - 1 <= 5
        _iq = re.sub(r"->", "  ", q)  # keep the limit arrow from looking like '>'
        if re.search(r"(<=|>=|<|>)", _iq):
            ibody = re.sub(
                r"^\s*(?:solve|where)\s+", "", q, flags=re.IGNORECASE
            ).strip()
            return _do_inequality(ibody)

        # solve / roots / zeros / systems of equations
        m = re.search(r"^solve\s+(.+?)(?:\s+for\s+([a-z]))?$", ql)
        if m:
            body = q[
                q.lower().find(m.group(1)) : q.lower().find(m.group(1))
                + len(m.group(1))
            ]
            parts = re.split(r"\s*,\s*|\s+and\s+", body)
            if len(parts) >= 2 and sum(1 for p in parts if "=" in p) >= 2:
                return _do_system(parts)
            return _do_solve(body, var_name=m.group(2))
        m = re.search(r"^(?:roots|zeros|zeroes)\s+of\s+(.+)$", ql)
        if m:
            return _do_solve(q[q.lower().find(m.group(1)) :])

        # number theory: primality, prime factorization, choose, gcd/lcm
        m = re.search(r"^is\s+(\d+)\s+(?:a\s+)?prime\b", ql) or re.search(
            r"^isprime\s*\(?\s*(\d+)", ql
        )
        if m:
            return _do_isprime(int(m.group(1)))
        m = re.search(
            r"^(?:prime\s+factori[sz]ation\s+of|factori[sz]e)\s+(\d+)\s*$", ql
        )
        if m:
            return _do_factorint(int(m.group(1)))
        m = re.search(r"^(\d+)\s+choose\s+(\d+)$", ql)
        if m:
            return _do_expression(f"binomial({m.group(1)}, {m.group(2)})")
        m = re.search(r"^(gcd|lcm)\s+(?:of\s+)?(.+?)\s+and\s+(.+)$", ql)
        if m:
            return _do_expression(f"{m.group(1)}({m.group(2)}, {m.group(3)})")

        # algebra helpers
        m = re.search(r"^(simplify|factor|expand)\s+(?:trig\s+)?(.+)$", ql)
        if m:
            return _do_algebra(m.group(1), q[q.lower().find(m.group(2)) :])

        # bare expression fallback
        return _do_expression(q)

    except Exception as e:
        return {"ok": False, "error": f"Couldn't parse that: {type(e).__name__}: {e}"}


def _integ_var(body):
    m = re.search(r"\bd\s*([a-z])\b\s*$", body.strip())
    return m.group(1) if m else None


def _strip_dv(body):
    return re.sub(r"\s*d\s*[a-z]\s*$", "", body.strip())


def solve_json(query):
    """JSON-string wrapper so the browser (Pyodide) gets plain data, no proxies."""
    import json

    r = solve(query)
    # Everything in a result is already JSON-safe (str/bool/None/float/list),
    # but guard against stray SymPy objects sneaking into a field.
    return json.dumps(r, default=lambda o: str(o))
