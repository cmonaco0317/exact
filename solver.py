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

import cmath
import math
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


def _stable_seed(text):
    """FNV-1a. Deliberately not Python's hash(), which is salted per process
    (PYTHONHASHSEED) and would make verification non-reproducible run-to-run."""
    h = 0x811C9DC5
    for ch in str(text):
        h ^= ord(ch) & 0xFF
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def _samples_for(expr):
    """Sample points for the numeric check, jittered per-expression.

    The points used to be a fixed public list of 8 values. That made any blind
    spot in the verifier *permanent* rather than random: an answer that happened
    to agree at exactly those 8 abscissae would pass forever, and the list is
    right there in the source to tune against.

    Deriving the jitter from a stable hash of the expression keeps every property
    the fixed list was there for -- the same input yields the same points on every
    run, and identically under CPython and Pyodide, so a result never flickers
    between verified and not -- while making the abscissae unpredictable across
    *different* problems. The base spread is preserved so coverage stays good.
    """
    rnd = _stable_seed(expr)
    out = []
    for base in _SAMPLES:
        rnd = (1103515245 * rnd + 12345) & 0x7FFFFFFF  # LCG, reproducible anywhere
        jitter = (rnd / 0x7FFFFFFF - 0.5) * 0.7  # +/- 0.35
        pt = base + jitter
        if abs(pt) < 0.05:  # keep away from 0, where 1/x-style terms blow up
            pt += 0.4
        out.append(round(pt, 6))
    return out


# Complex sample points, same rationale. Used to re-check complex-analysis
# answers against plain Python complex arithmetic.
_CSAMPLES = [
    complex(0.7213, 1.3179),
    complex(-1.1427, 0.5361),
    complex(1.9043, -0.8112),
    complex(-0.4519, -1.6234),
]


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


class UserError(ValueError):
    """An error whose message is written for the user and is safe to display.

    Everything else -- SyntaxError, TypeError, SympifyError and friends -- is an
    implementation detail. Those used to be printed verbatim ("TypeError:
    unsupported operand type(s) for ** or pow(): 'FunctionClass' and 'Integer'"),
    which tells the user nothing and reads like a crash.
    """


_GENERIC_PARSE_ERROR = (
    "I couldn't read that. Check for a missing parenthesis or operator — "
    "e.g. `cos^2(x)` rather than `cos^2 x`. If it's a word problem, try "
    "phrasing it as an expression: `derivative of x^2 sin(x)`."
)


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


# Complex analysis is the one place users reliably write lowercase `i` (3+4i).
# SymPy only knows the capital form, so plain `i` parses as a free symbol and
# every complex op silently returns nonsense: re(3+4i) -> 4*re(i) + 3. Alias it,
# but ONLY for complex handlers -- `i` is the conventional summation index
# everywhere else, and `sum of i^2 from 1 to 10` must keep treating it as one.
_COMPLEX_LOCALS = dict(_LOCALS, i=I)


def _P(expr_str, locals_=None):
    """Parse an expression string into a SymPy expression (may raise)."""
    expr_str = expr_str.strip().strip(".").strip()
    # strip a trailing "dx" / "d x" if it slipped through
    expr_str = re.sub(r"\s*d\s*[a-zA-Z]\s*$", "", expr_str)
    if not expr_str:
        raise UserError(
            "There's no expression there — type something like `derivative of x^2`."
        )
    return parse_expr(
        expr_str,
        transformations=_TRANSFORMS,
        local_dict=dict(_LOCALS if locals_ is None else locals_),
    )


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


# Tolerance for the central-difference derivative check. The measured error
# floor on *correct* answers is ~1.5e-9 (worst case sin(10x), dominated by
# O(h^2) truncation plus cancellation in (f(x+h)-f(x-h))/2h). 1e-7 sits ~66x
# above that floor: tight enough that "verified" means matched, loose enough
# that it has never false-flagged a correct symbolic answer across the suite.
# It was 1e-3, which silently rubber-stamped errors as large as 0.1%.
# (Re-measured after sample points became per-expression: floor unchanged.)
_DERIV_TOL = 1e-7


def _verify_derivative(f, var, dydx):
    """Central finite-difference check of a derivative at several points.

    Any *other* free symbols are pinned to sample values too, so partial
    derivatives get a real check. Previously only `var` was substituted, which
    left `d/dy [x^2 y]` un-evaluable — every partial derivative silently fell
    through to "computed symbolically".
    """
    h = 1e-5
    agree = 0
    total = 0
    others = [s for s in sorted(f.free_symbols, key=lambda s: s.name) if s != var]
    pts = _samples_for(f)
    for k, x0 in enumerate(pts):
        # NB: build these by copy-then-assign, not dict(fixed, **{var: ...}) —
        # ** requires string keys and `var` is a Symbol.
        fixed = {s: pts[(k + i + 1) % len(pts)] for i, s in enumerate(others)}
        at_up, at_dn, at_0 = dict(fixed), dict(fixed), dict(fixed)
        at_up[var], at_dn[var], at_0[var] = x0 + h, x0 - h, x0
        fp = _numeric(f, at_up)
        fm = _numeric(f, at_dn)
        exact = _numeric(dydx, at_0)
        if fp is None or fm is None or exact is None:
            continue
        approx = (fp - fm) / (2 * h)
        total += 1
        if abs(approx - exact) <= _DERIV_TOL * (1 + abs(exact)):
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


def _verify_complex(expr, op, val):
    """Re-check a complex-analysis answer against plain Python complex arithmetic.

    This one is genuinely independent: `cmath`/`complex` never touch SymPy's
    simplifier, so agreement is two separate implementations reaching the same
    number rather than the CAS confirming itself.
    """
    pyop = {
        "re": lambda z: complex(z.real, 0.0),
        "im": lambda z: complex(z.imag, 0.0),
        "conjugate": lambda z: z.conjugate(),
        "modulus": lambda z: complex(abs(z), 0.0),
        "argument": lambda z: complex(cmath.phase(z), 0.0),
    }[op]
    syms = sorted(expr.free_symbols, key=lambda s: s.name)
    agree = 0
    total = 0
    for k in range(len(_CSAMPLES)):
        subs = {s: _CSAMPLES[(k + j) % len(_CSAMPLES)] for j, s in enumerate(syms)}
        z = _numeric(expr, subs)
        got = _numeric(val, subs)
        if z is None or got is None:
            continue
        want = pyop(z)
        total += 1
        if abs(want - got) <= 1e-9 * (1 + abs(want)):
            agree += 1
    if total == 0:
        return None, "computed symbolically (numeric check not applicable)"
    if agree == total:
        return True, f"re-checked against Python complex arithmetic at {total} points"
    return False, f"numeric check disagreed ({agree}/{total} points matched)"


# --------------------------------------------------------------------------- #
#  Independent checks for operations that used to ship unverified.
#
#  "Independent" is the whole point. Asking SymPy the same question twice is not
#  a verification, so each of these reaches the answer by a different route:
#  hand-rolled Gaussian elimination for linear algebra, finite differences for
#  partials, plain-Python arithmetic over the raw data for statistics, and a
#  numeric contour integral for residues.
# --------------------------------------------------------------------------- #
def _mat_floats(M):
    """Matrix -> list-of-lists of Python complex, or None if not fully numeric."""
    try:
        out = []
        for i in range(M.rows):
            row = []
            for j in range(M.cols):
                v = _numeric(M[i, j], {})
                if v is None:
                    return None
                row.append(v)
            out.append(row)
        return out or None
    except Exception:
        return None


def _gauss_det(A):
    """Determinant by Gaussian elimination with partial pivoting."""
    n = len(A)
    a = [r[:] for r in A]
    det = complex(1)
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(a[r][c]))
        if abs(a[p][c]) < 1e-14:
            return complex(0)
        if p != c:
            a[c], a[p] = a[p], a[c]
            det = -det
        piv = a[c][c]
        det *= piv
        for r in range(c + 1, n):
            f = a[r][c] / piv
            for k in range(c, n):
                a[r][k] -= f * a[c][k]
    return det


def _gauss_rank(A):
    """Rank by row reduction with partial pivoting; tolerance scaled to the data."""
    a = [r[:] for r in A]
    rows, cols = len(a), len(a[0])
    biggest = max((abs(v) for r in a for v in r), default=0.0)
    tol = max(rows, cols) * 1e-12 * (biggest or 1.0)
    rank = r = 0
    for c in range(cols):
        if r >= rows:
            break
        p = max(range(r, rows), key=lambda k: abs(a[k][c]))
        if abs(a[p][c]) <= tol:
            continue
        a[r], a[p] = a[p], a[r]
        piv = a[r][c]
        for k in range(r + 1, rows):
            f = a[k][c] / piv
            for j in range(c, cols):
                a[k][j] -= f * a[r][j]
        r += 1
        rank += 1
    return rank


def _verify_determinant(M, val):
    A = _mat_floats(M)
    got = _numeric(val, {})
    if A is None or got is None or len(A) != len(A[0]):
        return None, "computed symbolically (numeric check not applicable)"
    ref = _gauss_det(A)
    if abs(ref - got) <= 1e-7 * (1 + abs(ref)):
        return True, "confirmed by independent Gaussian elimination"
    return (
        False,
        f"independent elimination gave {ref.real:.6g}, symbolic gave {got.real:.6g}",
    )


def _verify_trace(M, val):
    """Sum the diagonal directly rather than calling M.trace() again."""
    if M.rows != M.cols:
        return False, "trace is only defined for a square matrix"
    ref = sum((M[i, i] for i in range(M.rows)), S.Zero)
    if simplify(ref - val) == 0:
        return True, f"confirmed by summing the {M.rows} diagonal entries directly"
    return False, f"summing the diagonal gave {ref}, symbolic gave {val}"


def _verify_mode(data, modes, count):
    """Recount occurrences independently and confirm the reported set is exactly
    the set of values attaining the maximum."""
    tally = {}
    for v in data:
        tally[str(v)] = tally.get(str(v), 0) + 1
    if not tally:
        return None, "computed symbolically (numeric check not applicable)"
    top = max(tally.values())
    if top != count:
        return False, f"independent count found a max frequency of {top}, not {count}"
    want = {k for k, c in tally.items() if c == top}
    got = {str(v) for v in modes}
    if want != got:
        return False, "the reported values are not exactly those attaining the maximum"
    return True, f"confirmed by an independent count — appears {top} time(s)"


def _verify_rank(M, val):
    A = _mat_floats(M)
    if A is None:
        return None, "computed symbolically (numeric check not applicable)"
    ref = _gauss_rank(A)
    if ref == int(val):
        return True, f"confirmed by independent row reduction ({ref} pivots found)"
    return False, f"independent row reduction found rank {ref}, symbolic gave {val}"


def _verify_transpose(M, T):
    """Entry-by-entry: T[j][i] must equal M[i][j]. Checks the operation itself
    rather than re-running it."""
    if T.rows != M.cols or T.cols != M.rows:
        return False, "the result has the wrong shape for a transpose"
    for i in range(M.rows):
        for j in range(M.cols):
            if simplify(T[j, i] - M[i, j]) != 0:
                return False, f"entry ({j + 1},{i + 1}) does not match the original"
    return True, f"confirmed entry-by-entry across all {M.rows * M.cols} positions"


def _verify_rref(M, R):
    """Two checks SymPy's rref() doesn't do for us: that the result really is in
    reduced row-echelon form, and that row reduction preserved the rank (an
    independently computed one)."""
    A, B = _mat_floats(M), _mat_floats(R)
    if A is None or B is None:
        return None, "computed symbolically (numeric check not applicable)"
    eps = 1e-9
    last_pivot = -1
    seen_zero_row = False
    for row in B:
        piv = next((j for j, v in enumerate(row) if abs(v) > eps), None)
        if piv is None:
            seen_zero_row = True
            continue
        if seen_zero_row:
            return False, "a zero row appears above a nonzero row"
        if piv <= last_pivot:
            return False, "pivot columns are not strictly increasing"
        last_pivot = piv
        if abs(row[piv] - 1) > eps:
            return False, "a pivot entry is not 1"
        if any(abs(B[k][piv]) > eps for k in range(len(B)) if B[k] is not row):
            return False, "a pivot column has another nonzero entry"
    if _gauss_rank(A) != _gauss_rank(B):
        return False, "rank changed — row reduction did not preserve the row space"
    return True, "confirmed: valid reduced row-echelon form, rank preserved"


_PARTIAL_TOL = 1e-6
_SECOND_TOL = 1e-4


def _verify_gradient(f, syms, parts):
    """Finite-difference every partial derivative."""
    h = 1e-5
    pts = _samples_for(f)
    agree = total = 0
    for k in range(4):
        base = {s: pts[(k + i) % len(pts)] for i, s in enumerate(syms)}
        for i, s in enumerate(syms):
            up, dn = dict(base), dict(base)
            up[s], dn[s] = base[s] + h, base[s] - h
            fp, fm, ex = _numeric(f, up), _numeric(f, dn), _numeric(parts[i], base)
            if fp is None or fm is None or ex is None:
                continue
            total += 1
            if abs((fp - fm) / (2 * h) - ex) <= _PARTIAL_TOL * (1 + abs(ex)):
                agree += 1
    if total == 0:
        return None, "computed symbolically (numeric check not applicable)"
    if agree == total:
        return True, f"every partial confirmed by finite differences ({total} checks)"
    return False, f"numeric check disagreed ({agree}/{total} partials matched)"


def _verify_hessian(f, syms, H):
    """Finite-difference every second partial, including the mixed ones.
    Uses a larger step than the gradient: second differences divide by h twice
    and amplify floating-point noise accordingly."""
    h = 1e-3
    pts = _samples_for(f)
    agree = total = 0
    for k in range(3):
        base = {s: pts[(k + i) % len(pts)] for i, s in enumerate(syms)}
        f0 = _numeric(f, base)
        for i, si in enumerate(syms):
            for j, sj in enumerate(syms):

                def at(di, dj):
                    p = dict(base)
                    p[si] = p[si] + di * h
                    p[sj] = p[sj] + dj * h
                    return _numeric(f, p)

                if i == j:
                    fp, fm = at(1, 0), at(-1, 0)
                    if fp is None or fm is None or f0 is None:
                        continue
                    approx = (fp - 2 * f0 + fm) / (h * h)
                else:
                    pp, pm, mp, mm = at(1, 1), at(1, -1), at(-1, 1), at(-1, -1)
                    if pp is None or pm is None or mp is None or mm is None:
                        continue
                    approx = (pp - pm - mp + mm) / (4 * h * h)
                ex = _numeric(H[i, j], base)
                if ex is None:
                    continue
                total += 1
                if abs(approx - ex) <= _SECOND_TOL * (1 + abs(ex)):
                    agree += 1
    if total == 0:
        return None, "computed symbolically (numeric check not applicable)"
    if agree == total:
        return (
            True,
            f"every second partial confirmed by finite differences ({total} checks)",
        )
    return False, f"numeric check disagreed ({agree}/{total} entries matched)"


# Step size and tolerance per derivative order. Each differencing level divides
# by h again, so roundoff grows like h^-order: the step has to grow with order
# and the tolerance has to loosen with it. Measured, not guessed -- see
# test_solver.py's rejection suite for the wrong-answer side.
_NTH_DIFF = {2: (1e-3, 1e-4), 3: (1e-2, 1e-3), 4: (3e-2, 1e-2)}


def _verify_nth_derivative(f, var, dnf, order):
    """Repeated central differences for higher-order derivatives."""
    if order not in _NTH_DIFF:
        return None, f"order-{order} derivative computed symbolically"
    h, tol = _NTH_DIFF[order]
    agree = total = 0
    for x0 in _samples_for(f):
        vals = []
        for m in range(order + 1):
            v = _numeric(f, {var: x0 + (m - order / 2.0) * h})
            if v is None:
                break
            vals.append(v)
        if len(vals) != order + 1:
            continue
        ex = _numeric(dnf, {var: x0})
        if ex is None:
            continue
        approx = sum(
            (-1) ** m * math.comb(order, m) * vals[order - m] for m in range(order + 1)
        ) / (h**order)
        total += 1
        if abs(approx - ex) <= tol * (1 + abs(ex)):
            agree += 1
    if total == 0:
        return None, "computed symbolically (numeric check not applicable)"
    if agree == total:
        return True, f"numerically confirmed at {total} sample points"
    return False, f"numeric check disagreed ({agree}/{total} points matched)"


def _verify_stats(data, kind, val, sample=False):
    """Recompute from the raw values in plain Python floats.

    Independent two ways: float arithmetic rather than SymPy's exact rationals,
    and — for variance/std — the E[x^2] - E[x]^2 identity rather than the
    sum-of-squared-deviations formula the engine uses.
    """
    xs = []
    for v in data:
        f = _numeric(v, {})
        if f is None:
            return None, "computed symbolically (numeric check not applicable)"
        xs.append(f.real)
    got = _numeric(val, {})
    if got is None or not xs:
        return None, "computed symbolically (numeric check not applicable)"
    n = len(xs)
    if kind == "median":
        s = sorted(xs)
        ref = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0
        note = "confirmed by an independent sort"
    else:
        mean = sum(xs) / n
        pop = sum(x * x for x in xs) / n - mean * mean
        var = pop * n / (n - 1) if (sample and n > 1) else pop
        var = max(var, 0.0)
        ref = var if kind == "variance" else var**0.5
        note = "confirmed by an independent formula (E[x²] − E[x]²)"
    if abs(ref - got.real) <= 1e-6 * (1 + abs(ref)):
        return True, note
    return (
        False,
        f"independent computation gave {ref:.6g}, symbolic gave {got.real:.6g}",
    )


# SI magnitudes, entered by hand from the defining relations: 1 inch = 0.0254 m
# exactly, 1 lb = 0.45359237 kg exactly, 1 US gallon = 231 in^3 exactly. Typed
# out on purpose -- this is the INDEPENDENT check on SymPy's unit table, and
# re-reading SymPy's own factors would verify precisely nothing.
_SI_FACTOR = {
    "meter": (1.0, "length"),
    "kilometer": (1000.0, "length"),
    "centimeter": (0.01, "length"),
    "millimeter": (0.001, "length"),
    "inch": (0.0254, "length"),
    "foot": (0.3048, "length"),
    "yard": (0.9144, "length"),
    "mile": (1609.344, "length"),
    "kilogram": (1.0, "mass"),
    "gram": (0.001, "mass"),
    "pound": (0.45359237, "mass"),
    "second": (1.0, "time"),
    "minute": (60.0, "time"),
    "hour": (3600.0, "time"),
    "day": (86400.0, "time"),
    "liter": (0.001, "volume"),
    "gallon": (0.003785411784, "volume"),
}


def _verify_convert(qty, src_name, dst_name, coeff):
    """Recompute the conversion from hand-entered SI magnitudes."""
    a = _SI_FACTOR.get(_UNIT_MAP.get(src_name.lower().strip(), ""))
    b = _SI_FACTOR.get(_UNIT_MAP.get(dst_name.lower().strip(), ""))
    q = _numeric(qty, {})
    got = _numeric(coeff, {})
    if not a or not b or q is None or got is None:
        return None, "computed symbolically (numeric check not applicable)"
    if a[1] != b[1]:
        return False, f"those units measure different things ({a[1]} vs {b[1]})"
    ref = q.real * a[0] / b[0]
    if abs(ref - got.real) <= 1e-9 * (1 + abs(ref)):
        return True, "confirmed against an independent SI conversion table"
    return (
        False,
        f"independent table gave {ref:.6g}, symbolic gave {got.real:.6g}",
    )


def _verify_stats_pair(data, kind, pop_val, samp_val):
    """Variance and standard deviation are reported in both flavours, so both
    have to clear the check — verifying only the population figure would leave
    the sample figure (the /(n-1) one, where an off-by-one actually hides)
    completely unchecked."""
    v1, n1 = _verify_stats(data, kind, pop_val, sample=False)
    if samp_val is None:
        return v1, n1
    v2, n2 = _verify_stats(data, kind, samp_val, sample=True)
    if v1 is True and v2 is True:
        return True, n1 + ", for both population and sample"
    if v1 is False or v2 is False:
        return False, (n1 if v1 is False else n2)
    return None, "computed symbolically (numeric check not applicable)"


def _verify_residue(expr, z, pt, val):
    """Numeric contour integral: Res = (1/2πi) ∮ f dz around a small circle.

    That's the definition, evaluated directly — independent of the series
    expansion SymPy's residue() uses to get there.
    """
    c = _numeric(pt, {})
    want = _numeric(val, {})
    if c is None or want is None:
        return None, "computed symbolically (numeric check not applicable)"
    for radius in (0.1, 0.03):
        steps = 512
        acc = complex(0)
        ok = True
        for k in range(steps):
            th = 2 * cmath.pi * (k + 0.5) / steps
            e = cmath.exp(1j * th)
            fz = _numeric(expr, {z: c + radius * e})
            if fz is None:
                ok = False
                break
            acc += fz * (1j * radius * e) * (2 * cmath.pi / steps)
        if not ok:
            continue
        got = acc / (2j * cmath.pi)
        if abs(got - want) <= 1e-4 * (1 + abs(want)):
            return True, "confirmed by numeric contour integration"
        return (
            False,
            f"contour integration gave {got.real:.6g}, symbolic gave {want.real:.6g}",
        )
    return None, "computed symbolically (numeric check not applicable)"


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
        else _verify_nth_derivative(expr, var, d_s, order)
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


# Multi-letter tokens that legitimately appear inside an expression. Anything
# else with two or more letters in a row is prose, not math.
_KNOWN_TOKENS = (
    set(_FUNCS.split("|"))
    | {long for long, _ in _ALIASES}
    | {short for _, short in _ALIASES}
    | {
        "pi",
        "oo",
        "inf",
        "infinity",
        "infty",
        "gamma",
        "abs",
        "sqrt",
        "cbrt",
        "root",
        "factorial",
        "binomial",
        "floor",
        "ceiling",
        "ceil",
        "sign",
        "re",
        "im",
        "conjugate",
        "arg",
        "max",
        "min",
        "mod",
        "lcm",
        "gcd",
        "log",
        "ln",
        "exp",
        "atan2",
        "zoo",
        "nan",
        "eye",
        "diag",
        "det",
    }
)


def _prose_tokens(s):
    """Letter runs in `s` that aren't known math identifiers."""
    return [w for w in re.findall(r"[A-Za-z]{2,}", s) if w.lower() not in _KNOWN_TOKENS]


def _do_expression(expr_str):
    """Bare expression: evaluate if numeric, else show + quick calculus facts."""
    # The implicit-multiplication parser happily turns any unrecognized English
    # into a product of single-letter symbols -- "mean of 5" parsed as
    # 5*E*a*f*m*n*o and was displayed as a confident, badged answer. Refusing is
    # strictly better than fabricating: a wrong answer under a trust badge is
    # worse than no answer at all.
    stray = _prose_tokens(expr_str)
    if stray:
        raise UserError(
            "I couldn't read that as math — I don't recognize "
            + ", ".join(f"'{w}'" for w in dict.fromkeys(stray[:3]))
            + ". Try an expression like `derivative of x^2 sin(x)`."
        )
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
    # Boundary values stay fixed on purpose: 0 and +/-5 are exactly where an
    # inequality's solution set is most likely to be wrong.
    for x0 in _samples_for(rel) + [0.0, 5.0, -5.0, 3.1415, -2.5, 10.0, -10.0]:
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
        raise UserError("That doesn't look like a matrix. Try `[[1,2],[3,4]]`.")
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
        verified, note = _verify_determinant(M, val)
        return _result(
            type="determinant",
            input_latex=in_tex,
            answer_latex=rf"\det = {latex(val)}",
            answer_str=str(val),
            approx=_num_approx(val),
            verified=verified,
            verify_note=note,
        )
    if op == "trace":
        val = simplify(M.trace())
        verified, note = _verify_trace(M, val)
        return _result(
            type="trace",
            input_latex=in_tex,
            answer_latex=rf"\operatorname{{tr}} = {latex(val)}",
            answer_str=str(val),
            approx=_num_approx(val),
            verified=verified,
            verify_note=note,
        )
    if op == "rank":
        val = M.rank()
        verified, note = _verify_rank(M, val)
        return _result(
            type="rank",
            input_latex=in_tex,
            answer_latex=rf"\operatorname{{rank}} = {val}",
            answer_str=str(val),
            verified=verified,
            verify_note=note,
        )
    if op == "transpose":
        T = M.T
        verified, note = _verify_transpose(M, T)
        return _result(
            type="transpose",
            input_latex=in_tex,
            answer_latex=latex(T),
            answer_str=str(T),
            verified=verified,
            verify_note=note,
        )
    if op in ("rref", "rowreduce"):
        R = M.rref()[0]
        verified, note = _verify_rref(M, R)
        return _result(
            type="rref",
            input_latex=in_tex,
            answer_latex=latex(R),
            answer_str=str(R),
            verified=verified,
            verify_note=note,
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


def _miller_rabin(n):
    """Deterministic Miller-Rabin (first 12 primes as bases; exact for n < 3.3e24).

    Written out longhand on purpose: this is the *independent* check on
    sp.isprime, so it must not call back into SymPy. Two separate algorithms
    agreeing is a verification; asking SymPy twice is not.
    """
    if n < 2:
        return False
    bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in bases:
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in bases:
        y = pow(a, d, n)
        if y == 1 or y == n - 1:
            continue
        for _ in range(r - 1):
            y = y * y % n
            if y == n - 1:
                break
        else:
            return False
    return True


def _do_isprime(n):
    n = int(n)
    prime = sp.isprime(n)
    independent = _miller_rabin(n)
    if prime != independent:  # should be unreachable; surfaced, not hidden
        return _result(
            type="primality",
            input_latex=str(n),
            answer_latex=(
                rf"{n}\ \text{{is prime}}" if prime else rf"{n}\ \text{{is not prime}}"
            ),
            answer_str=f"{n} is {'prime' if prime else 'not prime'}",
            verified=False,
            verify_note="two independent primality tests disagreed — do not trust this",
        )
    if prime:
        return _result(
            type="primality",
            input_latex=str(n),
            answer_latex=rf"{n}\ \text{{is prime}}",
            answer_str=f"{n} is prime",
            verified=True,
            verify_note="confirmed by an independent Miller-Rabin test",
        )
    f = sp.factorint(n)
    sm = min(f) if f else None
    # A composite claim is verifiable outright: exhibit a divisor and check it.
    witness = sm is not None and 1 < sm < n and n % sm == 0
    return _result(
        type="primality",
        input_latex=str(n),
        answer_latex=rf"{n}\ \text{{is not prime}}",
        answer_str=f"{n} is not prime",
        verified=True if witness else None,
        verify_note=(
            f"confirmed: {n} = {sm} x {n // sm}"
            if witness
            else "computed symbolically (no divisor witness available)"
        ),
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
        med = _median(srt)
        v, note = _verify_stats(data, "median", med)
        return _res("median", med, note, v)
    if op == "mode":
        from collections import Counter

        c = Counter(str(v) for v in data)
        top = max(c.values())
        seen = []
        for v in data:
            if c[str(v)] == top and v not in seen:
                seen.append(v)
        mode_tex = ",\\ ".join(latex(v) for v in seen)
        verified, note = _verify_mode(data, seen, top)
        return _result(
            type="mode",
            input_latex=data_tex,
            answer_latex=rf"\text{{mode}} = {mode_tex}",
            answer_str=", ".join(str(v) for v in seen),
            verified=verified,
            verify_note=note,
        )

    ss = sum((simplify(v - mean)) ** 2 for v in data)
    pop_var = simplify(ss / n)
    samp_var = simplify(ss / (n - 1)) if n > 1 else None
    if op in ("variance", "var"):
        v, note = _verify_stats_pair(data, "variance", pop_var, samp_var)
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
            verified=v,
            verify_note=note,
        )
    if op in ("standarddeviation", "std", "sd", "stdev"):
        pop_sd = simplify(sp.sqrt(pop_var))
        samp_sd = simplify(sp.sqrt(samp_var)) if samp_var is not None else None
        v, note = _verify_stats_pair(data, "std", pop_sd, samp_sd)
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
            verified=v,
            verify_note=note,
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
    verified, note = _verify_gradient(expr, syms, parts)
    return _result(
        type="gradient",
        input_latex=latex(expr),
        answer_latex=rf"\nabla f = {latex(grad.T)}",
        answer_str=str(grad.T),
        steps=[
            rf"\frac{{\partial f}}{{\partial {latex(s)}}} = {latex(p)}"
            for s, p in zip(syms, parts)
        ],
        verified=verified,
        verify_note=note,
    )


def _do_hessian(expr_str):
    expr = _P(expr_str)
    syms = sorted(expr.free_symbols, key=lambda s: s.name) or [Symbol("x")]
    H = sp.hessian(expr, syms)
    verified, note = _verify_hessian(expr, syms, H)
    return _result(
        type="hessian",
        input_latex=latex(expr),
        answer_latex=latex(H),
        answer_str=str(H),
        verified=verified,
        verify_note=note,
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
#  Polar & parametric plots (Tier-2 scope; x,y samples reuse the Plotly renderer)
# --------------------------------------------------------------------------- #
def _do_polar(expr_str):
    import math

    s = re.sub(r"\b(theta|θ)\b", "t", expr_str)
    s = re.sub(r"^\s*r\s*=\s*", "", s.strip())
    expr = _P(s)
    t = _pick_var(expr, prefer="t")
    n = 400
    xs, ys = [], []
    for i in range(n + 1):
        ang = 2 * math.pi * i / n
        r = _numeric(expr, {t: ang})
        if r is None or abs(r.imag) > 1e-6:
            xs.append(None)
            ys.append(None)
            continue
        xs.append(r.real * math.cos(ang))
        ys.append(r.real * math.sin(ang))
    if all(v is None for v in xs):
        return {"ok": False, "error": "Couldn't sample that polar curve."}
    return _result(
        type="polar plot",
        input_latex=rf"r = {latex(expr)}",
        answer_latex=rf"r = {latex(expr)}",
        answer_str=f"polar: r = {expr}",
        verify_note="polar curve, θ from 0 to 2π",
        plot={
            "var": "theta",
            "equal_aspect": True,
            "traces": [{"label": "r", "x": xs, "y": ys}],
        },
    )


def _do_parametric(body):
    import math

    lo, hi = 0.0, 2 * math.pi
    mr = re.search(r"\s+for\s+t\s*=\s*(.+?)\s+to\s+(.+)$", body)
    if mr:
        lo = float(N(_P(mr.group(1))))
        hi = float(N(_P(mr.group(2))))
        body = body[: mr.start()]
    mx = re.search(r"x\s*=\s*(.+?)\s*,\s*y\s*=\s*(.+)$", body.strip())
    if not mx:
        return {
            "ok": False,
            "error": "Use 'x = f(t), y = g(t)' (optionally 'for t=0 to 2*pi').",
        }
    xe, ye = _P(mx.group(1)), _P(mx.group(2))
    t = Symbol("t")
    n = 400
    xs, ys = [], []
    for i in range(n + 1):
        tv = lo + (hi - lo) * i / n
        xv, yv = _numeric(xe, {t: tv}), _numeric(ye, {t: tv})
        if xv is None or yv is None or abs(xv.imag) > 1e-6 or abs(yv.imag) > 1e-6:
            xs.append(None)
            ys.append(None)
            continue
        xs.append(xv.real)
        ys.append(yv.real)
    if all(v is None for v in xs):
        return {"ok": False, "error": "Couldn't sample that parametric curve."}
    return _result(
        type="parametric plot",
        input_latex=rf"x = {latex(xe)},\ y = {latex(ye)}",
        answer_latex=rf"x = {latex(xe)},\quad y = {latex(ye)}",
        answer_str="parametric curve",
        verify_note="parametric curve",
        plot={
            "var": "t",
            "equal_aspect": True,
            "traces": [{"label": "curve", "x": xs, "y": ys}],
        },
    )


# --------------------------------------------------------------------------- #
#  Differential equations (Tier-3 scope)
# --------------------------------------------------------------------------- #
def _ode_prep(s, dep, ind):
    """Rewrite y'' / y' / y into Derivative(y(x),x,2) / Derivative(y(x),x) / y(x)."""
    s = re.sub(rf"\b{dep}\s*''", f"Derivative({dep}({ind}),{ind},2)", s)
    s = re.sub(rf"\b{dep}\s*'", f"Derivative({dep}({ind}),{ind})", s)
    s = re.sub(rf"\b{dep}\b(?!\s*\()", f"{dep}({ind})", s)  # standalone y -> y(x)
    return s


def _do_ode(eq_str, dep="y", ind="x"):
    y = sp.Function(dep)
    x = Symbol(ind)
    loc = dict(_LOCALS)
    loc.update({dep: y, ind: x, "Derivative": Derivative})
    prepped = _ode_prep(eq_str, dep, ind)

    def _p(part):
        return parse_expr(part, transformations=_TRANSFORMS, local_dict=loc)

    if "=" in prepped:
        lhs_s, _, rhs_s = prepped.partition("=")
        eq = Eq(_p(lhs_s), _p(rhs_s))
    else:
        eq = Eq(_p(prepped), S.Zero)
    sol = sp.dsolve(eq, y(x))
    sols = sol if isinstance(sol, list) else [sol]
    try:
        chk = sp.checkodesol(eq, sol)
        verified = all(c[0] for c in chk) if isinstance(chk, list) else bool(chk[0])
    except Exception:
        verified = None
    ans = r"\\".join(latex(s) for s in sols)
    if len(sols) > 1:
        ans = r"\begin{gathered}" + ans + r"\end{gathered}"
    return _result(
        type="differential equation",
        input_latex=latex(eq),
        answer_latex=ans,
        answer_str="; ".join(str(s) for s in sols),
        steps=[rf"\text{{Solve the ODE}}\ {latex(eq)}"],
        verified=verified,
        verify_note=(
            "confirmed: the solution satisfies the ODE"
            if verified
            else (
                "the solution did not check out"
                if verified is False
                else "computed symbolically"
            )
        ),
    )


# --------------------------------------------------------------------------- #
#  Complex analysis, vector calculus, units (Tier-3 scope)
# --------------------------------------------------------------------------- #
def _do_complex(op, expr_str):
    expr = _P(expr_str, locals_=_COMPLEX_LOCALS)
    fn = {
        "re": sp.re,
        "im": sp.im,
        "conjugate": sp.conjugate,
        "modulus": Abs,
        "argument": sp.arg,
    }[op]
    val = simplify(fn(expr))
    label = {
        "re": "Re",
        "im": "Im",
        "conjugate": "conjugate",
        "modulus": "modulus",
        "argument": "arg",
    }[op]
    verified, note = _verify_complex(expr, op, val)
    return _result(
        type=f"complex ({label})",
        input_latex=latex(expr),
        answer_latex=latex(val),
        answer_str=str(val),
        approx=_num_approx(val),
        verified=verified,
        verify_note=note,
    )


def _do_residue(expr_str, var_name, point_str):
    z = Symbol(var_name)
    expr = _P(expr_str, locals_=_COMPLEX_LOCALS)
    pt = _P(point_str, locals_=_COMPLEX_LOCALS)
    val = simplify(sp.residue(expr, z, pt))
    verified, note = _verify_residue(expr, z, pt, val)
    return _result(
        type="residue",
        input_latex=latex(expr),
        answer_latex=rf"\operatorname{{Res}} = {latex(val)}",
        answer_str=str(val),
        approx=_num_approx(val),
        verified=verified,
        verify_note=f"residue at {var_name} = {point_str} — {note}",
    )


def _parse_vector(s):
    s = s.strip().strip("()[]{}").strip()
    return [_P(e) for e in s.split(",") if e.strip()]


def _do_vectorcalc(op, field_str):
    from sympy.vector import CoordSys3D, curl, divergence

    comps = _parse_vector(field_str)
    if len(comps) != 3:
        return {"ok": False, "error": "Give a 3-component field, e.g. (x^2, y^2, z^2)."}
    Cs = CoordSys3D("C")
    x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
    to_C = {x: Cs.x, y: Cs.y, z: Cs.z}
    from_C = {Cs.x: x, Cs.y: y, Cs.z: z}
    F = (
        comps[0].subs(to_C) * Cs.i
        + comps[1].subs(to_C) * Cs.j
        + comps[2].subs(to_C) * Cs.k
    )
    in_tex = rf"\langle {', '.join(latex(c) for c in comps)} \rangle"
    if op == "divergence":
        val = simplify(divergence(F).subs(from_C))
        return _result(
            type="divergence",
            input_latex=in_tex,
            answer_latex=rf"\nabla \cdot F = {latex(val)}",
            answer_str=str(val),
            verify_note="divergence of the field",
        )
    cv = curl(F)
    out = sp.Matrix([simplify(cv.dot(b).subs(from_C)) for b in (Cs.i, Cs.j, Cs.k)])
    return _result(
        type="curl",
        input_latex=in_tex,
        answer_latex=rf"\nabla \times F = {latex(out.T)}",
        answer_str=str(out.T),
        verify_note="curl of the field",
    )


_UNIT_MAP = {
    "km": "kilometer",
    "kilometer": "kilometer",
    "kilometers": "kilometer",
    "m": "meter",
    "meter": "meter",
    "meters": "meter",
    "cm": "centimeter",
    "mm": "millimeter",
    "mile": "mile",
    "miles": "mile",
    "mi": "mile",
    "ft": "foot",
    "foot": "foot",
    "feet": "foot",
    "inch": "inch",
    "inches": "inch",
    "yard": "yard",
    "yd": "yard",
    "kg": "kilogram",
    "g": "gram",
    "gram": "gram",
    "grams": "gram",
    "lb": "pound",
    "pound": "pound",
    "pounds": "pound",
    "s": "second",
    "second": "second",
    "seconds": "second",
    "min": "minute",
    "minute": "minute",
    "minutes": "minute",
    "hour": "hour",
    "hours": "hour",
    "h": "hour",
    "day": "day",
    "days": "day",
    "liter": "liter",
    "liters": "liter",
    "l": "liter",
    "gallon": "gallon",
    "gallons": "gallon",
}


def _do_convert(qty_str, unit_str, to_unit_str):
    from sympy.physics import units as U
    from sympy.physics.units import convert_to

    def _u(name):
        key = _UNIT_MAP.get(name.lower().strip())
        if not key:
            raise UserError(f"I don't know the unit '{name}'.")
        return getattr(U, key)

    qty = _P(qty_str)
    src, dst = _u(unit_str), _u(to_unit_str)
    val = convert_to(qty * src, dst)
    coeff = simplify(val / dst)
    verified, note = _verify_convert(qty, unit_str, to_unit_str, coeff)
    return _result(
        type="unit conversion",
        input_latex=rf"{latex(qty)}\ \text{{{unit_str}}}",
        answer_latex=latex(val),
        answer_str=str(val),
        approx=_num_approx(coeff),
        verified=verified,
        verify_note=note,
    )


# --------------------------------------------------------------------------- #
#  3D surface plots (Tier-3 scope; a z-grid the Plotly renderer draws)
# --------------------------------------------------------------------------- #
def _do_surface(expr_str):
    s = re.sub(r"^\s*z\s*=\s*", "", expr_str.strip())
    expr = _P(s)
    syms = sorted(expr.free_symbols, key=lambda s: s.name)
    vx = next((v for v in syms if v.name == "x"), None) or (
        syms[0] if syms else Symbol("x")
    )
    vy = next((v for v in syms if v.name == "y"), None) or (
        next((v for v in syms if v is not vx), None) or Symbol("y")
    )
    lo, hi, n = -4.0, 4.0, 30
    axis = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    zgrid = []
    for yv in axis:
        row = []
        for xv in axis:
            z = _numeric(expr, {vx: xv, vy: yv})
            row.append(
                z.real
                if (z is not None and abs(z.imag) < 1e-6 and abs(z.real) < 1e6)
                else None
            )
        zgrid.append(row)
    return _result(
        type="surface plot",
        input_latex=rf"z = {latex(expr)}",
        answer_latex=rf"z = {latex(expr)}",
        answer_str=f"surface: z = {expr}",
        verify_note="surface z = f(x, y)",
        plot={
            "surface": True,
            "x": axis,
            "y": axis,
            "z": zgrid,
            "xlabel": vx.name,
            "ylabel": vy.name,
        },
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


# Answers that are correct internally and cryptic externally. "oo" and
# "AccumBounds(-1, 1)" are exactly right as SymPy objects and mean nothing to a
# student, so each gets a plain-English rider on the note line.
_SPECIAL_NOTES = [
    (
        r"AccumBounds",
        "the value oscillates between those bounds and never settles, so the limit does not exist",
    ),
    (r"\bzoo\b", "unbounded in the complex sense — the expression diverges"),
    (r"\bnan\b", "undefined — there is no well-defined value here"),
    (r"-\s*oo\b", "diverges to negative infinity"),
    (r"\boo\b", "diverges to infinity"),
]


# Notation where the parse is defensible but probably isn't what was meant. The
# verifier confirms SymPy operated correctly on *what it parsed* -- it cannot
# confirm that the parse matched the user's intent. These are the cases where
# that gap bites, so the reading gets flagged instead of silently badged.
_READING_RISKS = [
    (
        rf"\b(?:{_FUNCS})\s+[A-Za-z]\w*\s*(?:\*|\s)\s*[A-Za-z0-9(]",
        "a function written without parentheses swallowed the whole product — "
        "`sin x cos x` reads as sin(x·cos x). Write `sin(x)*cos(x)` to be sure.",
    ),
    # No whitespace allowed before the trailing letter: "x^2 from 0 to 3" and
    # "tangent to x^2 at x=1" are unambiguous, and flagging them would train
    # people to ignore the warning.
    (
        r"\^\s*-?\d+[A-Za-z]",
        "an exponent runs straight into a variable — `e^2x` reads as (e²)·x, "
        "not e^(2x). Add parentheses to be sure.",
    ),
    (
        r"[\dA-Za-z]\s*/\s*\d+[A-Za-z]",
        "a fraction runs straight into a variable — `1/2x` reads as (1/2)·x, "
        "not 1/(2x). Add parentheses to be sure.",
    ),
]


def _reading_risk(raw):
    """Return a warning when the input uses notation that parses ambiguously."""
    # Drop a trailing integration differential -- `integral of sin x dx` is not
    # an ambiguous juxtaposition, and _P strips the `dx` before parsing anyway.
    raw = re.sub(r"\s+d\s*[a-z]\s*$", "", raw)
    for pattern, message in _READING_RISKS:
        if re.search(pattern, raw):
            return message
    return None


def _explain_special(res):
    """Attach a plain-English reading to divergent / undefined / oscillating answers."""
    ans = str(res.get("answer_str", "")) + " " + str(res.get("answer_latex", ""))
    for pattern, phrase in _SPECIAL_NOTES:
        if re.search(pattern, ans):
            note = res.get("verify_note") or ""
            res["verify_note"] = phrase + (f" — {note}" if note else "")
            break
    return res


def solve(query):
    """Public entry point. Returns a result dict; never raises."""
    res = _solve(query)
    if not (isinstance(res, dict) and res.get("ok")):
        return res
    risk = _reading_risk(_clean(preprocess(query or "")))
    if risk:
        res["reading_risk"] = risk
    return _explain_special(res)


def _solve(query):
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
        # Index-preserving lowercase. The router matches its regexes against `ql`
        # and then slices the original-case `q` using match spans, so the two
        # must stay in 1:1 character correspondence. Plain str.lower() expands a
        # few Unicode codepoints (U+0130 -> two chars), which would shift every
        # index after it; skipping those keeps the mapping exact.
        ql = "".join(c.lower() if len(c.lower()) == 1 else c for c in q)

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
            return _do_derivative(q[m.start(2) :], order=o)
        m = re.search(r"^d\^?(\d+)\s*/\s*d\s*([a-z])\^?\d*\s*(?:of\s+)?(.+)$", ql)
        if m:
            return _do_derivative(
                q[m.start(3) :],
                var_name=m.group(2),
                order=int(m.group(1)),
            )

        # partial derivative
        m = re.search(
            r"partial\s+derivative\s+of\s+(.+?)\s+with\s+respect\s+to\s+([a-z])\b", ql
        )
        if m:
            return _do_derivative(
                q[m.start(1) : m.end(1)],
                var_name=m.group(2),
                order=1,
            )

        # first derivative:  derivative of / differentiate / d/dx
        m = re.search(r"^(?:the\s+)?derivative\s+of\s+(.+)$", ql) or re.search(
            r"^differentiate\s+(.+)$", ql
        )
        if m:
            body = q[m.start(1) :]
            vn = None
            mv = re.search(r"\s+with\s+respect\s+to\s+([a-z])\b", body.lower())
            if mv:
                vn = mv.group(1)
                body = body[: body.lower().find("with respect to")]
            return _do_derivative(body, var_name=vn)
        m = re.search(r"^d\s*/\s*d\s*([a-z])\s*(?:of\s+)?\(?(.+?)\)?$", ql)
        if m:
            return _do_derivative(q[m.start(2) :], var_name=m.group(1))

        # 3D surface plot: z = f(x, y)
        m = re.search(
            r"^(?:3d\s+plot|surface\s+plot)\s+(?:of\s+)?(.+)$", ql
        ) or re.search(r"^plot\s+z\s*=\s*(.+)$", ql)
        if m:
            return _do_surface(q[m.start(1) :])

        # polar / parametric plots
        m = re.search(r"^polar\s+plot\s+(?:of\s+)?(.+)$", ql)
        if m:
            return _do_polar(q[m.start(1) :])
        m = re.search(r"^parametric\s+plot\s+(?:of\s+)?(.+)$", ql)
        if m:
            return _do_parametric(q[m.start(1) :])

        # gradient / hessian (multivariable)
        m = re.search(r"^gradient\s+(?:of\s+)?(.+)$", ql)
        if m:
            return _do_gradient(q[m.start(1) :])
        m = re.search(r"^hessian\s+(?:of\s+)?(.+)$", ql)
        if m:
            return _do_hessian(q[m.start(1) :])

        # double / triple integral over explicit bounds
        m = re.search(
            r"^(?:double|triple)\s+integral\s+of\s+(.+?)\s+(?:over|for|with)\s+(.+)$",
            ql,
        )
        if m:
            body = q[m.start(1) : m.end(1)]
            bnds = q[m.start(2) :]
            return _do_multi_integral(body, bnds)

        # definite integral: "... from A to B"
        m = re.search(
            r"(?:integrate|integral\s+of|definite\s+integral\s+of|antiderivative\s+of|area\s+under(?:\s+the\s+curve)?)\s+(.+?)\s+from\s+(.+?)\s+to\s+(.+?)$",
            ql,
        )
        if m:
            body = q[m.start(1) : m.end(1)]
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
            body = q[m.start(1) :]
            vn = _integ_var(body)
            return _do_integral(_strip_dv(body), var_name=vn)

        # limit
        m = re.search(
            r"(?:limit|lim)\s+(?:of\s+)?(.+?)\s+as\s+([a-z])\s*(?:->|to|approaches)\s*(.+?)(?:\s+from\s+(?:the\s+)?(left|right|below|above))?$",
            ql,
        )
        if m:
            body = q[m.start(1) : m.end(1)]
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
            body_idx = 2 if m.re.groups >= 2 and m.lastindex >= 2 else 1
            body = q[m.start(body_idx) :]
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
            body = q[m.start(2) : m.end(2)]
            return _do_summation(
                body, m.group(3), m.group(4), m.group(5), is_product=is_prod
            )

        # tangent line
        m = re.search(
            r"tangent\s+(?:line\s+)?to\s+(.+?)\s+at\s+(?:[a-z]\s*=\s*)?(.+?)$", ql
        )
        if m:
            body = q[m.start(1) : m.end(1)]
            vn = _pick_var(_P(body)).name
            return _do_tangent(body, vn, m.group(2).strip())

        # critical points / extrema
        m = re.search(
            r"(?:critical\s+points?|extrema|local\s+(?:max|min)(?:ima|imum)?)\s+of\s+(.+)$",
            ql,
        )
        if m:
            return _do_critical(q[m.start(1) :])

        # differential equation: solve an ODE (contains y', y'', f', ...)
        if re.search(r"\bsolve\b", ql) and re.search(r"[a-z]\s*'", q):
            dep = re.search(r"([a-z])\s*'", q).group(1)
            ind = (
                "t" if (re.search(r"\bt\b", q) and not re.search(r"\bx\b", q)) else "x"
            )
            body = re.sub(r"^\s*solve\s+", "", q, flags=re.IGNORECASE)
            return _do_ode(body, dep=dep, ind=ind)

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
            body = q[m.start(1) : m.end(1)]
            parts = re.split(r"\s*,\s*|\s+and\s+", body)
            if len(parts) >= 2 and sum(1 for p in parts if "=" in p) >= 2:
                return _do_system(parts)
            return _do_solve(body, var_name=m.group(2))
        m = re.search(r"^(?:roots|zeros|zeroes)\s+of\s+(.+)$", ql)
        if m:
            return _do_solve(q[m.start(1) :])

        # complex analysis
        m = re.search(
            r"^(real\s*part|imaginary\s*part|conjugate|modulus|argument)\s+(?:of\s+)?(.+)$",
            ql,
        )
        if m:
            cop = {
                "real part": "re",
                "imaginary part": "im",
                "conjugate": "conjugate",
                "modulus": "modulus",
                "argument": "argument",
            }[re.sub(r"\s+", " ", m.group(1))]
            return _do_complex(cop, q[m.start(2) :])
        m = re.search(r"^residue\s+(?:of\s+)?(.+?)\s+at\s+([a-z])\s*=\s*(.+)$", ql)
        if m:
            body = q[m.start(1) : m.end(1)]
            pt = q[
                q.rfind("=") + 1 :
            ].strip()  # original case (I is the imaginary unit)
            return _do_residue(body, m.group(2), pt)

        # vector calculus
        m = re.search(r"^(divergence|curl)\s+(?:of\s+)?(.+)$", ql)
        if m:
            return _do_vectorcalc(m.group(1), q[m.start(2) :])

        # unit conversion
        m = re.search(r"^convert\s+(.+?)\s+([a-z]+)\s+to\s+([a-z]+)\s*$", ql)
        if m:
            return _do_convert(m.group(1), m.group(2), m.group(3))

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
        # `\s+` OR a following "(" — people type `simplify(x^2-1)` as readily as
        # `simplify x^2-1`, and without the lookahead the paren form fell through
        # to the bare-expression fallback and was refused as unreadable prose.
        m = re.search(
            r"^(simplify|factor|expand)(?:\s+|(?=\())\s*(?:trig\s+)?(.+)$", ql
        )
        if m:
            return _do_algebra(m.group(1), q[m.start(2) :])

        # bare expression fallback
        return _do_expression(q)

    except UserError as e:
        return {"ok": False, "error": str(e)}
    except RecursionError:
        return {"ok": False, "error": "That expression is too deeply nested to parse."}
    except Exception:
        # Deliberately not surfacing the exception text -- see UserError.
        return {"ok": False, "error": _GENERIC_PARSE_ERROR}


def _integ_var(body):
    m = re.search(r"\bd\s*([a-z])\b\s*$", body.strip())
    return m.group(1) if m else None


def _strip_dv(body):
    return re.sub(r"\s*d\s*[a-z]\s*$", "", body.strip())


# --------------------------------------------------------------------------- #
#  "Check my work" — verify a solution the USER wrote, line by line.
#
#  This is the one thing a CAS is structurally better at than any chat model, and
#  it inverts the product: instead of handing over an answer, it confirms your
#  own reasoning and points at the first line that breaks. It needs no API key
#  and no network — the whole point is that the most useful mode is the free one.
#
#  Two kinds of chain, detected automatically:
#    * EXPRESSION chain (x^2+2x, x(x+2), ...) — consecutive lines must be
#      algebraically equal.
#    * EQUATION chain (2x+6=10, 2x=4, x=2) — consecutive lines must have the
#      same solution set. That is the right invariant for solving: `2x=4` and
#      `x=2` are not equal as equations, but they are equivalent as problems.
#      Checking equality here would flag every correct solution.
# --------------------------------------------------------------------------- #
def _line_pairs(raw):
    """Split pasted work into comparable lines, dropping decoration."""
    out = []
    for ln in str(raw or "").replace("\r", "").split("\n"):
        s = ln.strip()
        s = re.sub(r"^(?:step\s*\d+\s*[:.)]|\d+\s*[.)])\s*", "", s, flags=re.I)
        s = s.lstrip("=").strip()  # leading "= x^3/3" continuation style
        s = re.sub(r"\+\s*C\b\s*$", "", s).strip()  # constant of integration
        if s and not re.fullmatch(r"[-–—_=\s]+", s):
            out.append(s)
    return out


def _solset(expr_str, var):
    """Solution set of an equation, as a canonical sorted tuple of strings."""
    lhs, rhs = expr_str.split("=", 1)
    eq = Eq(_P(lhs), _P(rhs))
    sols = sp_solve(eq, var, dict=False)
    if not isinstance(sols, (list, tuple)):
        sols = [sols]
    return tuple(sorted(str(simplify(s)) for s in sols))


def check_work(raw):
    """Verify a user's own solution chain and locate the first broken step."""
    lines = _line_pairs(raw)
    if len(lines) < 2:
        raise UserError(
            "Paste at least two lines of your own work — each line one step, e.g.\n"
            "2x + 6 = 10\n2x = 4\nx = 2"
        )
    is_eq = all("=" in ln for ln in lines)
    steps = []
    first_bad = None
    var = None
    if is_eq:
        syms = set()
        for ln in lines:
            try:
                syms |= _P(ln.replace("=", "-(") + ")").free_symbols
            except Exception:
                pass
        var = _pick_var(sp.Add(*syms) if syms else Symbol("x"))

    for i in range(1, len(lines)):
        a, b = lines[i - 1], lines[i]
        ok, note = None, ""
        try:
            # A line of prose isn't a step. Without this the implicit-
            # multiplication parser turns "integral" into E*a*g*i*l*n*r*t and
            # then confidently reports that it doesn't equal the next line.
            if _prose_tokens(a.replace("=", " ")) or _prose_tokens(b.replace("=", " ")):
                raise ValueError("prose")
            if is_eq:
                sa, sb = _solset(a, var), _solset(b, var)
                if sa == sb:
                    ok = True
                    note = f"same solution set: {{{', '.join(sa) or 'none'}}}"
                elif sb and set(sb) < set(sa):
                    # Narrowing to a subset is how people WRITE solutions —
                    # "(x-2)(x-3)=0" then "x = 2" is enumerating a root, not an
                    # error. Flagging it would false-alarm on the commonest
                    # homework pattern there is, so it passes, with a note that
                    # says which roots haven't been written down yet.
                    missing = [s for s in sa if s not in set(sb)]
                    ok = True
                    note = (
                        f"valid — one of the {len(sa)} solutions"
                        f" (still to write: {', '.join(missing)})"
                    )
                else:
                    extra = [s for s in sb if s not in set(sa)]
                    ok = False
                    note = (
                        f"introduces {', '.join(extra)}, which doesn't solve the line above"
                        if extra
                        else f"solutions changed: {{{', '.join(sa) or 'none'}}} became {{{', '.join(sb) or 'none'}}}"
                    )
            else:
                d = simplify(_P(a) - _P(b))
                ok = d == 0
                note = (
                    "algebraically identical to the line above"
                    if ok
                    else f"not equal — the difference is {d}, not 0"
                )
        except UserError:
            raise
        except Exception:
            ok, note = None, "couldn't read this line as math"
        steps.append({"from": a, "to": b, "ok": ok, "note": note})
        if ok is False and first_bad is None:
            first_bad = i

    checked = sum(1 for s in steps if s["ok"] is not None)
    good = sum(1 for s in steps if s["ok"] is True)
    if first_bad is not None:
        head = rf"\text{{First error on line }} {first_bad + 1}"
        summary = f"line {first_bad + 1} doesn't follow from line {first_bad}"
        verified = False
    elif checked == 0:
        head = r"\text{Couldn't read any of those steps}"
        summary = "none of the lines parsed as math"
        verified = None
    elif good == checked:
        skipped = len(steps) - checked
        head = rf"\text{{All }} {checked} \text{{ steps check out}}"
        summary = "every step follows from the one before it" + (
            " (same solution set at each stage)" if is_eq else ""
        )
        verified = True
        if skipped:
            # An unreadable line is a hole in the chain. Don't stamp a clean
            # verification across a gap -- say what was actually checked.
            head = rf"\text{{The }} {checked} \text{{ readable steps hold}}"
            summary += f" — but {skipped} line(s) weren't math and were skipped"
            verified = None
    else:
        head = r"\text{Some steps couldn't be checked}"
        summary = f"{good} of {checked} steps verified; the rest wouldn't parse"
        verified = None

    return _result(
        type="check my work",
        input_latex=r"\text{your work, " + str(len(lines)) + r" lines}",
        answer_latex=head,
        answer_str=summary,
        verified=verified,
        verify_note=(
            "each line checked against the one above it — "
            + ("solution sets compared" if is_eq else "algebraic equality compared")
        ),
        steps=[],
        work=steps,
        first_bad=first_bad,
    )


def check_step(previous, step):
    """Judge ONE step a student just proposed, against the line before it.

    This is the ground-truth half of the guided mode. The important property is
    the one that breaks trust in LLM tutors when it fails: it must never tell a
    student their correct step is wrong. So the verdict is the CAS's -- the same
    equivalence check `check_work` uses -- and a model, if one is even involved,
    only ever phrases the hint. It never decides.
    """
    prev = (_line_pairs(previous) or [""])[-1]
    nxt = (_line_pairs(step) or [""])[0]
    if not prev or not nxt:
        raise UserError(
            "Give me the line you're working from and the step you want to take."
        )
    if _prose_tokens(prev.replace("=", " ")) or _prose_tokens(nxt.replace("=", " ")):
        return {
            "ok": True,
            "status": "unreadable",
            "note": "I couldn't read that as math, so I can't judge it — try writing it as an equation or expression.",
            "solved": False,
        }
    is_eq = "=" in prev and "=" in nxt
    try:
        if is_eq:
            syms = set()
            for ln in (prev, nxt):
                try:
                    syms |= _P(ln.replace("=", "-(") + ")").free_symbols
                except Exception:
                    pass
            var = _pick_var(sp.Add(*syms) if syms else Symbol("x"))
            sa, sb = _solset(prev, var), _solset(nxt, var)
            partial = False
            if sa == sb:
                status, note = (
                    "ok",
                    f"That holds — same solution set: {{{', '.join(sa) or 'none'}}}.",
                )
            elif sb and set(sb) < set(sa):
                partial = True
                left = [s for s in sa if s not in set(sb)]
                status = "ok"
                note = f"Valid — that's one of the solutions. Still to find: {', '.join(left)}."
            else:
                extra = [s for s in sb if s not in set(sa)]
                status = "wrong"
                note = (
                    f"That introduces {', '.join(extra)}, which doesn't satisfy the line you started from."
                    if extra
                    else "That changes which values solve the equation."
                )
            # "Solved" = a bare `x = value` AND nothing left to find. Naming one
            # root of two is a valid step but not the end of the problem —
            # calling it done would close a guided session early.
            lhs, rhs = nxt.split("=", 1)
            solved = (
                status == "ok"
                and not partial
                and _P(lhs).free_symbols == {var}
                and _P(lhs) == var
                and not _P(rhs).free_symbols
            )
        else:
            d = simplify(_P(prev) - _P(nxt))
            if d == 0:
                status, note = (
                    "ok",
                    "That holds — algebraically identical to what you started from.",
                )
            else:
                status, note = "wrong", f"Not equivalent — the two differ by {d}."
            solved = False
    except UserError:
        raise
    except Exception:
        return {
            "ok": True,
            "status": "unreadable",
            "note": "I couldn't parse that step — check the notation and try again.",
            "solved": False,
        }
    return {"ok": True, "status": status, "note": note, "solved": bool(solved)}


def check_step_json(previous, step):
    """JSON wrapper for the browser."""
    import json

    try:
        r = check_step(previous, step)
    except UserError as e:
        r = {"ok": False, "error": str(e)}
    except Exception:
        r = {"ok": False, "error": _GENERIC_PARSE_ERROR}
    return json.dumps(r, default=lambda o: str(o))


def check_work_json(raw):
    """JSON wrapper for the browser, mirroring solve_json."""
    import json

    try:
        r = check_work(raw)
    except UserError as e:
        r = {"ok": False, "error": str(e)}
    except Exception:
        r = {"ok": False, "error": _GENERIC_PARSE_ERROR}
    return json.dumps(r, default=lambda o: str(o))


def solve_json(query):
    """JSON-string wrapper so the browser (Pyodide) gets plain data, no proxies."""
    import json

    r = solve(query)
    # Everything in a result is already JSON-safe (str/bool/None/float/list),
    # but guard against stray SymPy objects sneaking into a field.
    return json.dumps(r, default=lambda o: str(o))
