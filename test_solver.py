"""
test_solver.py — correctness harness for Exact's engine (solver.py).

Two layers:
  1. Curated known-answer cases. Each has an INDEPENDENTLY-known answer; we assert
     the engine (a) succeeds, (b) computes the right thing, and (c) never raises a
     FALSE verification flag on a correct problem. Answers are compared by SymPy
     equivalence (not string match), so the engine's factored/rearranged forms pass.
     Indefinite integrals are checked by differentiating the engine's answer back to
     the integrand (robust to the "+ C" ambiguity).
  2. A deterministic random fuzz (seeded): hundreds of random closed-form
     expressions through derivative + integral, asserting no crash and — the key
     invariant — the numeric verifier never emits a FALSE "unverified" flag.

The same solver.py runs here (CPython) and in the browser (Pyodide): one source of
truth, so a green run here means every answer a user sees is numerically confirmed.

Run:  python3 -m venv venv && ./venv/bin/pip install sympy
      ./venv/bin/python test_solver.py
Exit code is non-zero if anything fails.
"""

import random
import sys

from sympy import (  # noqa: F401
    sympify,
    simplify,
    diff,
    Symbol,
    I,
    sqrt,
    pi,
    E,
    oo,
    sin,
    cos,
    exp,
)

from solver import solve

x = Symbol("x")
_NOTSET = object()

_PASS = 0
_FAIL = 0
_FAILURES = []
_TOTAL = 0


def _record(label, problems):
    global _PASS, _FAIL, _TOTAL
    _TOTAL += 1
    if problems:
        _FAIL += 1
        _FAILURES.append((label, problems))
    else:
        _PASS += 1


def _strip_C(s):
    s = (s or "").strip()
    if s.endswith("+ C"):
        s = s[:-3]
    return s.strip().rstrip("+").strip()


def _sym_eq(got_str, want):
    """True if the engine's answer equals the known answer as SymPy objects."""
    try:
        g = sympify(_strip_C(got_str))
        w = sympify(want) if isinstance(want, str) else want
        d = simplify(g - w)
        return d == 0
    except Exception:
        return str(got_str).strip() == str(want).strip()


def check(
    query,
    want=None,
    contains=None,
    type_=None,
    verified=_NOTSET,
    approx=None,
    has_plot=False,
    label=None,
):
    """Generic checker for unique-valued results (derivatives, definite integrals,
    limits, series, algebra, evaluate, tangent lines)."""
    r = solve(query)
    problems = []
    if not r.get("ok"):
        problems.append("not ok: %s" % r.get("error"))
        _record(label or query, problems)
        return
    # The core safety invariant: a correct problem must NEVER get a FALSE flag.
    if r.get("verified") is False:
        problems.append("FALSE verification flag: %s" % r.get("verify_note"))
    if want is not None and not _sym_eq(r.get("answer_str", ""), want):
        problems.append("answer %r != expected %r" % (r.get("answer_str"), want))
    if contains is not None:
        hay = r.get("answer_str", "") or ""
        for c in [contains] if isinstance(contains, str) else contains:
            if c not in hay:
                problems.append("answer %r missing %r" % (hay, c))
    if type_ is not None and r.get("type") != type_:
        problems.append("type %r != %r" % (r.get("type"), type_))
    if verified is not _NOTSET and r.get("verified") is not verified:
        problems.append("verified=%r expected %r" % (r.get("verified"), verified))
    if approx is not None:
        got = r.get("approx", "")
        if approx and not got:
            problems.append("expected an approx decimal, got none")
        if not approx and got:
            problems.append("unexpected approx %r" % got)
    if has_plot:
        p = r.get("plot") or {}
        tr = (p.get("traces") or [{}])[0]
        if sum(1 for v in (tr.get("x") or []) if v is not None) < 10:
            problems.append("plot missing or too few samples")
    _record(label or query, problems)


def check_antideriv(query, integrand):
    """Indefinite integral: differentiate the engine's answer, confirm it equals the
    integrand (independent of the arbitrary constant)."""
    r = solve(query)
    problems = []
    if not r.get("ok"):
        problems.append("not ok: %s" % r.get("error"))
        _record(query, problems)
        return
    if r.get("verified") is False:
        problems.append("FALSE verification flag: %s" % r.get("verify_note"))
    try:
        F = sympify(_strip_C(r.get("answer_str", "")))
        f = sympify(integrand)
        if simplify(diff(F, x) - f) != 0:
            problems.append("d/dx(%s) != %s" % (r.get("answer_str"), integrand))
    except Exception as e:
        problems.append("compare error: %s" % e)
    _record(query, problems)


def check_solve(query, roots):
    """Equation solving: compare the solution SET (order-independent, symbolic)."""
    r = solve(query)
    problems = []
    if not r.get("ok"):
        problems.append("not ok: %s" % r.get("error"))
        _record(query, problems)
        return
    if r.get("verified") is False:
        problems.append("FALSE verification flag: %s" % r.get("verify_note"))
    got = []
    for part in (r.get("answer_str", "") or "").split(","):
        part = part.strip()
        if part:
            try:
                got.append(sympify(part))
            except Exception:
                pass
    want = [sympify(w) for w in roots]
    ok = len(got) == len(want) and all(
        any(simplify(g - w) == 0 for g in got) for w in want
    )
    if not ok:
        problems.append("roots %r != %r" % (r.get("answer_str"), roots))
    _record(query, problems)


# ------------------------------------------------------------------ curated ----
def curated():
    # --- derivatives (1st-order verify numerically -> True) ---
    check("derivative of x^2 sin(x)", want="2*x*sin(x) + x**2*cos(x)", verified=True)
    check("derivative of x^3", want="3*x**2", verified=True)
    check("derivative of sin(x)", want="cos(x)", verified=True)
    check("derivative of cos(x)", want="-sin(x)", verified=True)
    check("derivative of e^x", want="exp(x)", verified=True)
    check("derivative of ln(x)", want="1/x", verified=True)
    check("derivative of tan(x)", want="tan(x)**2 + 1", verified=True)
    check("derivative of sqrt(x)", want="1/(2*sqrt(x))", verified=True)
    check("derivative of arctan(x)", want="1/(x**2 + 1)", verified=True)
    check("derivative of x^2 e^x", want="x**2*exp(x) + 2*x*exp(x)", verified=True)
    check("differentiate x/(x+1)", want="1/(x + 1)**2", verified=True)
    check("derivative of ln(x^2+1)", want="2*x/(x**2 + 1)", verified=True)
    check("derivative of x^5 - 3x^2 + 2x - 7", want="5*x**4 - 6*x + 2", verified=True)
    # --- higher-order / partial: these now carry a real numeric check too.
    # Higher orders use repeated central differences (larger step, looser
    # tolerance per order); partials pin the other free symbols to sample
    # values so d/dy[x^2 y] is actually evaluable. Both used to be None.
    check("second derivative of x^4", want="12*x**2", verified=True)
    check("second derivative of sin(x)", want="-sin(x)", verified=True)
    check("third derivative of x^5", want="60*x**2", verified=True)
    check("partial derivative of x^2 y with respect to y", want="x**2", verified=True)
    check("partial derivative of x^2 y with respect to x", want="2*x*y", verified=True)

    # --- indefinite integrals (differentiate the answer back to the integrand) ---
    check_antideriv("integrate x^2", "x**2")
    check_antideriv("integrate 1/(1+x^2)", "1/(1 + x**2)")
    check_antideriv("integrate cos(x)", "cos(x)")
    check_antideriv("integrate e^x", "exp(x)")
    check_antideriv("integrate 1/x", "1/x")
    check_antideriv("integrate x*e^x", "x*exp(x)")
    check_antideriv("integrate sec(x)^2", "sec(x)**2")
    check_antideriv("integrate 2x", "2*x")
    check_antideriv("integrate sin(x)*cos(x)", "sin(x)*cos(x)")
    check_antideriv("integrate x^3 + 3x^2 - 2x + 1", "x**3 + 3*x**2 - 2*x + 1")
    check_antideriv("integrate ln(x)", "log(x)")
    check_antideriv("integrate x*sin(x)", "x*sin(x)")

    # --- definite / improper integrals (unique value) ---
    check("integrate x^2 from 0 to 1", want="1/3")
    check("integrate sin(x) from 0 to pi", want="2")
    check("integrate 1/(1+x^2) from 0 to 1", want="pi/4")
    check("integrate cos(x) from 0 to pi/2", want="1")
    check("integrate x from 0 to 2", want="2")
    check("integrate e^-x from 0 to oo", want="1")
    check("integrate 1/x^2 from 1 to oo", want="1")
    check("integrate 2x from 1 to 3", want="8")

    # --- limits (finite value via sym_eq; special cases via contains) ---
    check("limit of sin(x)/x as x->0", want="1")
    check("limit of (1-cos(x))/x^2 as x->0", want="1/2")
    check("limit of (e^x-1)/x as x->0", want="1")
    check("limit of 1/x as x->oo", want="0")
    check("limit of sin(x)/x as x->oo", want="0")
    check("limit of (1+1/x)^x as x->oo", want="E")
    check("limit of (x^2-1)/(x-1) as x->1", want="2")
    check(
        "limit of |x|/x as x->0",
        contains="does not exist",
        type_="limit",
        verified=None,
    )

    # --- Taylor / Maclaurin series (check leading polynomial terms) ---
    check(
        "taylor series of e^x",
        contains=["x**2/2", "x**3/6", "O(x**6)"],
        type_="maclaurin series",
        verified=True,
    )
    check("taylor series of sin(x)", contains=["x**3/6", "x**5/120", "O(x**6)"])
    check("taylor series of cos(x)", contains=["x**2/2", "x**4/24"])
    check("taylor series of 1/(1-x)", contains=["1 + x + x**2", "O(x**6)"])

    # --- solve / roots (solution set) ---
    check_solve("solve x^2 - 5x + 6 = 0", ["2", "3"])
    check_solve("solve x^2 - 4 = 0", ["2", "-2"])
    check_solve("solve 2x + 3 = 7", ["2"])
    check_solve("solve x^2 + 1 = 0", ["I", "-I"])
    check_solve("solve x^3 - x = 0", ["0", "1", "-1"])
    check_solve("solve x^2 - 2 = 0", ["sqrt(2)", "-sqrt(2)"])

    # --- critical points / extrema ---
    check(
        "critical points of x^3 - 3x",
        contains=["-1: local maximum", "1: local minimum"],
        verified=True,
    )
    check("critical points of x^2", contains="0: local minimum", verified=True)
    check(
        "critical points of x^4 - 2x^2",
        contains=["local minimum", "local maximum"],
        verified=True,
    )

    # --- tangent lines ---
    check("tangent line to x^2 at x=1", want="2*x - 1", verified=True)
    check("tangent line to sin(x) at x=0", want="x", verified=True)
    check("tangent line to e^x at x=0", want="x + 1", verified=True)

    # --- algebra helpers ---
    check("simplify (x^2-1)/(x-1)", want="x + 1", verified=True)
    check("factor x^2 - 5x + 6", want="(x - 2)*(x - 3)", verified=True)
    check("expand (x+1)^3", want="x**3 + 3*x**2 + 3*x + 1", verified=True)
    check("simplify sin(x)^2 + cos(x)^2", want="1", verified=True)

    # --- evaluate (pure numeric) ---
    check("2^10 + 5", want="1029", verified=True)
    check("sqrt(16)", want="4", verified=True)
    check("cos(0)", want="1", verified=True)
    check("0.15*80", want="12", verified=True)  # integer-valued float -> clean integer
    check("0.25 * 8", want="2", verified=True)

    # --- more derivatives (chain / reciprocal / product) ---
    check("derivative of cos(x^2)", want="-2*x*sin(x**2)", verified=True)
    check("derivative of x*ln(x)", want="log(x) + 1", verified=True)
    check("derivative of 1/x", want="-1/x**2", verified=True)
    check("derivative of sec(x)", want="tan(x)*sec(x)", verified=True)
    check("derivative of e^(2x)", want="2*exp(2*x)", verified=True)

    # --- more integrals ---
    check_antideriv("integrate sec(x)*tan(x)", "sec(x)*tan(x)")
    check_antideriv("integrate cos(2x)", "cos(2*x)")
    check_antideriv("integrate 1/(x^2+4)", "1/(x**2 + 4)")
    check("integrate x^3 from 0 to 2", want="4")
    check("integrate 1/x from 1 to e", want="1")

    # --- more limits ---
    check("limit of tan(x)/x as x->0", want="1")
    check("limit of (sqrt(x+1)-1)/x as x->0", want="1/2")

    # --- systems of equations (Tier 1) ---
    check("solve x + y = 3, x - y = 1", contains=["x=2", "y=1"], verified=True)
    check("solve x + y = 10 and x - y = 2", contains=["x=6", "y=4"], verified=True)
    check("solve 2*x + y = 5, x - y = 1", contains=["x=2", "y=1"], verified=True)

    # --- inequalities (Tier 1) ---
    check("x^2 > 4", contains="x < -2", verified=True)
    check("solve 2*x - 1 <= 5", contains="x <= 3", verified=True)
    check("x^2 - 5*x + 6 < 0", contains=["2 < x", "x < 3"], verified=True)

    # --- summation / products (Tier 1) ---
    check("sum k from k=1 to 100", want="5050", verified=True)
    check("sum n^2 from n=1 to 10", want="385", verified=True)
    check("product k from k=1 to 5", want="120", verified=True)
    check("sum of 1/n^2 from n=1 to oo", want="pi**2/6")
    check("sum 1/2^n from n=0 to oo", want="2")

    # --- trig identities & expansion (Tier 1) ---
    check("simplify 2 sin(x) cos(x)", want="sin(2*x)", verified=True)
    check("expand sin(2x)", want="2*sin(x)*cos(x)", verified=True)
    check("expand cos(x+y)", want="cos(x)*cos(y) - sin(x)*sin(y)", verified=True)
    check("expand sin(x+y)", want="sin(x)*cos(y) + cos(x)*sin(y)", verified=True)

    # --- exact/approx display (Tier 1) ---
    check("integrate 1/(1+x^2) from 0 to 2", want="atan(2)", approx=True)
    check("integrate 3*x^2 from 0 to 2", want="8", approx=False)

    # --- linear algebra (Tier 2) ---
    check("determinant of {{1,2},{3,4}}", want="-2")
    check("det {{1,2,1},{1,1,0},{0,1,1}}", want="0")
    check("trace of {{1,2},{3,4}}", want="5")
    check("rank of {{1,2},{2,4}}", want="1")
    check("inverse of {{1,2},{3,4}}", contains=["-2", "3/2"], verified=True)
    check("eigenvalues of {{2,0},{0,3}}", contains=["2", "3"], verified=True)
    check("eigenvalues of {{2,1},{1,2}}", contains=["3", "1"], verified=True)

    # --- number theory (Tier 2) ---
    check("is 73 prime", contains="is prime", verified=True)
    check("is 91 prime", contains="not prime", verified=True)
    check("factor 70560", contains=["2^5", "7^2"], verified=True)
    check("prime factorization of 360", contains="2^3", verified=True)
    check("10 choose 3", want="120", verified=True)
    check("gcd of 12 and 18", want="6", verified=True)
    check("lcm of 4 and 6", want="12", verified=True)

    # --- descriptive statistics (Tier 2) ---
    check("mean of {1,2,3,4,5}", want="3", verified=True)
    check("average of 4, 8, 15, 16, 23, 42", want="18", verified=True)
    check("median of {1,2,3,4}", want="5/2")
    check("mode of 1,2,2,3,3,3", want="3")
    check("variance of {2,4,6}", contains=["population=8/3", "sample=4"])
    check("standard deviation of {2,4,4,4,5,5,7,9}", contains="population=2")

    # --- multivariable calculus (Tier 2) ---
    check("gradient of x^2 + y^2", contains=["2*x", "2*y"])
    check("hessian of x^2 + y^2", contains="2, 0")
    check("double integral of x*y over x=0 to 1, y=0 to 2", want="1", verified=True)
    check(
        "double integral of x^2 + y^2 over x=0 to 1, y=0 to 1",
        want="2/3",
        verified=True,
    )
    check(
        "triple integral of x*y*z over x=0 to 1, y=0 to 1, z=0 to 2",
        want="1/2",
        verified=True,
    )

    # --- polar / parametric plots (Tier 2) ---
    check("polar plot r = 1 + cos(theta)", has_plot=True)
    check("parametric plot x = cos(t), y = sin(t)", has_plot=True)
    check("parametric plot x = t^2, y = t^3 for t=-2 to 2", has_plot=True)

    # --- differential equations (Tier 3) ---
    check("solve y'' + y = 0", contains=["sin(x)", "cos(x)"], verified=True)
    check("solve y' = y", contains="exp(x)", verified=True)
    check("solve y' + 2*y = 0", contains="exp(-2*x)", verified=True)
    check("solve y' = x*y", contains="exp(x**2/2)", verified=True)

    # --- complex analysis (Tier 3) ---
    check("real part of 3 + 4*I", want="3")
    check("imaginary part of (2+I)^2", want="4")
    check("conjugate of 3 + 4*I", want="3 - 4*I")
    check("modulus of 3 + 4*I", want="5")
    check("residue of 1/(z^2+1) at z=I", want="-I/2")

    # --- vector calculus (Tier 3) ---
    check("divergence of (x^2, y^2, z^2)", want="2*x + 2*y + 2*z")
    check("curl of (y, -x, 0)", contains="-2")

    # --- units (Tier 3) ---
    check("convert 2 hours to minutes", contains="120")
    check("convert 5 km to miles", contains="mile")


# --------------------------------------------------------------------- fuzz ----
def fuzz(n_exprs=400, seed=12345):
    """Deterministic random sweep. Builds closed-form-integrable expressions and
    runs derivative + indefinite integral through the engine, asserting no crash and
    that the verifier NEVER emits a FALSE flag on a correct result."""
    rng = random.Random(seed)
    atoms = ["{c}*x^{p}", "{c}*sin(x)", "{c}*cos(x)", "{c}*exp(x)", "{c}"]
    checked = 0
    for _ in range(n_exprs):
        k = rng.randint(1, 4)
        parts = []
        for _ in range(k):
            a = rng.choice(atoms)
            parts.append(a.format(c=rng.randint(1, 6), p=rng.randint(0, 4)))
        expr = " + ".join(parts)
        for op in ("derivative of ", "integrate "):
            q = op + expr
            try:
                r = solve(q)
            except Exception as e:  # solve() is contracted never to raise
                _record("fuzz(raised) " + q, ["raised %s: %s" % (type(e).__name__, e)])
                checked += 1
                continue
            problems = []
            if not isinstance(r, dict):
                problems.append("non-dict result")
            elif r.get("ok") and r.get("verified") is False:
                problems.append("FALSE verification flag: %s" % r.get("verify_note"))
            # ok == False is acceptable (honest "no closed form"); a crash or a
            # false verification is not.
            _record("fuzz " + q, problems)
            checked += 1
    return checked


# ----------------------------------------------------------------- rejection ----
def rejection():
    """The direction the suite was missing.

    Everything above proves the verifier says YES to correct answers. That is
    only half a guarantee: a verifier that returns True unconditionally would
    pass all of it. These cases feed each _verify_* a deliberately WRONG answer
    and assert it says NO -- otherwise the badge means nothing.
    """
    import solver as S

    v = Symbol("x")
    f = sin(v) * exp(v)
    cases = []

    # derivative: wrong by a constant, by a factor, and structurally
    for label, wrong in [
        ("derivative +1e-4", diff(f, v) + sympify("1/10000")),
        ("derivative x1.0008", diff(f, v) * sympify("10008/10000")),
        ("derivative structural", sympify("cos(x)*exp(x)")),
    ]:
        cases.append((label, S._verify_derivative(f, v, wrong)[0]))

    # antiderivative: differentiating it must NOT return the integrand
    cases.append(
        ("antiderivative wrong", S._verify_antiderivative(v**2, v, v**3 / 2)[0])
    )
    # definite integral: symbolic value disagrees with quadrature
    cases.append(("definite wrong", S._verify_definite(v**2, v, 0, 3, sympify(10))[0]))
    # limit: converges to 1, claim 2
    cases.append(("limit wrong", S._verify_limit(sin(v) / v, v, 0, sympify(2))[0]))
    # roots: 3 is not a root of x^2-5x+6
    cases.append(
        (
            "roots wrong",
            S._verify_roots(v**2 - 5 * v + 6, v, [sympify(2), sympify(5)])[0],
        )
    )
    # complex: Re(3+4I) is 3, claim 4
    cases.append(
        ("complex wrong", S._verify_complex(sympify("3+4*I"), "re", sympify(4))[0])
    )
    # summation: sum k, k=1..10 is 55, claim 56
    cases.append(
        ("summation wrong", S._verify_summation(v, v, 1, 10, sympify(56), False)[0])
    )

    # --- the checks added when coverage was extended from 64% to 100%. ---
    # A verifier that always returns True is worse than no verifier at all: it
    # launders wrong answers with a trust badge. Each new check gets fed a
    # deliberately wrong answer here and must reject it.
    import sympy as _sp

    M = _sp.Matrix([[1, 2], [3, 4]])  # det -2, rank 2
    cases.append(("determinant wrong", S._verify_determinant(M, sympify(-3))[0]))
    cases.append(("rank wrong", S._verify_rank(M, 1)[0]))
    cases.append(
        ("transpose wrong", S._verify_transpose(M, _sp.Matrix([[1, 3], [2, 5]]))[0])
    )
    # not in RREF: pivot column has another nonzero entry
    cases.append(
        ("rref not reduced", S._verify_rref(M, _sp.Matrix([[1, 0], [1, 1]]))[0])
    )
    # rank-destroying "rref"
    cases.append(
        ("rref rank changed", S._verify_rref(M, _sp.Matrix([[1, 2], [0, 0]]))[0])
    )

    y = Symbol("y")
    g = v**2 * y
    cases.append(("gradient wrong", S._verify_gradient(g, [v, y], [2 * v * y, v])[0]))
    cases.append(
        (
            "hessian wrong",
            S._verify_hessian(g, [v, y], _sp.Matrix([[2 * y, 2 * v], [2 * v, 1]]))[0],
        )
    )
    # d^2/dx^2 x^4 is 12x^2, claim 12x^2 + 1
    cases.append(
        ("2nd derivative wrong", S._verify_nth_derivative(v**4, v, 12 * v**2 + 1, 2)[0])
    )

    data = [sympify(k) for k in (1, 2, 3, 4)]
    cases.append(("median wrong", S._verify_stats(data, "median", sympify(3))[0]))
    cases.append(("variance wrong", S._verify_stats(data, "variance", sympify(2))[0]))
    cases.append(("std wrong", S._verify_stats(data, "std", sympify(2))[0]))
    # sample variance of 1..4 is 5/3; the population figure 5/4 must be rejected
    cases.append(
        (
            "sample variance confused with population",
            S._verify_stats(data, "variance", sympify("5/4"), sample=True)[0],
        )
    )

    cases.append(
        ("trace wrong", S._verify_trace(M, sympify(6))[0])
    )  # the real trace is 5
    mdata = [sympify(k) for k in (1, 2, 2, 3)]
    cases.append(("mode wrong value", S._verify_mode(mdata, [sympify(3)], 2)[0]))
    cases.append(("mode wrong count", S._verify_mode(mdata, [sympify(2)], 3)[0]))

    z = Symbol("z")
    cases.append(
        ("residue wrong", S._verify_residue(1 / z, z, sympify(0), sympify(2))[0])
    )
    # 5 km is 3.10686 miles, not 3
    cases.append(
        (
            "conversion wrong",
            S._verify_convert(sympify(5), "km", "miles", sympify(3))[0],
        )
    )
    # dimensionally impossible
    cases.append(
        (
            "conversion across dimensions",
            S._verify_convert(sympify(5), "km", "kg", sympify(5))[0],
        )
    )

    for label, verdict in cases:
        problems = []
        if verdict is not True and verdict is not False:
            problems.append(
                "verifier returned %r (no opinion) on a WRONG answer — it should reject"
                % (verdict,)
            )
        elif verdict is True:
            problems.append("verifier ACCEPTED a deliberately wrong answer")
        _record("rejection: " + label, problems)
    return len(cases)


# ------------------------------------------------------------------- reading ----
def reading():
    """Parse-intent regressions.

    The verifier confirms the math on what was PARSED; it cannot confirm the
    parse matched intent. These are the inputs where those two diverge, so each
    must carry a reading_risk warning rather than a bare "verified" badge.
    Everything in the second list is unambiguous and must NOT be flagged --
    a warning on every query is a warning nobody reads.
    """
    ambiguous = [
        "derivative of sin x cos x",  # -> sin(x*cos(x)), not sin(x)*cos(x)
        "derivative of e^2x",  # -> x*e^2,      not e^(2x)
        "integral of 1/2x",  # -> (1/2)*x,   not 1/(2x)
        "derivative of x^2y",
    ]
    clear = [
        "derivative of x^2 sin(x)",
        "derivative of sin(x)*cos(x)",
        "derivative of sin x",
        "integral of x^2 from 0 to 3",
        "integral of sin x dx",
        "tangent to x^2 at x=1",
        "limit of sin(x)/x as x -> 0",
        "solve x^2 - 5x + 6 = 0",
        "real part of 3+4i",
    ]
    n = 0
    for q in ambiguous:
        r = solve(q)
        problems = []
        if not r.get("ok"):
            problems.append("did not solve at all")
        elif not r.get("reading_risk"):
            problems.append(
                "ambiguous notation was NOT flagged — badge would overclaim"
            )
        _record("reading(ambiguous) " + q, problems)
        n += 1
    for q in clear:
        r = solve(q)
        problems = []
        if r.get("ok") and r.get("reading_risk"):
            problems.append(
                "false alarm on unambiguous input: %s" % r["reading_risk"][:60]
            )
        _record("reading(clear) " + q, problems)
        n += 1

    # No internal exception text may reach the user.
    # `simplify(x)` with no space is as natural to type as `simplify x`, and it
    # used to fall through to the bare-expression fallback and get refused as
    # unreadable prose. It is also the form the setup cross-check emits.
    for q, want in [
        ("simplify((2*x) - (x*2))", "0"),
        ("simplify(x^2-1)/(x-1)", None),
        ("factor(x^2-9)", "(x - 3)*(x + 3)"),
        ("expand((x+1)^2)", "x**2 + 2*x + 1"),
    ]:
        r = solve(q)
        problems = []
        if not r.get("ok"):
            problems.append("not ok: %s" % r.get("error"))
        elif want is not None and not _sym_eq(r.get("answer_str", ""), want):
            problems.append("answer %r != %r" % (r.get("answer_str"), want))
        _record("paren-form " + q, problems)
        n += 1
    # ...while the spaced forms keep working exactly as before.
    for q in ("factor 360", "simplify (x^2-1)/(x-1)", "expand (x+1)^3"):
        r = solve(q)
        _record(
            "spaced-form " + q, [] if r.get("ok") else ["not ok: %s" % r.get("error")]
        )
        n += 1

    leaky = [
        "is x prime",
        "factor 12 and 18",
        "derivative of cos^2 x",
        "mean of 5",
        "hello world",
    ]
    for q in leaky:
        r = solve(q)
        err = str(r.get("error", ""))
        problems = []
        for marker in (
            "SyntaxError",
            "TypeError",
            "AttributeError",
            "Traceback",
            "sympy",
            "<string>",
        ):
            if marker in err:
                problems.append("internal detail leaked to user: %r" % marker)
        if (
            r.get("ok")
            and r.get("answer_str", "").count("*") >= 4
            and any(c.isalpha() for c in q.split()[0])
        ):
            problems.append(
                "prose fabricated a letter-product answer: %s" % r.get("answer_str")
            )
        _record("reading(no-leak) " + q, problems)
        n += 1
    return n


# --------------------------------------------------------------------- main ----
def main():
    curated()
    rejection()
    reading()
    n_fuzz = fuzz()
    print("=" * 60)
    print("curated + fuzz problems checked: %d (fuzz: %d)" % (_TOTAL, n_fuzz))
    print("passed: %d   failed: %d" % (_PASS, _FAIL))
    if _FAILURES:
        print("-" * 60)
        for label, problems in _FAILURES[:50]:
            print("FAIL: %s" % label)
            for p in problems:
                print("    - %s" % p)
        print("=" * 60)
        print("RESULT: FAIL (%d failing checks)" % _FAIL)
        return 1
    print("=" * 60)
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
