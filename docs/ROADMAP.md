# Exact vs. Wolfram Alpha — capability gap & roadmap

_Research synthesis, 2026-07-22. Goal: make Exact competitive with Wolfram Alpha in
**math** capability and scope, without chasing WA's real-world data layer._

## The thesis

Exact already **owns single-variable calculus** (its credibility anchor). The fastest
path to Wolfram-competitive scope is **grammar-only SymPy wins** — the engine already
computes far more than the grammar currently exposes. And the durable moat is **not**
matching WA's scope; it's being **verified, free, and unpaywalled** where every
competitor is trusted-but-unchecked and paywalled.

> Key landmine the research flagged: `sympy.stats` is symbolic-distribution only — it
> does **not** do mean/median/sd of a data list or regression. Descriptive stats must
> be **hand-built** (cheap exact arithmetic over a list). Don't assume the module gives it.

## Where Exact stands

| Domain | Wolfram | Exact | SymPy can build | Value | Effort |
|---|---|---|---|---|---|
| Arithmetic | full | **full** | yes | high | low |
| Single-variable calculus | full | **full** | yes | high | low |
| Algebra: systems, inequalities | full | partial | yes | high | **low** |
| Limits & series; summation/sequences | full | partial | yes | high | **low** |
| Trigonometry (identities, equations) | full | partial | yes | high | **low** |
| Linear algebra (matrices, eigen) | full | none | yes | high | medium |
| Multivariable/vector calculus | partial | partial | partial | high | medium |
| Statistics & probability | full | none | partial* | high | medium |
| Number theory / combinatorics | partial | none | yes | medium | medium |
| Plotting (polar/parametric/3D) | full | partial | partial | medium | medium |
| Differential equations (ODE) | full | none | partial | high | **high** |
| Complex analysis | partial | partial | partial | low | medium |
| Units & dimensional analysis | full | none | yes | low | low |
| Step-by-step (all domains) | full (paywalled) | partial | LLM-narrated | high | medium |

\* stats: distributions via `sympy.stats`; descriptive stats + regression are hand-built.

## Roadmap

### Tier 1 — Quick wins (days, grammar+translation only; ~doubles perceived scope)
- **Systems of equations** — `linsolve` / `nonlinsolve` (multi-equation input)
- **Inequalities** — `reduce_inequalities` / `solve_univariate_inequality` → interval output
- **Summation, products & sequences** — `Sum` / `summation` / `product` (finite + infinite)
- **Trigonometry** — identity simplify (`trigsimp`/`expand_trig`), exact special-angle values, trig-equation solving
- **Exact-arithmetic polish** — guaranteed fraction/radical forms + an arbitrary-precision "approx" toggle

Rationale: pure parser/translator work over math SymPy already does perfectly, each
numerically verifiable (reinforces the trust moat), and it closes the most visible WA
gaps for the algebra/precalc/calc audience Exact already serves.

### Tier 2 — High value, moderate effort
- **Linear algebra** — matrix literal grammar → det/inverse/rref/rank/eigenvalues/eigenvectors/solve Ax=b (exact radicals, a differentiator vs numpy)
- **Multivariable calculus** — gradient, Hessian, Jacobian, double/triple integrals over explicit bounds
- **Descriptive statistics** — hand-built mean/median/mode/variance/sd/quartiles/linear-fit over a data list + `sympy.stats` named distributions
- **Polar & parametric 2D plots** — `lambdify` → existing Plotly renderer
- **Number theory & combinatorics** — isprime/factorint/gcd/lcm/binomial/factorial; `rsolve` recurrences

### Tier 3 — Hard / niche
- **Differential equations** — first-order + linear constant-coeff ODEs via `dsolve` with ICs (biggest parsing lift; skip PDEs)
- **3D surface / contour plots** — `lambdify` → a JS 3D renderer (new plumbing)
- **Vector calculus** — div/curl/Laplacian via `CoordSys3D`
- **Complex analysis** — residues/poles/re-im (niche)
- **Units** — `convert_to` (cheap filler)
- **Cross-domain unpaywalled step-by-step** — the strategic capstone: BYOK LLM narrates steps around verified SymPy calls, numeric self-verification keeps it honest

## Why Exact can win (the moat)

- **Verified by default** — every answer numerically re-checked in-browser. WA doesn't expose this; Symbolab/Mathway are documented to ship wrong steps. "We checked our own answer" is a claim no competitor makes.
- **Free + offline forever** — client-side SymPy, no server, no vendor kill-switch (Microsoft Math Solver was retired July 2025; an open local tool can't be pulled).
- **Step-by-step is never paywalled** — WA Pro / Symbolab / Mathway all lock the pedagogy students actually need. Exact gives verified steps free. Strongest wedge.
- **Open-source, auditable, extensible.**
- **BYOK teaching, no SaaS markup** — students pay their own API cents, not $10/mo rent.
- **Exact symbolic output** — radicals/rationals, not floats.
- **Phone-first** — photo input + clean mobile UX vs WA's desktop-era interface.

## Do NOT chase

WA's real-world **data layer** — weather, demographics, financial/market data, chemistry
& physics constants, geographic/astronomical facts. That's a knowledge graph, not a CAS;
a browser SymPy engine can't and shouldn't replicate it. Also skip: PDEs beyond trivial
cases (`pdsolve` is thin), general group theory, general contour integration, and
data-science stats (hypothesis testing / ANOVA — scipy territory, not WASM-safe).
