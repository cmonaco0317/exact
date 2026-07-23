# Exact — a math solver that shows its work on *why you should believe it*

Type a math problem (or snap a photo of one) and get an answer **computed by a real
computer-algebra system, never by a language model** — then re-checked by a *second,
independent method* before it's shown. Measured: **62 of 63 operations across every
family carry an independent check**, and the badge on each answer tells you exactly
which guarantee you got, including the rare case where you don't get one. Covers
calculus, algebra, linear algebra, differential equations, statistics, and more.
Runs entirely in your browser.

**▶ [Live demo](https://exactmath.vercel.app)** · desktop or phone.

## Why it's reliable

Language models pattern-match answers; they don't compute them, so they make silent
algebra mistakes. WolframAlpha is reliable because it's a *computer-algebra system*
(CAS), not an AI. Exact uses the same principle, with one rule:

> **The LLM never does the math.** It only helps with *language* — reading a photo,
> understanding plain-English problems, and explaining answers. Every number you see
> is computed by SymPy. The model only ever emits a *command string*; it never
> produces a result, and this is enforced structurally, not by prompt.

1. **Parse** — your input (`derivative of x^2 sin(x)`) → a precise SymPy expression.
2. **Compute** — [SymPy](https://www.sympy.org), a real CAS compiled to WebAssembly
   via [Pyodide](https://pyodide.org), does the calculus exactly, in your browser.
3. **Verify** — the answer is re-checked before display by a route *different from the
   one that produced it*. Asking SymPy the same question twice would verify nothing,
   so each check is a genuinely separate implementation:

   | Operation | How it's independently checked |
   |---|---|
   | derivatives (incl. partial) | central finite differences at 8 points, tolerance `1e-7` |
   | higher-order derivatives | repeated central differences, step and tolerance scaled per order |
   | gradient · Hessian | finite differences on every partial, including the mixed ones |
   | definite integrals | mpmath quadrature |
   | limits | approached numerically, convergence confirmed |
   | roots · critical points | substituted back, confirmed ≈ 0 |
   | determinant | hand-rolled Gaussian elimination with partial pivoting |
   | rank | hand-rolled row reduction — pivots counted directly |
   | rref | RREF invariants checked structurally + rank preservation |
   | transpose · trace | entry-by-entry / diagonal summed directly |
   | mean · median · variance · std | plain-Python floats, and the `E[x²] − E[x]²` identity rather than the sum-of-squared-deviations formula the engine uses |
   | mode | occurrences recounted independently |
   | primality | a separately-implemented Miller–Rabin, or an exhibited divisor |
   | complex operations | recomputed in plain Python `complex` arithmetic |
   | residue | numeric contour integral — `(1/2πi) ∮ f dz`, the definition |
   | unit conversion | a hand-entered SI table (1 in = 0.0254 m exactly, …) |

   **Measured coverage: 62 of 63 operations (98%) across every family carry an
   independent check.** The one that doesn't is a *divergent* definite integral, where
   the answer is ∞ and there is no finite value to compare against — it says so.

   One check is weaker than the others, and it's worth naming rather than burying:
   **indefinite integrals** are verified by differentiating the answer and confirming
   it returns the integrand. That's worth doing (differentiation is far more reliable
   than integration), but it is *SymPy checking SymPy* — a cross-check, not a second
   independent engine.

   The sample points are **derived per-expression** from a stable hash of the input,
   not drawn from a fixed list. Same input → same points on every run and identically
   under CPython and Pyodide, so a verdict never flickers; but a blind spot can't be
   permanent the way it is when every problem is probed at the same eight abscissae.

The badge states exactly what you got:

| Badge | Means |
|---|---|
| **✓ Verified** | An independent check ran and passed. |
| **· Exact (SymPy)** | Computed symbolically; no independent check applied (e.g. a divergent integral). |
| **⚠ Unverified** | A check disagreed — shown honestly instead of a confident wrong answer. |
| **✓ Math checked · reading unsure** | The arithmetic checked out, but your input used notation that parses ambiguously. |
| **⚙✓ Setup cross-checked** | A word problem or plain-English request whose setup was derived **twice, independently** — both readings reached the same answer. |
| **⚠ Readings disagree** | The two independent readings reached *different* answers. Both computations are verified; they disagree about what the problem is asking. |

### Verifying the setup, not just the arithmetic

There's a harder problem underneath the badges, and it's the one that actually
matters. A numeric check protects the *cheap* part: arithmetic is what SymPy already
does perfectly. But in the photo and plain-English flows the **model chooses the
problem** — the interpretation, the setup, which quantity is the unknown. Choosing the
wrong setup is far more consequential than an arithmetic slip, and the CAS will then
flawlessly, "verifiably", solve the wrong thing.

You can't check a setup by computing harder. So Exact derives it **twice,
independently**, and compares the two verified answers. Two properties make that a
real check rather than theatre:

1. **The second reading is blind to the first.** If it saw the first setup it would
   anchor to it and agreement would mean nothing. It's a separate call with a
   deliberately different framing — name the unknown and the givens first, *then*
   write commands — so the two attempts fail differently instead of making the same
   mistake twice.
2. **The comparison is on the CAS's answers, not the model's text.** Two correct
   setups routinely produce different command strings. Numeric answers compare
   directly; symbolic ones are compared by asking the CAS whether their difference
   simplifies to zero — using the engine to check itself, which is what it's for.

Agreement is evidence. Disagreement is **shown, never resolved by silently picking
one**. And "couldn't compare" is its own third state, reported as *not cross-checked*
rather than quietly counted as agreement.

This costs a second model call on your own key, and it only runs when the model chose
the setup — typed commands never touch the network.

The parse-vs-intent badge exists because of a related limitation worth naming: **a
numeric check confirms the math on what was *parsed*, not that the parse matched what
you meant.**
`sin x cos x` is a legal reading as `sin(x·cos x)`, and a verifier will happily confirm
the derivative *of that*. So when the input uses notation with more than one defensible
reading (`sin x cos x`, `e^2x`, `1/2x`), Exact shows the reading it chose in an
impossible-to-miss box and downgrades the badge rather than stamping a plain ✓ on a
possibly-misread problem. When SymPy genuinely can't solve something, it says so
rather than guessing.

## "Check my work" — the mode this architecture is actually best at

Paste **your own** steps, one per line. Every line is checked against the one above
it, and you get the first line that breaks and why:

```
2x + 6 = 10
2x = 16     ← ✗ introduces 8, which doesn't solve the line above
x = 8       ← ✓ same solution set: {8}
```

Note the second verdict: line 3 *does* follow from line 2. You were consistent after
the slip, and the tool says so instead of marking everything after the first error
wrong. `(a+b)^2 → a^2 + b^2` comes back with *"not equal — the difference is 2ab"*.

Two things make this the right shape for a CAS:

- **It checks the invariant that actually matters.** For a solve chain that's the
  *solution set*, not equality — `2x = 4` and `x = 2` aren't equal as equations but
  are equivalent as problems, and checking equality would flag every correct
  solution. For an expression chain it's algebraic identity.
- **Narrowing isn't an error.** Writing `(x-2)(x-3) = 0` then `x = 2` is enumerating
  a root, which is how solutions are *written*. It passes, with a note saying which
  roots you haven't written yet. Inventing a root that doesn't satisfy the previous
  line is what gets flagged. A false alarm here tells a correct student they're
  wrong, which is worse than not shipping the feature — so the tests pin both
  directions.

**It needs no API key and no network.** That's deliberate: the most differentiating
thing the tool does shouldn't be the part behind a paywall.

## "Guide me" — a tutor that can't mis-correct you

You drive. Type the problem, then each step you'd take. Every step is judged by the
**CAS**, not by a model:

```
▸ 2x + 6 = 10     Your problem. What's your first step?
✗ 2x = 16         That introduces 8, which doesn't satisfy the line you started from.
✓ 2x = 4          That holds — same solution set: {2}.
✓ x = 2           Solved — and every step to get there was checked.
```

The failure that erodes trust in an LLM tutor is being told a correct step is wrong.
Here that can't happen: the verdict comes from the same equivalence check the solver
uses, so a legal move is always accepted. It also never volunteers the answer — only
whether *your* move was legal — and a mistake doesn't poison the session, because the
next step is checked against the last line that actually held.

Offline, no key.

## What it solves

- **Calculus** — derivatives (higher-order, partial, gradient, Hessian), indefinite/
  definite/improper and double/triple integrals (with steps), limits (one-sided & DNE),
  Taylor/Maclaurin series, summation & products, critical points, tangent lines
- **Algebra** — single equations, systems (linear & nonlinear), inequalities, `factor` /
  `expand` / `simplify`, trig identities & equations
- **Linear algebra** — determinant, inverse, eigenvalues/eigenvectors, rank, rref,
  transpose, trace — *exact* symbolic output (radicals, not floats)
- **Differential equations** — ODEs (`solve y'' + y = 0`), verified with `checkodesol`
- **Discrete & number theory** — primality, prime factorization, gcd/lcm, combinatorics
- **Statistics** — mean, median, mode, variance, standard deviation, summary
- **Complex analysis** — real/imaginary part, conjugate, modulus, argument, residues
- **Vector calculus** — divergence, curl · **Units** — conversions
- **Plots** — 2D functions, polar, parametric, and interactive 3D surfaces

Input is Wolfram-style math: `2x`, `x^2`, `sin x`, `e^x`, `|x|`, `->`, `pi`, `oo`,
matrices as `{{1,2},{3,4}}`, `y'`/`y''` for ODE derivatives. Complex numbers accept
either `3+4i` or `3+4I`.

### Known limits

Named rather than buried, because a tool about trust shouldn't be coy about its own
edges:

- **The badge certifies the parse, not your intent.** See above — ambiguous notation is
  flagged, but a reading you don't check is a reading you're trusting.
- **Setup cross-checking is agreement, not proof.** Two independent readings landing on
  the same answer is real evidence, and it's the strongest check available for
  something a computation can't validate — but two readings can still be wrong the
  same way. It raises the floor; it doesn't close the gap.
- **The antiderivative check is SymPy checking SymPy**, not a second engine.
- **The tutor is model-written prose.** The headline answer is always SymPy's, but the
  "✨ Explain like a tutor" text is generated. It's labelled as such, and its numbers
  are checked against the verified answer — if they disagree, the panel says so.
- **Long computations are cut off at 8 seconds.** Some ordinary-looking input has no
  closed form — `solve x^5 - x - 1 = 0` runs for over a minute in SymPy. The engine runs
  in a Web Worker and is killed and restarted on overrun, so the page never freezes, but
  you get a timeout message instead of an answer.
- **Unrecognized phrasing is refused, not guessed.** The implicit-multiplication parser
  would otherwise read `mean of 5` as `5·E·a·f·m·n·o` and badge it as exact.
- **First load fetches ~13 MB** (Pyodide + SymPy) before the engine is usable, then
  it's cached. The 4.35 MB plot library is no longer part of that — it's fetched on
  demand the first time a result actually draws a graph, which most queries never do.
  "Check my work" needs no network at all once the engine is up.

## Photo input & AI features — bring your own key

The verified solver is **100% free, works offline, and needs no account or key** —
typed commands and expressions are handled entirely locally. Optional AI features run
on your **own** Anthropic API key:

- **✍️ Plain English** — type a problem in words ("what's the area between y=x² and
  y=x", "how fast is x³ changing?") and the model turns it into a solver command. It's
  **local-first**: clean commands never touch the network; the model is only called
  when the local parser can't handle your phrasing. It shows *how it read you*
  alongside the answer, and pauses for a confirm when it's unsure.
- **📷 Photo** — snap a picture; Claude vision transcribes the problem into the
  solver's grammar, you confirm *exactly what it read*, then SymPy solves + verifies.

In every case the model only handles **language** — it never does the math, so the
reliability guarantee is unchanged.

Your key is stored in **your browser only** (`localStorage`) and sent **only to
Anthropic**, straight from your browser — there is no server in between, and nothing
is uploaded or stored. Clear it anytime from the 🔑 panel. Get a key from the
[Anthropic console](https://console.anthropic.com/settings/keys).

> _Coming next: step-by-step tutoring, and word-problem setup — same rule, the CAS
> still does and verifies all the math. See [`docs/specs`](docs/specs) for the design._

## Architecture

- `index.html` — the whole UI (inline CSS/JS). Loads KaTeX (math rendering) and Plotly
  (graphs) from CDNs with SRI. AI calls go **browser-direct** to the Anthropic API with
  the visitor's own key — no backend.
- **The solve worker** — SymPy runs in a Web Worker, not on the main thread, so a slow
  computation can't freeze the tab and can be killed on timeout. The worker fetches
  Pyodide itself and re-verifies the same SHA-384 hash the `<script integrity>`
  attribute would have enforced, before evaluating a byte of it — moving work off the
  main thread shouldn't cost the supply-chain guarantee.
- `solver.py` — the entire engine (parse → compute → verify → steps → plot spec).
  Pure Python (SymPy only); the same file runs under CPython and in the browser via
  Pyodide, so there is one source of truth. `solve_json(query)` is the entry point.

Fully static — no server, no database, nothing stored. First load fetches the
~15 MB engine once (a few seconds), then it's cached.

## Run it locally

It's a static site — serve the folder over http (ES modules + Pyodide need http,
not `file://`):

```bash
python3 -m http.server 8000
# then open http://localhost:8000
```

## Correctness

`solver.py` is a single source of truth: the **same** code runs the checks in CPython
and the verifier live in the browser, so a green run here is the same verifier a user
gets — no drift between what's tested and what ships.

`test_solver.py` is the committable harness, **1019 checks, all passing**:

- **81 curated known-answer cases**, compared by SymPy equivalence rather than string
  match (indefinite integrals are checked by differentiating back to the integrand).
- **A deterministic 800-problem fuzz** asserting the verifier never *falsely flags* a
  correct result.
- **A rejection suite** — the direction that actually matters. The two above only prove
  the verifier says *yes* to right answers, which a function that returns `True`
  unconditionally would also pass. This one feeds **every** `_verify_*` a deliberately
  wrong answer and asserts it says **no**: derivatives off by `1e-4`, a wrong
  determinant, an understated rank, a matrix that isn't really in RREF, a rank-
  destroying "row reduction", a wrong mixed second partial, a sample variance quoted
  where the population figure belongs, a wrong residue, a bad unit conversion, a
  conversion between incompatible dimensions. Each was confirmed non-vacuous by
  stubbing its verifier to always accept and watching the suite fail.
- **A parse-intent suite** pinning the ambiguous-notation cases (`sin x cos x`, `e^2x`,
  `1/2x`) as flagged and unambiguous ones as clean, plus assertions that no internal
  exception text (`SyntaxError`, `TypeError`, tracebacks) can reach a user.

```bash
python3 -m venv venv && ./venv/bin/pip install sympy
./venv/bin/python test_solver.py     # -> RESULT: ALL PASS
```

## Deploy

Any static host works — it's just files. For GitHub Pages, serve the repo root. For
Vercel, deploy as a static project (`solver.py` sits at the root and is served as a
static asset, **not** an `/api` function).

## Positioning

Who this is for, what it is genuinely best at, and the weaknesses it hasn'''t solved
(the BYOK wall, the cold start, the absence of a moat): [docs/POSITIONING.md](docs/POSITIONING.md).

## License

[MIT](LICENSE) © Carter Monaco.
