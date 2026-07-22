# Exact — a calculus solver that verifies its own answers

Type a calculus problem (or snap a photo of one) and get an answer that is
**computed by a real computer-algebra system and numerically re-checked before it's
shown** — so the math is never guessed. Runs entirely in your browser.

**▶ [Live demo](https://calc-solver.vercel.app)** · desktop or phone.

## Why it's reliable

Language models pattern-match answers; they don't compute them, so they make silent
algebra mistakes. WolframAlpha is reliable because it's a *computer-algebra system*
(CAS), not an AI. Exact uses the same principle, with one rule:

> **The LLM never does the math.** It only helps with *language* — reading a photo,
> and (soon) understanding plain-English problems and explaining answers. Every
> number is computed by SymPy and independently re-verified.

1. **Parse** — your input (`derivative of x^2 sin(x)`) → a precise SymPy expression.
2. **Compute** — [SymPy](https://www.sympy.org), a real CAS compiled to WebAssembly
   via [Pyodide](https://pyodide.org), does the calculus exactly, in your browser.
3. **Verify** — every answer is *independently re-checked numerically* before display:
   - derivatives → central finite-difference at several sample points
   - indefinite integrals → differentiate the answer, confirm it returns the integrand
   - definite integrals → compare against independent numerical integration
   - limits → approach the point numerically and confirm convergence
   - roots / critical points → substitute back and confirm ≈ 0

The badge states the confidence: **✓ Verified** (numeric check passed), **· Exact**
(symbolic, numeric check not applicable), or **⚠ Unverified** (a check disagreed —
shown honestly instead of a confident wrong answer). When SymPy genuinely can't solve
something, it says so rather than guessing.

## What it solves

Derivatives (higher-order & partial), indefinite/definite/improper integrals (with
steps), limits (one-sided & DNE), Taylor/Maclaurin series, equation solving, critical
points & extrema, tangent lines, and `simplify` / `factor` / `expand`. Input is
Wolfram-style math: `2x`, `x^2`, `sin x`, `e^x`, `|x|`, `->`, `arctan`, `pi`, `oo`, …

## Photo input & AI features — bring your own key

The verified solver is **100% free, works offline, and needs no account or key.**
Optional AI features run on your **own** Anthropic API key:

- **📷 Photo** — snap a picture; Claude vision transcribes the problem into the
  solver's grammar, you confirm *exactly what it read*, then SymPy solves + verifies.
  The model only performs perception — it never does the math, so the reliability
  guarantee is unchanged.

Your key is stored in **your browser only** (`localStorage`) and sent **only to
Anthropic**, straight from your browser — there is no server in between, and nothing
is uploaded or stored. Clear it anytime from the 🔑 panel. Get a key from the
[Anthropic console](https://console.anthropic.com/settings/keys).

> _Coming next: plain-English problem understanding, step-by-step tutoring, and
> word-problem setup — same rule, the CAS still does and verifies all the math.
> See [`docs/specs`](docs/specs) for the design._

## Architecture

- `index.html` — the whole UI (inline CSS/JS). Loads Pyodide (SymPy → WebAssembly),
  KaTeX (math rendering), and Plotly (graphs) from CDNs with SRI. AI calls go
  **browser-direct** to the Anthropic API with the visitor's own key — no backend.
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
and the verifier live in the browser, so **every answer a user sees is numerically
confirmed as it's shown.**

`test_solver.py` is the committable harness — **81 curated known-answer cases** (each
compared by SymPy equivalence, not string match; indefinite integrals are checked by
differentiating the answer back to the integrand) plus a deterministic **800-problem
fuzz** that asserts the verifier never falsely flags a correct result. 881 checks,
all passing:

```bash
python3 -m venv venv && ./venv/bin/pip install sympy
./venv/bin/python test_solver.py     # -> RESULT: ALL PASS
```

## Deploy

Any static host works — it's just files. For GitHub Pages, serve the repo root. For
Vercel, deploy as a static project (`solver.py` sits at the root and is served as a
static asset, **not** an `/api` function).

## License

[MIT](LICENSE) © Carter Monaco.
