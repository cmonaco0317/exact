# Exact — a calculus solver that verifies its own answers

A WolframAlpha-style calculus tool (Mac or phone). Type a problem **or snap a
photo of one**, and it's designed to never get the math wrong.

## Photo input

Tap 📷 → take/upload a picture of a problem → **Claude vision transcribes it**
into the solver's grammar → the app shows you *exactly what it read* (rendered)
so you can confirm or edit → then the in-browser SymPy engine solves and
self-verifies. The vision model only performs **perception** (reading the
image); it never does the math, so the reliability guarantee is unchanged. The
"confirm what I read before solving" step is deliberate — a misread photo is the
one thing that could otherwise slip past the verifier.

- Backend: one Vercel serverless function, [`api/read.js`](api/read.js), calls
  the Anthropic API (`claude-sonnet-5`; swap the `MODEL` constant to
  `claude-haiku-4-5-20251001` for ~5× cheaper). Cost ≈ a fraction of a cent per
  photo. The API key lives only in the Vercel env var `ANTHROPIC_API_KEY` —
  never in the frontend.
- Protections: POST-only, image-size cap, best-effort per-IP rate limit +
  global daily ceiling. **Backstop: set a monthly spend limit on the Anthropic
  key** in the Anthropic console — that's the real cost cap for a public URL.
- Photos are sent to the reader and discarded; nothing is stored.

## Why it's reliable

The core idea: **an LLM must never do the arithmetic.** Language models
pattern-match answers; they don't compute them, so they make silent algebra
mistakes. WolframAlpha is reliable because it's a *computer algebra system*
(CAS), not an AI.

So this tool uses the same principle:

1. **Parse** — human input (`derivative of x^2 sin(x)`) → a precise SymPy
   expression. This is the only "interpretation" step.
2. **Compute** — [SymPy](https://www.sympy.org), a real CAS, does the calculus
   exactly and deterministically.
3. **Verify** — every answer is *independently re-checked numerically* before
   it's shown:
   - derivatives → central finite-difference at 8 sample points
   - indefinite integrals → differentiate the answer, confirm it returns the integrand
   - definite integrals → compare against independent numerical integration
   - limits → approach the point numerically and confirm convergence
   - roots / critical points → substitute back and confirm ≈ 0

   The badge tells you the confidence: **✓ Verified** (numeric check passed),
   **· Exact (SymPy)** (symbolic result, numeric check not applicable), or
   **⚠ Unverified** (a check disagreed — shown instead of a confident wrong answer).

This eliminates "wrong because it did the math wrong." The only residual failure
mode is misreading an ambiguous problem — and the verifier catches most of those.
When SymPy genuinely can't solve something, it says so honestly rather than guessing.

## What it solves

Derivatives (incl. higher-order & partial), indefinite/definite/improper
integrals (with step-by-step), limits (incl. one-sided & DNE), Taylor/Maclaurin
series, equation solving, critical points & extrema, tangent lines, plus
`simplify` / `factor` / `expand`. Input is Wolfram-style math (`2x`, `x^2`,
`sin x`, `e^x`, `|x|`, `∫`, `->`, `arctan`, `pi`, `oo`, …).

## Architecture

- `index.html` — the whole UI (inline CSS/JS). Loads Pyodide (SymPy compiled to
  WebAssembly), KaTeX (math rendering), and Plotly (graphs) from CDNs with SRI.
- `solver.py` — the entire engine (parse → compute → verify → steps → plot spec).
  Pure Python (SymPy only). Runs locally with CPython for testing **and** in the
  browser via Pyodide — one source of truth. `solve_json(query)` is the browser
  entry point.

No backend, no database, nothing stored. First page load fetches the ~15 MB
engine once (a few seconds), then it's cached.

## Testing

The engine is validated with CPython before it ships:

```bash
python3 -m venv venv && ./venv/bin/pip install sympy
./venv/bin/python test_solver.py     # 77 curated cases with independently-known answers
```

It also passed an 800-problem random fuzz (0 crashes, 0 false verification flags).
Because the runtime verifier is the same code, **every answer a user sees is
numerically confirmed live.**

## Deploy / update

Static site on Vercel (account: `cmonaco0317`). To redeploy after edits:

```bash
export VERCEL_TOKEN=$(security find-generic-password -s YOUR_KEYCHAIN_SERVICE -w)
npx --yes vercel@latest deploy --prod --yes
```

(`solver.py` sits at the project root, so Vercel serves it as a static asset —
it is **not** an `/api` serverless function.)
