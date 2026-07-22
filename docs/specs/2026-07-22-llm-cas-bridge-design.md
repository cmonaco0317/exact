# Exact — LLM × verified-CAS bridge (design)

_Status: approved 2026-07-22. Source of truth for the multi-phase build._

## One-line thesis

Bridge the gap between a computer-algebra system (SymPy, à la Wolfram Alpha) and an
LLM, keeping the best of both: **the LLM does language; the CAS does all math; an
independent verifier is the referee.** LLM proposes, CAS disposes, verifier judges.

This preserves Exact's moat — *every displayed answer is CAS-computed and numerically
re-verified* — while removing the CAS's weaknesses (rigid input, no teaching, no
word-problem setup).

## Non-negotiable invariant (the moat)

> The LLM never performs or alters arithmetic/symbolic computation. If the LLM's
> contribution cannot be independently checked, the result must not wear a
> ✓ Verified badge.

Break this and Exact becomes "another chatbot that's sometimes wrong." Every design
choice below is subordinate to it.

## Architecture

The existing CAS core (`solver.py`: parse → SymPy compute → numeric re-verify) is
**unchanged and authoritative**. Two LLM *language* layers wrap it:

```
        ┌─────────── LLM: UNDERSTAND (front) ───────────┐
input → │ messy text / photo / word problem → command(s)│
        └───────────────────────┬───────────────────────┘
                                ▼
       ╔═══════════ CAS CORE (unchanged) ══════════════╗
       ║  parse → SymPy computes → numeric verify       ║  ← moat
       ╚═══════════════════════┬════════════════════════╝
                                ▼
        ┌──────────── LLM: TEACH (back) ────────────────┐
        │  verified result → tutor explanation          │
        └────────────────────────────────────────────────┘
```

- **Understand** and **Extend** are the same front door: single-command translation
  vs. multi-step decomposition.
- **Teach** is the back door: it narrates a result that is already locked + verified.

### System principle: local-first, LLM-as-fallback

The regex router in `solver.py` handles clean input instantly, free, and offline.
The LLM fires **only** when the router misses (natural language / word problem) or the
user explicitly asks to be taught. No token or network round-trip is spent to
differentiate `x^2`.

## Deployment model — BYOK, fully static

Because we chose **bring-your-own-key**, the app returns to **pure static** and the
owner's API key leaves the public demo entirely:

- **`api/read.js` (serverless) is deleted.** Its "image → command" job moves
  client-side and becomes one modality of the Understand layer.
- The visitor's Anthropic key lives in **their** browser (`localStorage`), and the
  browser calls Anthropic **directly** via the `anthropic-dangerous-direct-browser-access`
  header (standard static BYOK pattern; the "danger" is only that the key is in the
  browser — fine when it is the user's own key).
- Hostable straight from the repo (GitHub Pages) or the existing Vercel URL.
- The full ✓-verified CAS core works for **everyone with no key, offline**. The LLM
  layers are pure enhancements. No key → the smart features show a quiet
  "add your Anthropic key to unlock" affordance, not a broken state.

## Layers

### Layer 1 — Understand (front translator)

- Local parse first; on miss, a cheap local classifier decides "clean command vs.
  natural language" and only then invokes the LLM.
- Photo OCR is the same path, image-modality in.
- **Always confirms** LLM-translated input with the existing "I read this as:
  [rendered math]" step before solving. The user is the check on *intent*; the
  verifier is the check on *math*.

### Layer 2 — Teach (tutor behind)

- On-demand "Explain like a tutor" button under any verified result (key required —
  teaching is the token-costly part).
- The LLM receives the **pinned verified answer** + SymPy's steps and explains
  *around* it (intuition, why this technique). It **cannot change the number**; the
  CAS answer stays authoritative and on-screen. Rendered as a distinct "Tutor" panel,
  clearly labelled *explanation*, not *computation*.

### Layer 3 — Extend (word problems → verified steps)

- The LLM decomposes an applied problem into (a) a stated setup and (b) a sequence of
  solver commands.
- **Each computation runs through the CAS + verifier** — every number is still
  verified.
- **The setup is NOT verifiable** and wears a distinct badge:
  **⚙ "Setup by AI — each computation verified, the setup is not."** (Confirmed for
  v1: shipping AI-set-up word problems under an explicitly partial-trust label is
  acceptable.)
- Scoped to standard Calc-I applied families: **related rates · optimization ·
  area/volume between curves · motion (position/velocity/acceleration)**. Anything
  outside → honest refusal, not a bluff.

## Trust model (the differentiator)

| State | Badge | Meaning |
|---|---|---|
| CAS computed + numerically re-checked | **✓ Verified** | the number is proven |
| CAS exact, not numerically checkable | **· Exact** | symbolic, trusted |
| LLM setup, computations verified | **⚙ Setup by AI** | _new_ — honest partial trust |
| couldn't verify | **⚠ Unverified** | says so plainly |

## Repo readiness (open-source)

- **No key in git, ever.** `.env.example`, `.gitignore` covers `.env` + `.vercel/` +
  `__pycache__/`, pre-commit secret scan, verify the live key lives only in Vercel/nowhere.
- `README.md` telling the "LLM understands + teaches, a real CAS proves" story, with
  the "works free & offline" hook and clear BYOK setup ("your key never leaves your
  browser except to Anthropic" + a "clear key" control).
- `LICENSE` = **MIT**.
- The existing **77 known-answer + 800 fuzz tests** become the repo's proof-of-correctness.

## Chosen defaults (revisit if wrong)

- **Tutor voice:** a sharp peer who's genuinely good at calc — intuitive, concise, no
  fluff; adapts to the problem.
- **Model:** default **Haiku** for translate (matched Sonnet in prior testing, cheap
  for the visitor), **Sonnet** for teach/extend (better teaching). Advanced users can
  switch; the visitor pays.
- **Extend scope:** related rates · optimization · area/volume between curves · motion.
  Outside → honest refusal.
- **Hosting:** keep the Vercel URL and enable GitHub Pages (both static).

## Phasing

- **Phase 0** — repo hardening + static/BYOK refactor: delete `api/read.js`, move the
  vision-OCR call client-side, add the BYOK key UI (`localStorage` + clear-key),
  `.gitignore`/`.env.example`, `git init`, `LICENSE`, README. No new math.
- **Phase 1 — Understand:** local-first classifier + LLM translate fallback + confirm
  step; photo path folded in.
- **Phase 2 — Teach:** on-demand tutor panel, answer pinned.
- **Phase 3 — Extend:** word-problem decomposition, per-step verification, ⚙ badge,
  scoped refusals.

## Open risks / to-validate

- **Teach answer-pinning:** ensure the tutor text can't silently contradict the
  pinned number (frame around it; consider a lightweight consistency check).
- **Extend setup honesty:** the ⚙ badge must be visually unmistakable vs. ✓.
- **BYOK key hygiene:** key in `localStorage` has mild XSS exposure; app already
  loads CDN scripts with SRI. Document clearly; never log the key; offer clear-key.
- **Classifier false-positives:** must not send clean expressions to the LLM (cost +
  latency). Bias toward local-parse-first.
