# Positioning

Who this is for, who it isn't, and what it would have to become to matter
commercially. Written down because building without naming the user is how you end
up with a capability list instead of a product.

This is a **portfolio-first** project. There is no pricing here and no go-to-market
plan, deliberately — see [Not covered](#not-covered).

## The market as of mid-2026

Two camps, and the gap between them is the whole thesis.

**Engine-backed solvers** (Wolfram ~96%, Photomath ~94% on head-to-head accuracy
testing) are reliable because a computer-algebra system computes the answer. They are
also opaque: you get a result and are asked to trust it.

**Chat-style solvers** (ChatGPT ~88%) explain beautifully and are confidently wrong on
a real fraction of problems. Buyer's guides now say so outright — when the two
disagree, trust the engine.

The synthesis everyone is groping toward is *CAS computes, LLM explains, and something
verifies*. That's this repo's bet, and it is no longer a novel one: AskSia (2M+
students) advertises a symbolic verification pass and claims 98% accuracy. **"We
verify" is becoming table stakes.**

What is *not* taken is **transparent, auditable, free** verification. Every rival's
self-check is a black box asking for trust. This one publishes the checks, names the
one operation it can't verify and why, and ships the numbers behind each verdict.
That's a defensible position for an open-source project specifically — a funded cloud
product has little reason to expose its own error surface.

## Primary ICP — "the anxious self-checker"

A STEM undergrad or AP/IB student in calculus, linear algebra, or diff-eq, doing
**graded** work, who has been burned or warned that chat solvers are confidently
wrong.

- They are not trying to cheat. They did the problem and need to know they're right.
- They value: free, no account, works offline in a library or exam setting, and above
  all *legible* trust — being told which guarantee they got.
- **The killer feature for them is not "solve it."** It's `Check my work` and
  `Guide me`, which is precisely why both are keyless and offline. A tool that only
  hands over answers trains dependence; one that verifies your reasoning trains skill.

Secondary: teachers and tutors who want something that verifies and guides without
handing over final answers.

## Anti-ICP — the photo-answer user

Someone who wants to snap a picture of homework and receive the answer. Photomath owns
them, it's free, it's one tap, and they place **zero** value on verification. Building
for them means competing on convenience against an incumbent, from behind, on the one
axis where this design is structurally slower. Don't.

## What this repo is genuinely best at

Ranked by how hard it would be for a competitor to copy:

1. **Verifying work the user already did.** A CAS is structurally suited to it and
   almost nobody does it. It also sidesteps the academic-integrity problem dragging on
   the whole category.
2. **Guided steps that can't mis-correct.** The verdict is a computation, so a correct
   step is always accepted. An LLM tutor cannot promise that.
3. **Publishing its own error surface.** 62/63 operations independently checked, the
   1 that isn't named explicitly, the weaker check (antiderivatives) called out as
   SymPy checking SymPy rather than buried.
4. **Provably local.** No account, no upload, works offline once cached.

## The honest weaknesses

- **The BYOK wall.** Photo, plain-English and the narrative tutor need the visitor's
  own Anthropic key. The overlap between "needs a math solver" and "will create an API
  key" is small. Mitigated but not solved: the two most differentiating modes are now
  free and keyless, so the wall no longer sits in front of the best part.
- **Cold start.** ~13 MB before the engine is usable. Wolfram answers before this
  finishes booting. The 4.35 MB plot library has been moved off the critical path;
  Pyodide + SymPy is the irreducible remainder of "runs in your browser".
- **No moat.** SymPy + Pyodide + KaTeX + a router. The original IP is the verifier and
  the check-my-work logic. There is no data advantage, no network effect, no
  distribution.
- **The setup gap.** Cross-checking a model-chosen setup by independent agreement
  raises the floor; two readings can still be wrong the same way.

## Not covered

Deliberately absent, and it's a scoping decision rather than an oversight:

- **Pricing.** There is nothing to price. Zero customers, and the stated goal is a
  repo worth reading.
- **Distribution.** This is the binding constraint on every project here, and it isn't
  an engineering problem — it's the author talking to users. No amount of work inside
  the repo substitutes for it, and pretending otherwise in a strategy doc would be the
  same kind of overclaim the README was cleaned of.

If the goal ever changes to revenue, those two become the whole job, and everything
above is the input to them rather than a replacement for them.
