// Serverless OCR endpoint (Vercel Node function).
//
// Reads a photo of a math problem with Claude vision and returns the problem
// transcribed into the solver's command grammar. It NEVER solves — the browser's
// SymPy engine does that. The model only performs perception (reading the image).
//
// Protections (the endpoint is public and backed by a paid key):
//   - POST only, image-size cap, small max_tokens, one vision model
//   - best-effort per-IP rate limit + global daily ceiling (in-memory)
//   - the real backstop is a monthly spend limit set on the Anthropic key
//
// Requires env var ANTHROPIC_API_KEY (set in Vercel project settings).

const MODEL = "claude-sonnet-5"; // swap to "claude-haiku-4-5-20251001" for ~5x cheaper
const MAX_TOKENS = 500;
const MAX_IMAGE_BYTES = 5 * 1024 * 1024; // 5 MB decoded
const PER_IP_MAX = 15; // requests
const PER_IP_WINDOW_MS = 60 * 1000; // per 60s
const DAILY_MAX = 400; // total reads/day (cost ceiling backstop)

const SYSTEM =
  "You transcribe a single math problem from an image into a calculus-solver " +
  "command. You do NOT solve it. Reply with ONLY one JSON object, no prose, no code fences.";

const INSTRUCTION =
  "Read the math problem in this image and output JSON:\n" +
  '{"found": true|false, "latex": "<problem as LaTeX for display>", ' +
  '"query": "<problem as ONE solver command>", "note": "<caveat if unclear, else empty>"}\n' +
  "The query MUST use this grammar: 'derivative of EXPR', 'second derivative of EXPR', " +
  "'integrate EXPR', 'integrate EXPR from A to B', 'limit of EXPR as x -> A', " +
  "'taylor series of EXPR', 'solve EQUATION', 'critical points of EXPR', " +
  "'tangent line to EXPR at x=A', 'factor EXPR', 'simplify EXPR', or just 'EXPR'.\n" +
  "Notation in query: ^ powers, * multiply, sqrt(), sin/cos/tan/ln/log/exp, pi, e, oo=infinity, |x|=abs.\n" +
  "If no math problem is present, found=false. Never solve; only transcribe.";

// --- best-effort in-memory limiters (per warm instance) ---
const ipHits = new Map(); // ip -> [timestamps]
let dayKey = "";
let dayCount = 0;

function rateLimited(ip) {
  const now = Date.now();
  const arr = (ipHits.get(ip) || []).filter((t) => now - t < PER_IP_WINDOW_MS);
  arr.push(now);
  ipHits.set(ip, arr);
  if (ipHits.size > 5000) ipHits.clear(); // crude memory guard
  return arr.length > PER_IP_MAX;
}

function dailyExceeded() {
  const today = new Date().toISOString().slice(0, 10);
  if (today !== dayKey) {
    dayKey = today;
    dayCount = 0;
  }
  dayCount += 1;
  return dayCount > DAILY_MAX;
}

module.exports = async (req, res) => {
  res.setHeader("Content-Type", "application/json");
  if (req.method !== "POST") {
    res.statusCode = 405;
    return res.end(JSON.stringify({ ok: false, error: "POST only." }));
  }
  if (!process.env.ANTHROPIC_API_KEY) {
    res.statusCode = 500;
    return res.end(JSON.stringify({ ok: false, error: "Server not configured (no API key)." }));
  }

  const ip =
    (req.headers["x-forwarded-for"] || "").split(",")[0].trim() ||
    req.headers["x-real-ip"] ||
    "unknown";

  if (rateLimited(ip)) {
    res.statusCode = 429;
    return res.end(JSON.stringify({ ok: false, error: "Too many photos, too fast — wait a moment and retry." }));
  }
  if (dailyExceeded()) {
    res.statusCode = 429;
    return res.end(JSON.stringify({ ok: false, error: "Daily photo limit reached. Try again tomorrow, or type the problem in." }));
  }

  let body = req.body;
  if (typeof body === "string") {
    try { body = JSON.parse(body); } catch { body = {}; }
  }
  let image = (body && body.image) || "";
  let mediaType = (body && body.mediaType) || "image/jpeg";
  // accept a full data URL too
  const m = /^data:(image\/[a-zA-Z+]+);base64,(.*)$/.exec(image);
  if (m) { mediaType = m[1]; image = m[2]; }
  if (!image) {
    res.statusCode = 400;
    return res.end(JSON.stringify({ ok: false, error: "No image provided." }));
  }
  const approxBytes = Math.floor((image.length * 3) / 4);
  if (approxBytes > MAX_IMAGE_BYTES) {
    res.statusCode = 413;
    return res.end(JSON.stringify({ ok: false, error: "Image too large — try a smaller / more cropped photo." }));
  }

  try {
    const apiRes = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "x-api-key": process.env.ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: MODEL,
        max_tokens: MAX_TOKENS,
        system: SYSTEM,
        messages: [
          {
            role: "user",
            content: [
              { type: "image", source: { type: "base64", media_type: mediaType, data: image } },
              { type: "text", text: INSTRUCTION },
            ],
          },
        ],
      }),
    });

    if (!apiRes.ok) {
      const detail = await apiRes.text();
      res.statusCode = 502;
      return res.end(JSON.stringify({ ok: false, error: "Vision service error.", status: apiRes.status, detail: detail.slice(0, 300) }));
    }

    const data = await apiRes.json();
    let text = ((data.content && data.content[0] && data.content[0].text) || "").trim();
    // strip accidental code fences
    if (text.startsWith("```")) {
      const s = text.indexOf("{");
      const e = text.lastIndexOf("}");
      if (s >= 0 && e > s) text = text.slice(s, e + 1);
    }
    let parsed;
    try { parsed = JSON.parse(text); }
    catch { parsed = null; }

    if (!parsed || parsed.found === false || !parsed.query) {
      res.statusCode = 200;
      return res.end(JSON.stringify({
        ok: true, found: false,
        error: (parsed && parsed.note) || "Couldn't find a clear math problem in that photo.",
      }));
    }

    res.statusCode = 200;
    return res.end(JSON.stringify({
      ok: true, found: true,
      query: String(parsed.query),
      latex: parsed.latex ? String(parsed.latex) : "",
      note: parsed.note ? String(parsed.note) : "",
    }));
  } catch (err) {
    res.statusCode = 502;
    return res.end(JSON.stringify({ ok: false, error: "Could not reach the vision service: " + err.message }));
  }
};
