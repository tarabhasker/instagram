# FastAPI service for LLM-only Instagram-style caption suggestions (no heavy ML).
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os, re, json, requests

# ---------- env ----------
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE    = os.getenv("OPENROUTER_BASE", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL   = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3-0324:free")

# ---------- app ----------
app = FastAPI(title="PhotoFeed AI (LLM-only)", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://<your-frontend>.vercel.app",   # <-- replace with your real Vercel domain
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True, "mode": "llm-only"}

# ---------- utils ----------
_STOP = {"a","an","the","and","or","with","in","on","of","to","for","my","me","our","your",
         "we","us","you","his","her","their","at","by","from","as"}
def _tokens(s: str): return [w for w in re.findall(r"[a-z']+", (s or "").lower())]
def sanitize_vibe(s: str) -> str:
    toks = [w for w in _tokens(s) if w not in _STOP]
    return " ".join(toks[:6])

def _to_tag(s: str) -> str:
    return "#" + re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")

def clean_caption(text: str) -> str:
    if not text: return text
    t = re.sub(r"\s+", " ", text).strip()
    return t[0].upper() + t[1:] if t else t

def call_llm_captions(grounding: str, vibe: Optional[str], n_variants: int = 3) -> dict:
    """Return {"captions": [...], "hashtags": [...]} via OpenRouter; fallback locally if no key."""
    tone = sanitize_vibe(vibe or "")

    if not OPENROUTER_API_KEY:
        base = clean_caption(f"{(tone or '').strip()}, {grounding}".strip(", ")) or "Captured the moment."
        tags = list({t for t in (_to_tag(w) for w in tone.split()) if t and t != "#"})
        return {"captions": [base][:n_variants], "hashtags": list(tags)[:6]}

    system = (
        "You write Instagram-ready captions that feel human and specific.\n"
        "Use the grounding for facts; do NOT describe literally or repeat the user's style words.\n"
        "Never include these words: photo, picture, image, man, woman, couple, laying, lying, grass.\n"
        "No quotes or hashtags inside captions.\n"
        f"Create {n_variants} distinct options across lengths: "
        "1 micro (2–4 words), 2 short (≤12), 1 mid (12–22), 1 longer (22–40).\n"
        'Return STRICT JSON: {"captions":["..."], "hashtags":["#..."]}\n'
        "Hashtags: 5–8, specific, lowercase kebab-case, no duplicates."
    )
    user = {"grounding": grounding, "style_prompt": tone}

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("OPENROUTER_REFERRER", "http://localhost:5173"),
        "X-Title": os.getenv("OPENROUTER_TITLE", "Mini Instagram AI"),
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user)},
        ],
        "temperature": 0.65,
        "top_p": 0.9,
    }
    try:
        r = requests.post(f"{OPENROUTER_BASE}/chat/completions", headers=headers, json=payload, timeout=45)
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        m = re.search(r"\{.*\}", text, re.S)
        obj = json.loads(m.group(0)) if m else {"captions": [text.strip()], "hashtags": []}
        caps = [clean_caption(c) for c in obj.get("captions", []) if c][:n_variants]
        tags = obj.get("hashtags") or []
        tags = [t if str(t).startswith("#") else _to_tag(str(t)) for t in tags]
        # unique + max 8
        tags = list(dict.fromkeys([t for t in tags if t and t != "#"]))[:8]
        return {"captions": caps or ["Captured the moment."], "hashtags": tags}
    except Exception:
        base = clean_caption(f"{tone}, {grounding}".strip(", ")) or "Captured the moment."
        vibe_tags = [_to_tag(w) for w in (tone or "").split()[:6]]
        return {"captions": [base], "hashtags": list(dict.fromkeys([t for t in vibe_tags if t]))[:6]}

# ---------- schemas ----------
class SuggestReq(BaseModel):
    imageUrl: str
    prompt: Optional[str] = None
    n_variants: int = 3

# ---------- endpoints ----------
@app.post("/ai/suggest")
def suggest(req: SuggestReq):
    # LLM-only mode: we don't fetch/process the image; we trust the user's prompt
    # If you want a little grounding, derive something simple from the URL:
    filename = os.path.basename(req.imageUrl).split("?")[0]
    guess = " ".join(w for w in re.findall(r"[a-zA-Z]+", filename)) or "a moment"
    grounding = guess if req.prompt is None else req.prompt
    out = call_llm_captions(grounding=grounding, vibe=req.prompt, n_variants=max(1, min(5, req.n_variants)))
    return {
        "analysis": {"note": "LLM-only mode; no image analysis performed"},
        "grounding": grounding,
        "captions": out["captions"],
        "hashtags": out["hashtags"],
    }
