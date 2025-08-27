# ai_service.py
# FastAPI service for image analysis + Instagram-style caption suggestions.

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Tuple
from PIL import Image
import requests, io, os, re, json
import torch
import numpy as np

# ------------------- Env / Config -------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE    = os.getenv("OPENROUTER_BASE", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL   = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3-0324:free")
USE_BLIP           = os.getenv("USE_BLIP", "false").lower() == "true"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------- App ----------------------
app = FastAPI(title="PhotoFeed AI", version="0.5")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://<your-frontend>.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True, "use_blip": USE_BLIP}

# ------------------- CLIP (analysis) ----------
import open_clip
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
clip_model = clip_model.to(device).eval()
clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")

OBJECT_LABELS = [
    "person","girl","boy","man","woman","child","couple",
    "dog","cat","bird","horse","bicycle","motorcycle","car","bus","train","boat","airplane",
    "bench","swing","kite","ball","backpack","umbrella","handbag","camera","phone","laptop",
    "flower","tree","leaf","mountain","river","ocean","beach","sunset","sunrise","sky","cloud",
]
SCENE_LABELS = [
    "park","playground","forest","garden","street","city","rooftop","cafe","bridge",
    "lake","waterfall","desert","snow","stadium","museum","temple","beach","cliff"
]
GEAR_BAN = {"camera", "phone", "laptop", "backpack", "handbag"}
PERSON_WORDS = {"person","girl","boy","man","woman","child"}

# ------------------- Utils -------------------
def fetch_image(url: str) -> Image.Image:
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")

def clip_rank(image: Image.Image, candidates: List[str], template: str = "a photo of {}") -> List[Tuple[str, float]]:
    img_t = clip_preprocess(image).unsqueeze(0).to(device)
    texts = [template.format(c) for c in candidates]
    text_t = clip_tokenizer(texts)
    with torch.no_grad():
        img_feat = clip_model.encode_image(img_t)
        txt_feat = clip_model.encode_text(text_t)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        sim = (img_feat @ txt_feat.T).squeeze(0).float().cpu().numpy()
    return sorted(zip(candidates, sim.tolist()), key=lambda x: x[1], reverse=True)

# ---- Caption quality helpers ------------------------------------
_STOP = {
    "a","an","the","and","or","with","in","on","of","to","for","my","me","our","your",
    "we","us","you","his","her","their","at","by","from","as"
}
_REL_WORDS = {"husband","wife","boyfriend","girlfriend","fiance","fiancé","fiancee","partner"}

def _tokens(s: str):
    return [w for w in re.findall(r"[a-z']+", (s or "").lower())]

def sanitize_vibe(s: str) -> str:
    """Keep only mood/style words; drop pronouns/relationship words."""
    toks = [w for w in _tokens(s) if w not in _STOP and w not in _REL_WORDS]
    return " ".join(toks[:6])

def prompt_echo(caption: str, prompt: str) -> bool:
    """True if caption is mostly the prompt."""
    a = set([w for w in _tokens(caption) if w not in _STOP])
    b = set([w for w in _tokens(prompt) if w not in _STOP])
    if not a or not b: return False
    j = len(a & b) / max(1, len(a | b))
    starts = caption.lower().startswith(prompt.lower()[:20])
    return j >= 0.45 or starts

def style_score(c: str) -> float:
    """Light heuristic: avoid literal words, reward concise, poetic phrasing."""
    t = c.lower()
    bad = any(x in t for x in ["photo","picture","image","man","woman","couple","laying","lying","grass"])
    n = len(t.split())
    score = 0.0
    if 4 <= n <= 14: score += 0.4
    if any(w in t for w in ["twilight","dusk","hush","quiet","soft","fog","whisper","evening","still"]):
        score += 0.1
    if bad: score -= 1.0
    return score

def rank_with_clip(img_feat, captions: List[str]) -> List[str]:
    if not captions: return []
    text_t = clip_tokenizer(captions)
    with torch.no_grad():
        txt = clip_model.encode_text(text_t)
        txt = txt / txt.norm(dim=-1, keepdim=True)
        sims = (img_feat @ txt.T).squeeze(0).float().cpu().numpy().tolist()
    scored = [(c, s + style_score(c)) for c, s in zip(captions, sims)]
    scored.sort(key=lambda x: x[1], reverse=True)
    uniq, seen = [], set()
    for c, _ in scored:
        k = c.strip().lower()
        if k and k not in seen:
            seen.add(k); uniq.append(c)
    return uniq

# ------------------- (Optional) BLIP ----------
if USE_BLIP:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device).eval()

def collapse_repeats(text: str) -> str:
    if not text: return text
    for n in (3, 2, 1):
        pattern = re.compile(r"\b((?:\w+\s+){%d}\w+)(?:\s+\1)+" % (n-1), re.IGNORECASE)
        while True:
            new_text = pattern.sub(r"\1", text)
            if new_text == text: break
            text = new_text
    return re.sub(r"\s+", " ", text).strip()

def clean_caption(text: str) -> str:
    if not text: return text
    t = collapse_repeats(text)
    return t[0].upper() + t[1:] if t else t

def blip_describe(image: Image.Image, hint: Optional[str]) -> str:
    """Short literal description for grounding (only if USE_BLIP=True)."""
    if not USE_BLIP:
        return ""
    inputs = {"images": image}
    if hint:
        inputs["text"] = hint
    t = blip_processor(**inputs, return_tensors="pt").to(device)
    with torch.no_grad():
        out = blip_model.generate(
            **t, max_new_tokens=18, num_beams=5,
            no_repeat_ngram_size=3, repetition_penalty=1.2
        )
    raw = blip_processor.decode(out[0], skip_special_tokens=True)
    return clean_caption(raw)

# ------------------- LLM (OpenRouter) ---------
def _to_tag(s: str) -> str:
    return "#" + re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")

def call_llm_captions(grounding: str, vibe: Optional[str], objects=None, scenes=None, n_variants: int = 3) -> dict:
    """
    Call OpenRouter chat model to create captions.
    Return {"captions": ["..."], "hashtags": ["#..."]} with graceful fallbacks.
    """
    # Fallback when no key (or offline)
    if not OPENROUTER_API_KEY:
        base = clean_caption(f"{(vibe or '').strip()}, {grounding}".strip(", "))
        base = base[:90] if base else "Captured the moment."
        return {"captions": [base], "hashtags": []}

    tone = sanitize_vibe(vibe or "")

    system = (
        "You write Instagram-ready captions that feel human and specific.\n"
        "Use grounding/objects/scenes for facts; do NOT describe literally or repeat the user's style words.\n"
        "Never include these words: photo, picture, image, man, woman, couple, laying, lying, grass.\n"
        "No quotes or hashtags inside captions.\n"
        f"Create {n_variants} distinct options across lengths: "
        "1 micro (2–4 words), 2 short (≤12), 1 mid (12–22), 1 longer (22–40).\n"
        'Return STRICT JSON: {"captions":["..."], "hashtags":["#..."]}\n'
        "Hashtags: 5–8, specific, lowercase kebab-case, no duplicates."
    )

    user = {
        "grounding": grounding,
        "objects": objects or [],
        "scenes": scenes or [],
        "style_prompt": tone
    }

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
        resp = requests.post(f"{OPENROUTER_BASE}/chat/completions",
                             headers=headers, json=payload, timeout=45)
        if not resp.ok:
            try:
                err = resp.json()
                msg = err.get("error", {}).get("message") or str(err)
            except Exception:
                msg = resp.text
            raise RuntimeError(f"OpenRouter HTTP {resp.status_code}: {msg}")

        data = resp.json()
        if isinstance(data, dict) and "error" in data:
            raise RuntimeError(f"OpenRouter error: {data['error']}")

        text = data["choices"][0]["message"]["content"]
        m = re.search(r"\{.*\}", text, re.S)
        obj = json.loads(m.group(0)) if m else {"captions": [text.strip()], "hashtags": []}

        captions = [clean_caption(c) for c in obj.get("captions", []) if c]
        if not captions:
            captions = [clean_caption(grounding) or "Captured the moment."]
        tags = obj.get("hashtags") or []
        tags = [
            t if isinstance(t, str) and t.startswith("#")
            else _to_tag(str(t))
            for t in tags
        ]
        tags = list(dict.fromkeys([t for t in tags if t and t != "#"]))[:8]
        return {"captions": captions[:n_variants], "hashtags": tags}

    except Exception:
        base = clean_caption(f"{(tone or '').strip()}, {grounding}".strip(", "))
        base = base[:90] if base else "Captured the moment."
        vibe_tags = [_to_tag(w) for w in (tone or "").split()[:3]]
        return {"captions": [base], "hashtags": list(dict.fromkeys([t for t in vibe_tags if t]))[:6]}

# ------------------- Schemas ------------------
class AnalyzeReq(BaseModel):
    imageUrl: str
    top_k: int = 6

class SuggestReq(BaseModel):
    imageUrl: str
    prompt: Optional[str] = None
    top_k: int = 6
    n_variants: int = 3

# ------------------- Internal helpers ------------------
def _analyze_image(img: Image.Image, top_k: int = 6):
    obj = clip_rank(img, OBJECT_LABELS)[:top_k]
    scn = clip_rank(img, SCENE_LABELS)[:top_k]
    return {
        "objects": [{"label": l, "score": float(s)} for l, s in obj],
        "scenes":  [{"label": l, "score": float(s)} for l, s in scn],
    }

# ------------------- Endpoints ----------------
@app.post("/ai/analyze")
def analyze(req: AnalyzeReq):
    img = fetch_image(req.imageUrl)
    return _analyze_image(img, req.top_k)

@app.post("/ai/suggest")
def suggest(req: SuggestReq):
    # 1) Fetch & analyze once
    img = fetch_image(req.imageUrl)
    analysis = _analyze_image(img, req.top_k)

    # also get image embedding for later ranking
    img_t = clip_preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        img_feat = clip_model.encode_image(img_t)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

    # Extract and re-rank objects (prefer subject over gear)
    obj_labels = [x["label"] for x in analysis["objects"]]
    obj_pref = [o for o in obj_labels if o not in GEAR_BAN] + [o for o in obj_labels if o in GEAR_BAN]
    scene = analysis["scenes"][0]["label"] if analysis["scenes"] else None

    # 2) Grounding text (neutral)
    if USE_BLIP:
        hint = f"a {(req.prompt or '').strip()} photo" if req.prompt else None
        grounding = blip_describe(img, hint)
    else:
        parts = []
        core = [o for o in obj_pref[:3] if o not in GEAR_BAN]
        persons = [o for o in core if o in PERSON_WORDS]
        others  = [o for o in core if o not in PERSON_WORDS]

        subj = None
        if len(persons) >= 2:
            subj = "two people"
        elif len(persons) == 1 and others:
            subj = f"someone with a {others[0]}"
        elif len(persons) == 1:
            subj = "someone"
        elif len(others) >= 2:
            subj = f"a {others[0]} and a {others[1]}"
        elif len(others) == 1:
            subj = f"a {others[0]}"

        if subj: parts.append(subj)
        if scene: parts.append(f"in a {scene}")
        grounding = clean_caption(", ".join(parts)) or "a scene"

    # 3) LLM: get N caption options + base hashtags
    objs = [x["label"] for x in analysis["objects"]][:5]
    scns = [x["label"] for x in analysis["scenes"]][:3]
    out = call_llm_captions(
        grounding, req.prompt, objs, scns,
        n_variants=max(1, min(5, req.n_variants))
    )

    # Filter out any caption that echoes the user's prompt
    caps = [clean_caption(c) for c in (out.get("captions") or []) if c]
    if req.prompt:
        caps = [c for c in caps if not prompt_echo(c, req.prompt)]

    # If everything got filtered, fallback to grounded options
    if not caps:
        caps = [clean_caption(grounding),
                "quiet hours, softer light",
                "notes from the in-between"]

    # Rank with CLIP + style and keep top N
    ranked = rank_with_clip(img_feat, caps)
    out["captions"] = ranked[:max(1, min(5, req.n_variants))]

    # 4) Enrich hashtags with object/scene add-ons (max 8 total)
    extra = []
    for w in [objs[0] if objs else None, scene]:
        if not w: continue
        tag = _to_tag(w)
        if tag not in out["hashtags"]:
            extra.append(tag)
    tags = (out["hashtags"] + extra)[:8]

    return {
        "analysis": analysis,
        "grounding": grounding,
        "captions": out["captions"],   # array of strings
        "hashtags": tags               # array of strings starting with '#'
    }
