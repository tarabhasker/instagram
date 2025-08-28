# ai_service.py
# FastAPI service: lightweight CLIP analysis + LLM captions with safe fallbacks.

from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Tuple, Dict
from PIL import Image
import requests, io, os, re, json, math
import torch

# -------------------- Env / Config --------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --- OpenRouter (LLM) ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
OPENROUTER_BASE    = os.getenv("OPENROUTER_BASE", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL   = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")

# --- CLIP backend (lightweight) ---
# RN50 is smallest; ViT-B/32 is also fairly light. Both work on CPU.
CLIP_MODEL_NAME   = os.getenv("CLIP_MODEL_NAME", "ViT-B-32")   # or "RN50"
CLIP_PRETRAINED   = os.getenv("CLIP_PRETRAINED", "laion2b_s34b_b79k")  # for RN50: "openai"
CLIP_DEVICE       = "cuda" if (os.getenv("USE_CUDA", "false").lower() == "true" and torch.cuda.is_available()) else "cpu"
TORCH_THREADS     = int(os.getenv("TORCH_THREADS", "1"))
USE_CAPTION_RERANK = os.getenv("RERANK_CAPTIONS", "false").lower() == "true"  # keep False for extra-light mode

torch.set_num_threads(TORCH_THREADS)
torch.set_grad_enabled(False)

# -------------------- App --------------------
app = FastAPI(title="PhotoFeed AI (light CLIP)", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app$",
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,   # simple fetch from browser
)

@app.get("/health")
def health():
    return {
        "ok": True,
        "clip": {
            "model": CLIP_MODEL_NAME,
            "pretrained": CLIP_PRETRAINED,
            "device": CLIP_DEVICE,
            "threads": TORCH_THREADS,
            "rerank": USE_CAPTION_RERANK,
        },
        "llm": {
            "has_key": bool(OPENROUTER_API_KEY),
            "model": OPENROUTER_MODEL,
        },
    }

# -------------------- CLIP (open_clip) --------------------
import open_clip

# Create model + preprocess; keep CPU by default
_clip_model, _clip_preprocess = None, None
_clip_tokenizer = None

def _init_clip():
    global _clip_model, _clip_preprocess, _clip_tokenizer
    if _clip_model is not None:
        return
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
    )
    model.eval()
    model.to(CLIP_DEVICE)
    if CLIP_DEVICE == "cuda":
        model.half()
    _clip_model = model
    _clip_preprocess = preprocess
    _clip_tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)

# Label banks (kept intentionally small)
OBJECT_LABELS = [
    "person","dog","cat","bird","bicycle","car","bench","flower","tree","leaf",
    "mountain","river","ocean","beach","sunset","sky","cloud","coffee","book","camera",
]
SCENE_LABELS = [
    "park","forest","garden","street","city","rooftop","cafe","bridge","lake",
    "waterfall","desert","snow","museum","temple","beach","cliff"
]

# Precomputed text embeddings cache
_text_bank: Dict[str, torch.Tensor] = {}

def _text_features(labels: List[str], template: str) -> torch.Tensor:
    """
    Return a (len(labels), d) tensor of normalized text features, cached per template+labels key.
    """
    key = f"{template}||{'|'.join(labels)}"
    if key in _text_bank:
        return _text_bank[key]
    _init_clip()
    texts = [template.format(l) for l in labels]
    toks = _clip_tokenizer(texts)
    toks = toks.to(CLIP_DEVICE)
    with torch.no_grad():
        txt = _clip_model.encode_text(toks)
        txt = txt / txt.norm(dim=-1, keepdim=True)
    if CLIP_DEVICE == "cuda":
        txt = txt.float()  # store as fp32 CPU to save VRAM
    txt = txt.cpu()
    _text_bank[key] = txt
    return txt

def _image_features(img: Image.Image) -> torch.Tensor:
    """
    Encode an image; returns 1xd normalized tensor on CPU.
    """
    _init_clip()
    # Preprocess -> tensor
    t = _clip_preprocess(img).unsqueeze(0)
    if CLIP_DEVICE == "cuda":
        t = t.half().to(CLIP_DEVICE)
    else:
        t = t.to(CLIP_DEVICE)

    with torch.no_grad():
        feat = _clip_model.encode_image(t)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    if CLIP_DEVICE == "cuda":
        feat = feat.float()
    return feat.cpu()  # keep CPU tensors onward

def _rank_labels(img: Image.Image, labels: List[str], template="a photo of {}") -> List[Tuple[str, float]]:
    """
    Cosine similarity image vs label texts; returns [(label, score)] sorted desc.
    """
    if not labels:
        return []
    img_f = _image_features(img)  # (1, d)
    txt_f = _text_features(labels, template)  # (n, d)
    sims = (img_f @ txt_f.T).squeeze(0).numpy().tolist()
    pairs = list(zip(labels, sims))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs

# -------------------- Utils (grounding, cleaning) --------------------
_STOP = {"a","an","the","and","or","with","in","on","of","to","for","my","me",
         "our","your","we","us","you","his","her","their","at","by","from","as"}
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

def safe_grounding_from_url(url: str) -> Optional[str]:
    name = os.path.basename(url).split("?")[0].lower()
    toks = [t for t in re.findall(r"[a-z]{3,}", name)]
    ban = {"image","img","photo","pic","file","upload","final","edit","share",
           "jpg","jpeg","png","webp"}
    toks = [t for t in toks if t not in ban]
    if not toks or sum(c in "aeiou" for c in "".join(toks)) < 2:
        return None
    return " ".join(toks[:4])

def fallback_hashtags(prompt: Optional[str], grounding: str) -> List[str]:
    base = sanitize_vibe(prompt or "") or sanitize_vibe(grounding)
    tags = [_to_tag(w) for w in base.split() if w]
    defaults = ["#soft-light","#evening-tones","#street-moment","#quiet-hours","#daily-notes","#mood"]
    uniq = list(dict.fromkeys([t for t in (tags + defaults) if t and t != "#"]))
    return uniq[:6]

# -------------------- LLM call (OpenRouter) --------------------
def call_llm_captions(grounding: str, vibe: Optional[str], objects=None, scenes=None, n_variants: int = 3) -> dict:
    """
    Returns {"captions": [...], "hashtags": [...]}
    Robust to missing key or API errors.
    """
    tone = sanitize_vibe(vibe or "")

    # No key → deterministic local fallback
    if not OPENROUTER_API_KEY:
        base = clean_caption(f"{(tone or '').strip()}, {grounding}".strip(", ")) or "Captured the moment."
        tags = list({t for t in (_to_tag(w) for w in tone.split()) if t and t != "#"})
        return {"captions": [base][:n_variants], "hashtags": tags[:6]}

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
        r = requests.post(f"{OPENROUTER_BASE}/chat/completions", headers=headers, json=payload, timeout=45)
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        m = re.search(r"\{.*\}", text, re.S)
        obj = json.loads(m.group(0)) if m else {"captions": [text.strip()], "hashtags": []}
        caps = [clean_caption(c) for c in obj.get("captions", []) if c][:n_variants]
        tags = obj.get("hashtags") or []
        tags = [t if str(t).startswith("#") else _to_tag(str(t)) for t in tags]
        tags = list(dict.fromkeys([t for t in tags if t and t != "#"]))[:8]
        return {"captions": caps or ["Captured the moment."], "hashtags": tags}
    except Exception:
        base = clean_caption(f"{tone}, {grounding}".strip(", ")) or "Captured the moment."
        vibe_tags = [_to_tag(w) for w in (tone or "").split()[:6]]
        return {"captions": [base], "hashtags": list(dict.fromkeys([t for t in vibe_tags if t]))[:6]}

# -------------------- IO helpers --------------------
def fetch_image(url: str) -> Image.Image:
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    # Small memory trick: limit max side to 640 before CLIP preprocess
    max_side = 640
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w*scale), int(h*scale)))
    return img

# -------------------- Schemas --------------------
class AnalyzeReq(BaseModel):
    imageUrl: str
    top_k: int = 6

class SuggestReq(BaseModel):
    imageUrl: str
    prompt: Optional[str] = None
    top_k: int = 6
    n_variants: int = 3

# -------------------- API Router --------------------
api = APIRouter(prefix="/api")

@api.post("/ai/analyze")
def analyze(req: AnalyzeReq):
    img = fetch_image(req.imageUrl)
    k = max(1, min(8, req.top_k))

    objs = _rank_labels(img, OBJECT_LABELS, template="a photo of {}")[:k]
    scns = _rank_labels(img, SCENE_LABELS, template="a {} scene")[:k]

    return {
        "objects": [{"label": l, "score": float(s)} for l, s in objs],
        "scenes":  [{"label": l, "score": float(s)} for l, s in scns],
    }

@api.post("/ai/suggest")
def suggest(req: SuggestReq):
    # 1) Fetch + analyze (lightweight CLIP)
    img = fetch_image(req.imageUrl)
    k = max(1, min(8, req.top_k))
    obj_pairs = _rank_labels(img, OBJECT_LABELS, template="a photo of {}")[:k]
    scn_pairs = _rank_labels(img, SCENE_LABELS, template="a {} scene")[:k]
    analysis = {
        "objects": [{"label": l, "score": float(s)} for l, s in obj_pairs],
        "scenes":  [{"label": l, "score": float(s)} for l, s in scn_pairs],
    }

    # 2) Neutral grounding from labels (no people words in text)
    PERSON_WORDS = {"man","woman","boy","girl","person","people","couple"}
    core_objs = [l for (l, _) in obj_pairs if l not in PERSON_WORDS]
    scene = scn_pairs[0][0] if scn_pairs else None

    parts = []
    if core_objs[:2]:
        if len(core_objs) >= 2:
            parts.append(f"a {core_objs[0]} and {core_objs[1]}")
        else:
            parts.append(f"a {core_objs[0]}")
    if scene:
        parts.append(f"in a {scene}")
    guessed = safe_grounding_from_url(req.imageUrl)
    grounding = clean_caption(", ".join(parts)) or req.prompt or (guessed or "a quiet moment")

    # 3) LLM captions (+ fallback)
    obj_labels = [l for (l, _) in obj_pairs][:5]
    scn_labels = [l for (l, _) in scn_pairs][:3]
    out = call_llm_captions(
        grounding=grounding,
        vibe=req.prompt,
        objects=obj_labels,
        scenes=scn_labels,
        n_variants=max(1, min(5, req.n_variants)),
    )

    captions = out.get("captions") or [grounding]
    hashtags = out.get("hashtags") or fallback_hashtags(req.prompt, grounding)

    # 4) (Optional) CLIP rerank captions against image (off by default)
    if USE_CAPTION_RERANK and len(captions) > 1:
        _init_clip()
        img_f = _image_features(img)              # (1, d)
        toks = open_clip.get_tokenizer(CLIP_MODEL_NAME)(captions).to(CLIP_DEVICE)
        with torch.no_grad():
            txt = _clip_model.encode_text(toks)
            txt = txt / txt.norm(dim=-1, keepdim=True)
        sims = (img_f @ txt.cpu().T).squeeze(0).numpy().tolist()
        order = sorted(range(len(captions)), key=lambda i: sims[i], reverse=True)
        captions = [captions[i] for i in order]

    return {
        "analysis": analysis,
        "grounding": grounding,
        "captions": captions[:max(1, min(5, req.n_variants))],
        "hashtags": hashtags[:8],
    }

app.include_router(api)
