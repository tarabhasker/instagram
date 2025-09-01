# ai_service.py  (drop-in)
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Tuple, Dict
from PIL import Image, UnidentifiedImageError
import requests, io, os, re, json
import torch

# -------------------- Env / Config --------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE    = os.getenv("OPENROUTER_BASE", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL   = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")

# NOTE: RN50+openai is smallest; keep ViT-B-32 if you prefer accuracy
CLIP_MODEL_NAME   = os.getenv("CLIP_MODEL_NAME", "ViT-B-32")  # or "RN50"
CLIP_PRETRAINED   = os.getenv("CLIP_PRETRAINED", "laion2b_s34b_b79k")  # RN50 → "openai"
CLIP_DEVICE       = "cuda" if (os.getenv("USE_CUDA", "false").lower()=="true" and torch.cuda.is_available()) else "cpu"
TORCH_THREADS     = int(os.getenv("TORCH_THREADS", "1"))
USE_CAPTION_RERANK = os.getenv("RERANK_CAPTIONS", "false").lower() == "true"

torch.set_num_threads(TORCH_THREADS)
torch.set_grad_enabled(False)

# -------------------- App --------------------
app = FastAPI(title="PhotoFeed AI (light CLIP)", version="1.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_origin_regex=r"https://.*\.vercel\.app$",
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

@app.get("/")
def root():
    return {"ok": True, "service": "PhotoFeed AI", "endpoints": ["/health", "/api/ai/analyze", "/api/ai/suggest"]}

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
            "ready": _clip_ready,
            "error": _clip_error,
        },
        "llm": {"has_key": bool(OPENROUTER_API_KEY), "model": OPENROUTER_MODEL},
    }

# -------------------- CLIP (open_clip) --------------------
_clip_model, _clip_preprocess, _clip_tokenizer = None, None, None
_clip_ready = False
_clip_error = None

def _init_clip():
    global _clip_model, _clip_preprocess, _clip_tokenizer, _clip_ready, _clip_error
    if _clip_model is not None or _clip_ready:
        return
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED)
        model.eval()
        model.to(CLIP_DEVICE)
        if CLIP_DEVICE == "cuda":
            model.half()
        _clip_model = model
        _clip_preprocess = preprocess
        _clip_tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
        _clip_ready = True
    except Exception as e:
        _clip_error = str(e)
        _clip_ready = False

OBJECT_LABELS = ["person","dog","cat","bird","bicycle","car","bench","flower","tree","leaf","mountain","river","ocean","beach","sunset","sky","cloud","coffee","book","camera"]
SCENE_LABELS  = ["park","forest","garden","street","city","rooftop","cafe","bridge","lake","waterfall","desert","snow","museum","temple","beach","cliff"]

_text_bank: Dict[str, torch.Tensor] = {}

def _text_features(labels: List[str], template: str) -> torch.Tensor:
    key = f"{template}||{'|'.join(labels)}"
    if key in _text_bank: return _text_bank[key]
    _init_clip()
    toks = _clip_tokenizer([template.format(l) for l in labels]).to(CLIP_DEVICE)
    with torch.no_grad():
        txt = _clip_model.encode_text(toks)
        txt = txt / txt.norm(dim=-1, keepdim=True)
    if CLIP_DEVICE == "cuda": txt = txt.float()
    txt = txt.cpu()
    _text_bank[key] = txt
    return txt

def _image_features(img: Image.Image) -> torch.Tensor:
    _init_clip()
    if not _clip_ready:
        return torch.zeros((1, 512))
    t = _clip_preprocess(img).unsqueeze(0)
    t = t.half().to(CLIP_DEVICE) if CLIP_DEVICE=="cuda" else t.to(CLIP_DEVICE)
    with torch.no_grad():
        feat = _clip_model.encode_image(t)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    if CLIP_DEVICE == "cuda": feat = feat.float()
    return feat.cpu()

def _rank_labels(img: Image.Image, labels: List[str], template="a photo of {}") -> List[Tuple[str, float]]:
    if not labels:
        return []
    try:
        img_f = _image_features(img)
        if img_f.numel() == 0:
            return []
        txt_f = _text_features(labels, template)
        if txt_f.numel() == 0:
            return []
        sims = (img_f @ txt_f.T).squeeze(0).numpy().tolist()
        pairs = list(zip(labels, sims))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs
    except Exception:
        return []


# -------------------- Caption utils --------------------
_STOP = {"a","an","the","and","or","with","in","on","of","to","for","my","me","our","your","we","us","you","his","her","their","at","by","from","as"}
def _tokens(s: str): return [w for w in re.findall(r"[a-z']+", (s or "").lower())]
def sanitize_vibe(s: str) -> str: return " ".join([w for w in _tokens(s) if w not in _STOP][:6])
def _to_tag(s: str) -> str: return "#" + re.sub(r"[^a-z0-9]+","-",(s or "").lower()).strip("-")
def clean_caption(text: str) -> str:
    if not text: return text
    t = re.sub(r"\s+"," ",text).strip()
    return t[0].upper()+t[1:] if t else t
def safe_grounding_from_url(url: str) -> Optional[str]:
    name = os.path.basename(url).split("?")[0].lower()
    toks = [t for t in re.findall(r"[a-z]{3,}", name) if t not in {"image","img","photo","pic","file","upload","final","edit","share","jpg","jpeg","png","webp"}]
    if not toks or sum(c in "aeiou" for c in "".join(toks)) < 2: return None
    return " ".join(toks[:4])
def fallback_hashtags(prompt: Optional[str], grounding: str) -> List[str]:
    base = sanitize_vibe(prompt or "") or sanitize_vibe(grounding)
    tags = [_to_tag(w) for w in base.split() if w]
    defaults = ["#soft-light","#evening-tones","#street-moment","#quiet-hours","#daily-notes","#mood"]
    uniq = list(dict.fromkeys([t for t in (tags + defaults) if t and t != "#"]))
    return uniq[:6]

# -------------------- LLM --------------------
def call_llm_captions(grounding: str, vibe: Optional[str], objects=None, scenes=None, n_variants: int = 3) -> dict:
    tone = sanitize_vibe(vibe or "")
    if not OPENROUTER_API_KEY:
        base = clean_caption(f"{(tone or '').strip()}, {grounding}".strip(", ")) or "Captured the moment."
        tags = list({t for t in (_to_tag(w) for w in tone.split()) if t and t != "#"})
        return {"captions": [base][:n_variants], "hashtags": tags[:6]}

    system = (
        "You write Instagram-ready captions that feel human and specific.\n"
        "Use grounding/objects/scenes for facts; do NOT describe literally or repeat the user's style words.\n"
        "Never include these words: photo, picture, image, man, woman, couple, laying, lying, grass.\n"
        "No quotes or hashtags inside captions.\n"
        f"Create {n_variants} distinct options across lengths: 1 micro (2–4 words), 2 short (≤12), 1 mid (12–22), 1 longer (22–40).\n"
        'Return STRICT JSON: {"captions":["..."], "hashtags":["#..."]}\n'
        "Hashtags: 5–8, specific, lowercase kebab-case, no duplicates."
    )
    user = {"grounding": grounding, "objects": objects or [], "scenes": scenes or [], "style_prompt": tone}

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("OPENROUTER_REFERRER","http://localhost:5173"),
        "X-Title": os.getenv("OPENROUTER_TITLE","Mini Instagram AI"),
    }
    payload = {"model": OPENROUTER_MODEL, "messages": [{"role":"system","content":system},{"role":"user","content":json.dumps(user)}], "temperature":0.65, "top_p":0.9}
    try:
        r = requests.post(f"{OPENROUTER_BASE}/chat/completions", headers=headers, json=payload, timeout=45)
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        m = re.search(r"\{.*\}", text, re.S)
        obj = json.loads(m.group(0)) if m else {"captions":[text.strip()], "hashtags":[]}
        caps = [clean_caption(c) for c in obj.get("captions", []) if c][:n_variants]
        tags = obj.get("hashtags") or []
        tags = [t if str(t).startswith("#") else _to_tag(str(t)) for t in tags]
        tags = list(dict.fromkeys([t for t in tags if t and t != "#"]))[:8]
        return {"captions": caps or ["Captured the moment."], "hashtags": tags}
    except Exception:
        base = clean_caption(f"{tone}, {grounding}".strip(", ")) or "Captured the moment."
        vibe_tags = [_to_tag(w) for w in (tone or "").split()[:6]]
        return {"captions": [base], "hashtags": list(dict.fromkeys([t for t in vibe_tags if t]))[:6]}

# -------------------- HTTP session w/ retry --------------------
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_session = requests.Session()
_session.headers.update({"User-Agent": "Flashgram-AI/1.1 (+https://example.com)"})
_retry = Retry(
    total=2, connect=2, read=2,
    backoff_factor=0.6,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=frozenset(["GET"])
)
_adapter = HTTPAdapter(max_retries=_retry, pool_maxsize=10)
_session.mount("http://", _adapter)
_session.mount("https://", _adapter)

# -------------------- IO helpers --------------------
def fetch_image(url: str) -> Image.Image:
    try:
        r = _session.get(url, timeout=(6, 30))  # (connect, read)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
    except (requests.RequestException, UnidentifiedImageError) as e:
        raise RuntimeError(f"image fetch failed: {e}")

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
    try:
        img = fetch_image(req.imageUrl)
    except Exception as e:
        return JSONResponse(status_code=422, content={"error":{"code":"IMAGE_FETCH_FAILED","message":str(e)}})

    k = max(1, min(8, req.top_k))
    objs = _rank_labels(img, OBJECT_LABELS, template="a photo of {}")[:k]
    scns = _rank_labels(img, SCENE_LABELS, template="a {} scene")[:k]
    return {"objects":[{"label":l,"score":float(s)} for l,s in objs], "scenes":[{"label":l,"score":float(s)} for l,s in scns]}

@api.post("/ai/suggest")
def suggest(req: SuggestReq):
    # Try to fetch the image; if it fails, we still return captions.
    img = None
    analysis = {"objects": [], "scenes": []}
    try:
        img = fetch_image(req.imageUrl)
    except Exception:
        pass  # graceful fallback

    if img is not None:
        k = max(1, min(8, req.top_k))
        obj_pairs = _rank_labels(img, OBJECT_LABELS, template="a photo of {}")[:k]
        scn_pairs = _rank_labels(img, SCENE_LABELS, template="a {} scene")[:k]
        analysis = {
            "objects": [{"label": l, "score": float(s)} for l, s in obj_pairs],
            "scenes":  [{"label": l, "score": float(s)} for l, s in scn_pairs],
        }
        PERSON_WORDS = {"man","woman","boy","girl","person","people","couple"}
        core_objs = [l for (l, _) in obj_pairs if l not in PERSON_WORDS]
        scene = scn_pairs[0][0] if scn_pairs else None
        parts = []
        if core_objs[:2]:
            parts.append(f"a {core_objs[0]}"+(f" and {core_objs[1]}" if len(core_objs)>=2 else ""))
        if scene: parts.append(f"in a {scene}")
        guessed = safe_grounding_from_url(req.imageUrl)
        grounding = clean_caption(", ".join([p for p in parts if p])) or req.prompt or (guessed or "a quiet moment")
    else:
        # No image → grounding from URL/prompt only
        grounding = clean_caption(safe_grounding_from_url(req.imageUrl) or (req.prompt or "a quiet moment"))

    obj_labels = [o["label"] for o in analysis["objects"]][:5]
    scn_labels = [s["label"] for s in analysis["scenes"]][:3]
    out = call_llm_captions(grounding=grounding, vibe=req.prompt, objects=obj_labels, scenes=scn_labels, n_variants=max(1, min(5, req.n_variants)))

    captions = out.get("captions") or [grounding]
    hashtags = out.get("hashtags") or fallback_hashtags(req.prompt, grounding)

    # (Optional) rerank ...
    if img is not None and USE_CAPTION_RERANK and len(captions) > 1:
        _init_clip()
        img_f = _image_features(img)
        toks = open_clip.get_tokenizer(CLIP_MODEL_NAME)(captions).to(CLIP_DEVICE)
        with torch.no_grad():
            txt = _clip_model.encode_text(toks)
            txt = txt / txt.norm(dim=-1, keepdim=True)
        sims = (img_f @ txt.cpu().T).squeeze(0).numpy().tolist()
        order = sorted(range(len(captions)), key=lambda i: sims[i], reverse=True)
        captions = [captions[i] for i in order]

    # --- FIX 1: synthesize fallback labels ---
    fallback_labels: List[str] = []
    if not analysis["objects"] and not analysis["scenes"]:
        if hashtags:
            fallback_labels.extend([h.lstrip("#") for h in hashtags if isinstance(h, str)])
        for w in re.findall(r"[a-z0-9]{3,}", (grounding or "").lower()):
            if w not in _STOP:
                fallback_labels.append(w)
        for w in (sanitize_vibe(req.prompt or "")).split():
            fallback_labels.append(w)

    fallback_labels = [re.sub(r"[^a-z0-9-]+", "", w) for w in fallback_labels]
    fallback_labels = [w for w in fallback_labels if w]
    seen = set()
    fallback_labels = [w for w in fallback_labels if not (w in seen or seen.add(w))][:6]

    out_labels = (
        [o["label"] for o in analysis["objects"][:3]] +
        [s["label"] for s in analysis["scenes"][:3]]
    )
    if not out_labels:
        out_labels = fallback_labels

    # final response (with labels now included)
    return {
        "analysis": analysis,
        "grounding": grounding,
        "captions": captions[:max(1, min(5, req.n_variants))],
        "hashtags": hashtags[:8],
        "labels": out_labels,   # <-- new
    }

app.include_router(api)
