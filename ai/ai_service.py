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
OPENROUTER_MODEL   = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3.1:free")

# Feature switches
USE_CLIP           = os.getenv("USE_CLIP", "false").lower() == "true"
CLIP_MODEL_NAME    = os.getenv("CLIP_MODEL_NAME", "RN50")
CLIP_PRETRAINED    = os.getenv("CLIP_PRETRAINED", "openai")
CLIP_DEVICE        = "cuda" if (os.getenv("USE_CUDA","false").lower()=="true" and torch.cuda.is_available()) else "cpu"
TORCH_THREADS      = int(os.getenv("TORCH_THREADS", "1"))
USE_CAPTION_RERANK = os.getenv("RERANK_CAPTIONS", "true").lower() == "true"

USE_HF_SCENES      = os.getenv("USE_HF_SCENES", "false").lower() == "true"
HF_TOKEN           = os.getenv("HF_TOKEN", "").strip()
HF_SCENE_ENDPOINT  = os.getenv("HF_SCENE_ENDPOINT", "https://api-inference.huggingface.co/models/zhanghang1989/ResNet50-Places365").strip()

torch.set_num_threads(TORCH_THREADS)
torch.set_grad_enabled(False)

# -------------------- App --------------------
app = FastAPI(title="PhotoFeed AI (robust)", version="1.3")
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

# -------------------- CLIP (lazy & optional) --------------------
_clip_model = _clip_preprocess = _clip_tokenizer = None
_clip_ready = False
_clip_error = None

def _try_import_open_clip():
    global open_clip
    try:
        import open_clip  # type: ignore
        return open_clip
    except Exception as e:
        return None

def _init_clip():
    """Never block startup. If CLIP is disabled or unavailable, mark not ready and return quickly."""
    global _clip_model, _clip_preprocess, _clip_tokenizer, _clip_ready, _clip_error
    if not USE_CLIP:
        _clip_ready, _clip_error = False, "CLIP disabled by USE_CLIP=false"
        return
    if _clip_ready or _clip_model is not None:
        return
    oc = _try_import_open_clip()
    if oc is None:
        _clip_error, _clip_ready = "open_clip import failed", False
        return
    try:
        model, _, preprocess = oc.create_model_and_transforms(CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED)
        model.eval()
        model.to(CLIP_DEVICE)
        if CLIP_DEVICE == "cuda":
            model.half()
        _clip_model = model
        _clip_preprocess = preprocess
        _clip_tokenizer = oc.get_tokenizer(CLIP_MODEL_NAME)
        _clip_ready = True
    except Exception as e:
        # Do NOT crash request; just mark unavailable.
        _clip_error, _clip_ready = str(e), False

OBJECT_LABELS = [
    "person","dog","cat","bird","bicycle","car","bench","flower","tree","leaf",
    "mountain","river","ocean","beach","sunset","sky","cloud","coffee","book","camera",
    "cup","table","chair","window","streetlight","bridge","building","notebook","train","bus"
]
SCENE_LABELS  = [
    "park","forest","garden","street","city","rooftop","cafe","bridge","lake","waterfall",
    "desert","snow","museum","temple","beach","cliff","library","studio","kitchen","market"
]

_text_bank: Dict[str, torch.Tensor] = {}

def _text_features(labels: List[str], template: str) -> torch.Tensor:
    if not labels or not _clip_ready:
        return torch.zeros((0, 512))
    key = f"{template}||{'|'.join(labels)}"
    if key in _text_bank: return _text_bank[key]
    toks = _clip_tokenizer([template.format(l) for l in labels]).to(CLIP_DEVICE)
    with torch.no_grad():
        txt = _clip_model.encode_text(toks)
        txt = txt / txt.norm(dim=-1, keepdim=True)
    if CLIP_DEVICE == "cuda": txt = txt.float()
    txt = txt.cpu()
    _text_bank[key] = txt
    return txt

def _image_features(img: Image.Image) -> torch.Tensor:
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
    if not labels or not _clip_ready:
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

# -------------------- HF Places365 (strictly opt-in) --------------------
def _img_to_jpeg_bytes(img: Image.Image, max_side=256, quality=85) -> bytes:
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w*scale), int(h*scale)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()

def hf_scene_labels(img: Image.Image, top_k=3) -> List[str]:
    if not USE_HF_SCENES or not HF_TOKEN or not HF_SCENE_ENDPOINT:
        return []
    try:
        payload = _img_to_jpeg_bytes(img, max_side=256)
        r = requests.post(
            HF_SCENE_ENDPOINT,
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            data=payload,
            timeout=12
        )
        r.raise_for_status()
        preds = r.json()
        if isinstance(preds, list):
            labs = []
            for p in preds[:top_k]:
                lbl = str(p.get("label","")).split(",")[0].strip().lower()
                if lbl: labs.append(lbl)
            out, seen = [], set()
            for l in labs:
                l = re.sub(r"[^a-z0-9\- ]+","", l).split("/")[-1].strip()
                if l and l not in seen:
                    seen.add(l); out.append(l)
            return out[:top_k]
        return []
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

# -------------------- LLM (DeepSeek via OpenRouter) --------------------
def call_llm_captions(grounding: str, vibe: Optional[str], objects=None, scenes=None, n_variants: int = 3) -> dict:
    objs = [o for o in (objects or []) if o]
    scns = [s for s in (scenes or []) if s]

    RAW_VIBE = sanitize_vibe(vibe or "")
    vibe_rules = []
    if RAW_VIBE:
        vibe_map = {
            "aesthetic": "minimal, moody, refined",
            "moody": "atmospheric, introspective",
            "retro": "nostalgic, playful",
            "cinematic": "evocative, scene-like",
            "cozy": "warm, intimate",
            "vintage": "grainy, timeless",
        }
        vibe_rules.append("tone: " + vibe_map.get(RAW_VIBE.split()[0], RAW_VIBE))
    banned_tokens = list(set(RAW_VIBE.split())) if RAW_VIBE else []
    if objs or scns:
        vibe_rules.append("reference at least 1 concrete element from objects/scenes (not a literal list)")

    system = (
        "You write Instagram captions that feel human and specific.\n"
        "Use grounding/objects/scenes for facts; do NOT repeat the user's style words.\n"
        "Avoid generic filler and repetition. No quotes, emojis, or hashtags inside captions.\n"
        "Vary lengths across options.\n"
        "Output STRICT JSON: {\"captions\":[\"...\"], \"hashtags\":[\"#...\"]}\n"
        f"Banned words: {', '.join(banned_tokens) if banned_tokens else '(none)'}\n"
        "If style words are in input, interpret them as tone only."
    )

    user = {
        "grounding": grounding,
        "objects": objs[:5],
        "scenes": scns[:3],
        "style_guidance": vibe_rules,
        "instructions": [
            "Create distinct options: 1 micro (2–4 words), 2 short (≤12), 1 mid (12–22), 1 long (22–40).",
            "Keep it specific; avoid repetition.",
        ],
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("OPENROUTER_REFERRER","http://localhost:5173"),
        "X-Title": os.getenv("OPENROUTER_TITLE","Mini Instagram AI"),
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role":"system","content":system},
            {"role":"user","content":json.dumps(user)},
        ],
        "temperature": 0.7,
        "top_p": 0.9,
    }

    def _postprocess_caps(caps: List[str]) -> List[str]:
        out = []
        ban = set(banned_tokens)
        for c in (caps or []):
            if not c: continue
            c = clean_caption(c)
            c = re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", c, flags=re.I)  # remove immediate duplicates
            words = _tokens(c)
            words = [w for w in words if w.lower() not in ban]
            if not words: continue
            c = words[0].capitalize() + (" " + " ".join(words[1:]) if len(words) > 1 else "")
            if c not in out: out.append(c)
        return out

    try:
        if not OPENROUTER_API_KEY:
            base_txt = grounding if isinstance(grounding, str) else ", ".join(objs[:2] + scns[:1])
            caps = [
                clean_caption(re.sub(r"[, ]+$","", base_txt)) or "Captured the moment.",
                clean_caption(f"{objs[0]} by {scns[0]}") if (objs and scns) else clean_caption(base_txt),
                clean_caption(f"Quiet {scns[0]} detail") if scns else clean_caption(base_txt),
            ]
            tags = [f"#{re.sub(r'[^a-z0-9]+','-',t)}" for t in (objs[:2] + scns[:2])]
            return {"captions": _postprocess_caps(caps)[:n_variants], "hashtags": list(dict.fromkeys(tags))[:6]}

        r = requests.post(f"{OPENROUTER_BASE}/chat/completions", headers=headers, json=payload, timeout=45)
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        m = re.search(r"\{.*\}", text, re.S)
        obj = json.loads(m.group(0)) if m else {"captions":[text.strip()], "hashtags":[]}

        caps = _postprocess_caps(obj.get("captions", []))
        tags = obj.get("hashtags") or [f"#{re.sub(r'[^a-z0-9]+','-',t)}" for t in (objs[:3] + scns[:3])]
        tags = [t if str(t).startswith("#") else "#"+str(t) for t in tags]
        tags = list(dict.fromkeys([t for t in tags if t and t != "#"]))[:8]
        if not caps:
            caps = _postprocess_caps([grounding]) or ["Captured the moment."]
        return {"captions": caps[:max(1, n_variants)], "hashtags": tags}
    except Exception:
        base = clean_caption(grounding) or "Captured the moment."
        tags = [f"#{re.sub(r'[^a-z0-9]+','-',t)}" for t in (objs[:3] + scns[:3])]
        return {"captions": [base], "hashtags": list(dict.fromkeys(tags))[:6]}

# -------------------- HTTP session w/ retry --------------------
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_session = requests.Session()
_session.headers.update({"User-Agent": "Flashgram-AI/1.3 (+https://example.com)"})
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

# -------------------- Router --------------------
api = APIRouter(prefix="/api")

@app.get("/health")
def health():
    # No CLIP init here (fast & safe)
    return {
        "ok": True,
        "clip": {
            "enabled": USE_CLIP,
            "model": CLIP_MODEL_NAME,
            "pretrained": CLIP_PRETRAINED,
            "device": CLIP_DEVICE,
            "threads": TORCH_THREADS,
            "rerank": USE_CAPTION_RERANK,
            "ready": _clip_ready,
            "error": _clip_error,
        },
        "hf": {"enabled": USE_HF_SCENES, "has_token": bool(HF_TOKEN)},
        "llm": {"has_key": bool(OPENROUTER_API_KEY), "model": OPENROUTER_MODEL},
    }

@api.post("/ai/analyze")
def analyze(req: AnalyzeReq):
    try:
        img = fetch_image(req.imageUrl)
    except Exception as e:
        return JSONResponse(status_code=422, content={"error":{"code":"IMAGE_FETCH_FAILED","message":str(e)}})

    # Lazy-init CLIP on demand, but never block if disabled/unavailable
    _init_clip()

    k = max(1, min(8, req.top_k))
    objs = _rank_labels(img, OBJECT_LABELS, template="a photo of {}")[:k]
    scns = _rank_labels(img, SCENE_LABELS, template="a {} scene")[:k]

    # Only if CLIP gave no scenes AND HF is explicitly enabled
    if not scns:
        hf_s = hf_scene_labels(img, top_k=min(3, k))
        if hf_s:
            scns = [(s, 0.0) for s in hf_s]

    return {
        "objects":[{"label":l,"score":float(s)} for l,s in objs],
        "scenes":[{"label":l,"score":float(s)} for l,s in scns]
    }

@api.post("/ai/suggest")
def suggest(req: SuggestReq):
    img = None
    analysis = {"objects": [], "scenes": []}
    try:
        img = fetch_image(req.imageUrl)
    except Exception:
        pass

    if img is not None:
        _init_clip()
        k = max(1, min(8, req.top_k))
        obj_pairs = _rank_labels(img, OBJECT_LABELS, template="a photo of {}")[:k]
        scn_pairs = _rank_labels(img, SCENE_LABELS, template="a {} scene")[:k]

        if not scn_pairs:  # optional HF scenes
            hf_s = hf_scene_labels(img, top_k=min(3, k))
            if hf_s:
                scn_pairs = [(s, 0.0) for s in hf_s]

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
        grounding = clean_caption(", ".join([p for p in parts if p])) or (guessed or "a quiet moment")
    else:
        grounding = clean_caption(safe_grounding_from_url(req.imageUrl) or "a quiet moment")

    obj_labels = [o["label"] for o in analysis["objects"]][:5]
    scn_labels = [s["label"] for s in analysis["scenes"]][:3]
    out = call_llm_captions(
        grounding=grounding,
        vibe=req.prompt,
        objects=obj_labels,
        scenes=scn_labels,
        n_variants=max(1, min(5, req.n_variants))
    )

    captions = out.get("captions") or [grounding]
    hashtags = out.get("hashtags") or fallback_hashtags(req.prompt, grounding)

    # Optional rerank (only if CLIP actually ready)
    if img is not None and USE_CAPTION_RERANK and len(captions) > 1:
        _init_clip()
        if _clip_ready:
            oc = _try_import_open_clip()
            img_f = _image_features(img)
            toks = oc.get_tokenizer(CLIP_MODEL_NAME)(captions).to(CLIP_DEVICE)
            with torch.no_grad():
                txt = _clip_model.encode_text(toks)
                txt = txt / txt.norm(dim=-1, keepdim=True)
            sims = (img_f @ txt.cpu().T).squeeze(0).numpy().tolist()
            order = sorted(range(len(captions)), key=lambda i: sims[i], reverse=True)
            captions = [captions[i] for i in order]

    # Fallback labels if analysis empty
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

    return {
        "analysis": analysis,
        "grounding": grounding,
        "captions": captions[:max(1, min(5, req.n_variants))],
        "hashtags": hashtags[:8],
        "labels": out_labels,
    }

app.include_router(api)
