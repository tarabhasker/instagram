# ai_service.py
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Tuple, Dict
from PIL import Image, UnidentifiedImageError
import requests, io, os, re, json
import torch
import concurrent.futures

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

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

# Default HF endpoints (override via env if you want)
HF_OBJECTS_ENDPOINT = os.getenv(
    "HF_OBJECTS_ENDPOINT",
    "https://api-inference.huggingface.co/models/facebook/detr-resnet-50"
).strip()
HF_SCENE_ENDPOINT = os.getenv(
    "HF_SCENE_ENDPOINT",
    "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"
).strip()

torch.set_num_threads(TORCH_THREADS)
torch.set_grad_enabled(False)

print(f"[config] USE_CLIP={USE_CLIP} "
      f"HF_OBJECTS_ENDPOINT={HF_OBJECTS_ENDPOINT} "
      f"HF_SCENE_ENDPOINT={HF_SCENE_ENDPOINT}")

# -------------------- App --------------------
app = FastAPI(title="PhotoFeed AI (robust)", version="1.5")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://tara-takehome-ui.vercel.app",
    ],
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
    except Exception:
        return None

def _init_clip():
    """Try to init CLIP once, but never block the request for too long."""
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
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(
                lambda: oc.create_model_and_transforms(
                    CLIP_MODEL_NAME,
                    pretrained=CLIP_PRETRAINED,
                    cache_dir=os.getenv("OPENCLIP_CACHE_DIR", "/tmp/open_clip_cache")
                )
            )
            model, _, preprocess = fut.result(timeout=8)
        model.eval()
        model.to(CLIP_DEVICE)
        if CLIP_DEVICE == "cuda":
            model.half()
        _clip_model = model
        _clip_preprocess = preprocess
        _clip_tokenizer = oc.get_tokenizer(CLIP_MODEL_NAME)
        _clip_ready = True
        print(f"[clip] loaded {CLIP_MODEL_NAME} on {CLIP_DEVICE}")
    except concurrent.futures.TimeoutError:
        _clip_error, _clip_ready = "CLIP init timeout (>8s)", False
        print("[clip] init timed out (skipping)")
    except Exception as e:
        _clip_error, _clip_ready = f"CLIP init failed: {e}", False
        print("[clip] init failed:", e)

OBJECT_LABELS = [
    "person","hat","straw hat","dress","polka dot dress","sunglasses","smile","fountain",
    "building","church","cathedral","square","plaza","cafe table","bicycle","scooter",
    "bench","flower","tree","cloud","sky","camera","book","coffee","backpack"
]

SCENE_LABELS = [
    "city square","historic center","old town","street","plaza","rooftop","cafe","market",
    "courtyard","bridge","waterfront","park","museum","temple","church","beach","library"
]

_text_bank: Dict[str, torch.Tensor] = {}

def _text_features(labels: List[str], template: str) -> torch.Tensor:
    if not labels or not _clip_ready:
        return torch.zeros((0, 512))
    key = f"{template}||{'|'.join(labels)}"
    if key in _text_bank:
        return _text_bank[key]
    toks = _clip_tokenizer([template.format(l) for l in labels]).to(CLIP_DEVICE)
    with torch.no_grad():
        txt = _clip_model.encode_text(toks)
        txt = txt / txt.norm(dim=-1, keepdim=True)
    if CLIP_DEVICE == "cuda":
        txt = txt.float()
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
    if CLIP_DEVICE == "cuda":
        feat = feat.float()
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

# -------------------- HF helpers (robust) --------------------
def _img_to_jpeg_bytes(img: Image.Image, max_side=256, quality=85) -> bytes:
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w*scale), int(h*scale)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()

def hf_object_labels(img: Image.Image, top_k=5) -> List[str]:
    """Remote DETR object words; raw bytes body (octet-stream)."""
    if not HF_TOKEN or not HF_OBJECTS_ENDPOINT:
        print("[hf_object_labels] skipped (missing HF token or endpoint)")
        return []
    try:
        payload = _img_to_jpeg_bytes(img, max_side=320)
        r = requests.post(
            HF_OBJECTS_ENDPOINT,
            headers={
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": "application/octet-stream",
            },
            data=payload,
            timeout=20
        )
        if r.status_code >= 400:
            print(f"[hf_object_labels] HTTP {r.status_code}: {r.text[:300]}")
        r.raise_for_status()
        preds = r.json()
        labels: List[str] = []
        if isinstance(preds, list):
            for p in preds:
                lbl = str(p.get("label", "")).strip().lower()
                if lbl:
                    lbl = re.sub(r"[^a-z0-9\- ]+", "", lbl)
                    if lbl not in labels:
                        labels.append(lbl)
        print(f"[hf_object_labels] detected: {labels}")
        return labels[:top_k]
    except Exception as e:
        print(f"[hf_object_labels] failed: {e}")
        return []

def hf_scene_labels(img: Image.Image, top_k=3) -> List[str]:
    """Remote ViT image-classification labels; raw bytes body."""
    if not HF_TOKEN or not HF_SCENE_ENDPOINT:
        print("[hf_scene_labels] skipped (missing HF token or endpoint)")
        return []
    try:
        payload = _img_to_jpeg_bytes(img, max_side=224)
        r = requests.post(
            HF_SCENE_ENDPOINT,
            headers={
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": "application/octet-stream",
            },
            data=payload,
            timeout=20
        )
        if r.status_code >= 400:
            print(f"[hf_scene_labels] HTTP {r.status_code}: {r.text[:300]}")
        r.raise_for_status()
        preds = r.json()
        labels: List[str] = []
        if isinstance(preds, list):
            for p in preds[:top_k]:
                lbl = str(p.get("label", "")).split(",")[0].strip().lower()
                if lbl and lbl not in labels:
                    labels.append(re.sub(r"[^a-z0-9\- ]+", "", lbl))
        print(f"[hf_scene_labels] detected: {labels}")
        return labels
    except Exception as e:
        print(f"[hf_scene_labels] failed: {e}")
        return []

# -------------------- Caption utils --------------------
_STOP = {"a","an","the","and","or","with","in","on","of","to","for","my","me","our","your",
         "we","us","you","his","her","their","at","by","from","as"}
_GENERIC = {"nice","beautiful","amazing","cool","awesome","great","pretty","lovely"}
_BAD_TAGS = {"jpg","jpeg","png","image","photo","pic","bca","insta","follow","like","love"}

def _tokens(s: str) -> List[str]:
    return [w for w in re.findall(r"[a-z']+", (s or "").lower())]

def _norm_words(s: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9\-]{2,}", (s or "").lower())]

def sanitize_vibe(s: str) -> str:
    return " ".join([w for w in _tokens(s) if w not in _STOP][:6])

def _to_tag(s: str) -> str:
    return "#" + re.sub(r"[^a-z0-9]+","-",(s or "").lower()).strip("-")

def clean_caption(text: str) -> str:
    if not text: return text
    t = re.sub(r"\s+"," ",text).strip()
    return t[0].upper()+t[1:] if t else t

def safe_grounding_from_url(url: str) -> Optional[str]:
    name = os.path.basename(url).split("?")[0].lower()
    toks = re.findall(r"[a-z]{3,}", name)
    DROP = {
        "image","img","photo","picture","pic","file","upload","final","edit","share",
        "dreamstime","thumbs","close","up","b","jpg","jpeg","png","webp","size","resolution",
        "stock","royalty","free","model","copyspace","girl","boy","man","woman","people"
    }
    keep = [t for t in toks if t not in DROP]
    PRIORITY = {"fountain","plaza","square","italy","rome","florence","venice","street",
                "city","old","town","hat","dress","polka","dots","summer","travel"}
    keep.sort(key=lambda w: (0 if w in PRIORITY else 1))
    keep = keep[:8]
    phrase = " ".join(keep).strip()
    return phrase or None

def build_hashtags(objects: List[str], scenes: List[str], url_hint: str, vibe: Optional[str]) -> List[str]:
    """Prioritize detected labels + vibe; avoid file ext & random URL bits."""
    seeds: List[str] = []
    seeds += objects[:3]
    seeds += scenes[:3]
    vt = sanitize_vibe(vibe or "")
    if vt: seeds.append(vt)
    out, seen = [], set()
    for s in seeds:
        tag = "#" + re.sub(r"[^a-z0-9]+","-", str(s).lower()).strip("-")
        if len(tag) > 1 and tag not in seen:
            seen.add(tag); out.append(tag)
    defaults = ["#soft-light", "#street-moment", "#daily-notes"]
    for d in defaults:
        if len(out) >= 6: break
        if d not in out: out.append(d)
    return out[:6]

def url_aesthetic_hint(url: str, vibe: Optional[str]) -> str:
    base = os.path.basename((url or "").split("?")[0]).lower()
    words = [w for w in re.findall(r"[a-z]{3,}", base)]
    drop = {"image","img","photo","picture","stock","royalty","free","model","people",
            "girl","boy","man","woman","close","up","thumbs","dreamstime","jpeg","jpg","png","webp"}
    keep = [w for w in words if w not in drop]
    place = next((w for w in keep if w in {"square","plaza","street","harbor","cafe","market","church","cathedral","fountain"}), "")
    country = next((w for w in keep if w in {"italy","france","spain","greece","japan"}), "")
    season = next((w for w in keep if w in {"summer","spring","autumn","winter"}), "")
    bits = []
    if season: bits.append(season)
    if place and country: bits.append(f"{place} in {country}")
    elif place: bits.append(place)
    elif country: bits.append(country)
    if not bits: bits.append("sunny old town")
    sv = sanitize_vibe(vibe or "")
    if sv and sv not in {"happy","aesthetic"}:
        bits.append(sv)
    result = ", ".join(bits)
    print(f"[url_aesthetic_hint] fallback hint: {result} (from url: {base})")
    return result

def blind_caption_templates(hint: str) -> List[str]:
    return [
        clean_caption(hint.split(",")[0] + " light"),
        clean_caption("Wandering " + hint.split(",")[0]),
        clean_caption("Postcards from " + hint.split(",")[-1].strip()),
        clean_caption("Slow steps through " + hint.split(",")[0])
    ]

def _clean_hashtags(candidates: List[str], allowed_terms: List[str]) -> List[str]:
    """Keep only tasteful, on-topic tags. Force kebab-case & dedupe."""
    allow = set([t.strip("-") for t in allowed_terms if t])
    out, seen = [], set()
    for t in candidates or []:
        t = str(t).lower().strip()
        t = t[1:] if t.startswith("#") else t
        t = re.sub(r"[^a-z0-9\-]+", "-", t).strip("-")
        if not t or t in seen: continue
        if t in _BAD_TAGS: continue
        words = [w for w in t.split("-") if w]
        if allow and not all((w in allow or w in {"soft","light","golden","hour","street","moment","quiet","hours"}) for w in words):
            continue
        seen.add(t); out.append("#"+t)
        if len(out) >= 8: break
    return out

# -------------------- LLM (DeepSeek via OpenRouter) --------------------
def call_llm_captions(
    grounding: str,
    vibe: Optional[str],
    objects=None,
    scenes=None,
    n_variants: int = 3
) -> dict:
    objs = [o for o in (objects or []) if o][:5]
    scns = [s for s in (scenes or []) if s][:3]

    # Build allow-list for hashtags (keeps tags on-topic)
    allow_terms = set()
    for s in objs + scns + _norm_words(grounding):
        for w in _norm_words(s):
            if w not in _STOP and w not in _BAD_TAGS:
                allow_terms.add(w)
    allow_terms.update({"soft","light","golden","hour","street","moment","quiet","hours","vintage","film","retro","cozy"})

    # Prompt
    vibe_short = sanitize_vibe(vibe or "")
    system = (
        "You are a concise Instagram caption writer.\n"
        "Use the PROVIDED objects/scenes as the concrete anchors.\n"
        "Output JSON only. No markdown. No extra text.\n"
        "Write 4 distinct captions that feel human and specific:\n"
        "  • 1 micro (2–4 words)\n"
        "  • 2 short (≤12 words)\n"
        "  • 1 mid (12–22 words)\n"
        "Rules: no emojis, no quotes, no hashtags inside captions, avoid generic adjectives "
        "(nice, beautiful, amazing). Prefer sensory details (light, texture, weather) and place clues."
    )

    user = {
        "grounding_hint": grounding,
        "objects": objs,
        "scenes": scns,
        "style": vibe_short or None,
        "hashtags_policy": {
            "count": 6,
            "allowed_terms": sorted(list(allow_terms)),
            "disallow": sorted(list(_BAD_TAGS)),
            "examples": ["soft-light","street-moment","golden-hour","quiet-hours"]
        },
        "return": {"captions": 4, "hashtags": 6}
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

    def _post_caps(caps: List[str]) -> List[str]:
        out, seen = [], set()
        for c in (caps or []):
            if not c: continue
            c = clean_caption(c)
            c = re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", c, flags=re.I)  # dedupe immediate repeats
            words = _norm_words(c)
            if not words: continue
            if (len(words) <= 3) and any(w in _GENERIC for w in words):
                continue
            k = c.strip().lower()
            if k in seen: continue
            seen.add(k); out.append(c)
        # respect n_variants
        return out[:max(1, n_variants)]

    try:
        if not OPENROUTER_API_KEY:
            base = clean_caption(grounding) or "Captured the moment."
            caps = [
                base,
                (objs[0] + " in " + (scns[0] if scns else "soft light")).title() if objs else base,
                ("Quiet " + (scns[0] if scns else "street")).title(),
                "Soft light and stone"
            ]
            tags = _clean_hashtags([*(objs[:2] or []), *(scns[:2] or [])], list(allow_terms))
            return {"captions": _post_caps(caps), "hashtags": tags}

        r = requests.post(f"{OPENROUTER_BASE}/chat/completions", headers=headers, json=payload, timeout=45)
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        m = re.search(r"\{.*\}", text, re.S)
        obj = json.loads(m.group(0)) if m else {"captions":[text.strip()], "hashtags":[]}

        caps = _post_caps(obj.get("captions", []))
        if (not caps) and (objs or scns):
            scene_hint = (scns[0] if scns else "").replace("-", " ")
            obj_hint   = (objs[0] if objs else "")
            filler = "in the" if scene_hint else "with"
            caps = [clean_caption(f"{obj_hint} {filler} {scene_hint}, soft light and stone textures").strip(", ")]

        llm_tags = obj.get("hashtags") or []
        hashtags = _clean_hashtags(llm_tags, list(allow_terms))
        return {
            "captions": caps or [clean_caption(grounding) or "A quiet moment"],
            "hashtags": hashtags
        }
    except Exception as e:
        print(f"[call_llm_captions] failed: {e}")
        base = clean_caption(grounding) or "A quiet moment"
        hashtags = _clean_hashtags([*(objs[:3] or []), *(scns[:3] or [])], list(allow_terms))
        return {"captions": [base], "hashtags": hashtags}

# -------------------- HTTP session (for image fetch) --------------------
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_session = requests.Session()
_session.headers.update({"User-Agent": "Flashgram-AI/1.4 (+https://example.com)"})
_retry = Retry(
    total=2, connect=2, read=2,
    backoff_factor=0.6,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=frozenset(["GET"])
)
_adapter = HTTPAdapter(max_retries=_retry, pool_maxsize=10)
_session.mount("http://", _adapter)
_session.mount("https://", _adapter)

def fetch_image(url: str) -> Image.Image:
    try:
        r = _session.get(url, timeout=(6, 30))
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

@app.api_route("/health", methods=["GET", "HEAD"])
def health():
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
            "last_event": globals().get("_clip_last_event", "n/a"),
        },
        "hf": {
            "has_token": bool(HF_TOKEN),
            "objects_endpoint": HF_OBJECTS_ENDPOINT,
            "scenes_endpoint": HF_SCENE_ENDPOINT,
        },
        "llm": {"has_key": bool(OPENROUTER_API_KEY), "model": OPENROUTER_MODEL},
    }

@api.post("/ai/analyze")
def analyze(req: AnalyzeReq):
    try:
        img = fetch_image(req.imageUrl)
    except Exception as e:
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "IMAGE_FETCH_FAILED", "message": str(e)}},
        )

    obj_pairs: List[Tuple[str, float]] = []
    scn_pairs: List[Tuple[str, float]] = []

    if USE_CLIP:
        _init_clip()
        if _clip_ready:
            k = max(1, min(8, req.top_k))
            obj_pairs = _rank_labels(img, OBJECT_LABELS, template="a photo of {}")[:k]
            scn_pairs = _rank_labels(img, SCENE_LABELS, template="a {} scene")[:k]

    # Fallback to HF if CLIP missing/unready or produced nothing
    if not obj_pairs and not scn_pairs:
        obj_labels = hf_object_labels(img, top_k=min(5, req.top_k))
        scn_labels = hf_scene_labels(img, top_k=min(3, req.top_k))
        obj_pairs = [(o, 0.0) for o in obj_labels]
        scn_pairs = [(s, 0.0) for s in scn_labels]
        if not obj_pairs and not scn_pairs:
            print("[analyze] no labels from CLIP or HF; likely URL-only fallback if used later")

    return {
        "objects": [{"label": l, "score": float(s)} for l, s in obj_pairs],
        "scenes":  [{"label": l, "score": float(s)} for l, s in scn_pairs],
    }

@api.post("/ai/suggest")
def suggest(req: SuggestReq):
    # 1) Fetch image
    img: Optional[Image.Image] = None
    try:
        img = fetch_image(req.imageUrl)
    except Exception:
        pass

    # 2) Labels: CLIP then HF fallback
    obj_pairs: List[Tuple[str, float]] = []
    scn_pairs: List[Tuple[str, float]] = []

    if img is not None and USE_CLIP:
        _init_clip()
        if _clip_ready:
            k = max(1, min(8, req.top_k))
            obj_pairs = _rank_labels(img, OBJECT_LABELS, template="a photo of {}")[:k]
            scn_pairs = _rank_labels(img, SCENE_LABELS, template="a {} scene")[:k]

    if img is not None and (not obj_pairs and not scn_pairs):
        obj_labels = hf_object_labels(img, top_k=min(5, req.top_k))
        scn_labels = hf_scene_labels(img, top_k=min(3, req.top_k))
        obj_pairs = [(o, 0.0) for o in obj_labels]
        scn_pairs = [(s, 0.0) for s in scn_labels]

    # 3) Build grounding phrase
    analysis = {
        "objects": [{"label": l, "score": float(s)} for l, s in obj_pairs],
        "scenes":  [{"label": l, "score": float(s)} for l, s in scn_pairs],
    }

    PERSON_WORDS = {"man","woman","boy","girl","person","people","couple"}
    core_objs = [l for (l, _) in obj_pairs if l not in PERSON_WORDS]
    scene = scn_pairs[0][0] if scn_pairs else None

    parts = []
    if core_objs[:2]:
        parts.append(f"a {core_objs[0]}" + (f" and {core_objs[1]}" if len(core_objs) >= 2 else ""))
    if scene:
        parts.append(f"in a {scene}")

    guessed = safe_grounding_from_url(req.imageUrl) or "a quiet moment"
    grounding = clean_caption(", ".join([p for p in parts if p])) or clean_caption(guessed)

    if not obj_pairs and not scn_pairs:
        # nothing detected, make nicer URL-based hint
        grounding = url_aesthetic_hint(req.imageUrl, req.prompt)

    # 4) LLM
    obj_labels = [o["label"] for o in analysis["objects"]][:5]
    scn_labels = [s["label"] for s in analysis["scenes"]][:3]

    out = call_llm_captions(
        grounding=grounding,
        vibe=req.prompt,
        objects=obj_labels,
        scenes=scn_labels,
        n_variants=max(1, min(5, req.n_variants)),
    )

    captions = out.get("captions") or []
    if not captions:
        captions = blind_caption_templates(grounding)
    print(f"[suggest] LLM captions: {captions}")

    # 5) Hashtags
    hashtags = out.get("hashtags") or []
    if len(hashtags) < 3:
        url_hint = os.path.basename((req.imageUrl or "").split("?")[0])
        hashtags = build_hashtags(obj_labels, scn_labels, url_hint, req.prompt)
    print(f"[suggest] hashtags: {hashtags}")

    # 6) Optional CLIP re-rank
    if img is not None and USE_CAPTION_RERANK and len(captions) > 1 and _clip_ready:
        oc = _try_import_open_clip()
        if oc is not None:
            try:
                img_f = _image_features(img)
                toks = oc.get_tokenizer(CLIP_MODEL_NAME)(captions).to(CLIP_DEVICE)
                with torch.no_grad():
                    txt = _clip_model.encode_text(toks)
                    txt = txt / txt.norm(dim=-1, keepdim=True)
                sims = (img_f @ txt.cpu().T).squeeze(0).numpy().tolist()
                order = sorted(range(len(captions)), key=lambda i: sims[i], reverse=True)
                captions = [captions[i] for i in order]
            except Exception as e:
                print(f"[rerank] skipped: {e}")

    # 7) Final labels for UI
    out_labels = ([o["label"] for o in analysis["objects"][:3]]
                  + [s["label"] for s in analysis["scenes"][:3]])
    if not out_labels:
        derived = []
        for w in re.findall(r"[a-z0-9]{3,}", (grounding or "").lower()):
            if w not in _STOP:
                derived.append(w)
        for w in (sanitize_vibe(req.prompt or "")).split():
            derived.append(w)
        seen = set()
        out_labels = [w for w in derived if not (w in seen or seen.add(w))][:6]

    return {
        "analysis": analysis,
        "grounding": grounding,
        "captions": captions[:max(1, min(5, req.n_variants))],
        "hashtags": hashtags[:8],
        "labels": out_labels,
    }

app.include_router(api)
