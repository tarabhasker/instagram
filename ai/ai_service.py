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

HF_TOKEN           = os.getenv("HF_TOKEN", "").strip()
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
app = FastAPI(title="PhotoFeed AI (robust)", version="1.4")

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

# -------------------- HF helpers --------------------
def _img_to_jpeg_bytes(img: Image.Image, max_side=256, quality=85) -> bytes:
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w*scale), int(h*scale)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()

def hf_object_labels(img: Image.Image, top_k=5) -> List[str]:
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

# -------------------- Caption utils & LLM --------------------
# (keep your existing sanitize_vibe, clean_caption, build_hashtags, call_llm_captions, etc.)
# --- omitted for brevity, unchanged from your last version except they remain below ---

# -------------------- HTTP session --------------------
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

# --- analyze and suggest routes ---
# (keep your analyze/suggest implementations, they call hf_object_labels/hf_scene_labels now)

app.include_router(api)
