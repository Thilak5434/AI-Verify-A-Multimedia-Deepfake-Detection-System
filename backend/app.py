# ============================================
# TECH STACK: FastAPI, PyTorch, TorchVision, Pillow, OpenCV, scikit-learn
# ============================================

import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Dict, Tuple

# LIBRARY: FastAPI - REST API framework
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# CORE DETECTION MODULES (imported from core.py)
from core import (
    build_inference_pack,          # IMAGE DETECTION: ResNet50 with MC Dropout
    load_audio_model,              # AUDIO DETECTION: AudioClassifier module
    load_video_model,              # VIDEO DETECTION: CNN-LSTM module
    predict_audio_file,            # FUNCTION: Audio classification inference
    predict_image_bytes,           # FUNCTION: Image classification inference
    predict_video_file,            # FUNCTION: Frame-based video detection (ResNet)
    predict_video_with_cnnlstm,    # FUNCTION: Temporal video detection (CNN-LSTM)
    provenance_hash,               # FUNCTION: Integrity checking and hashing
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================
# FILE PURPOSE: HTTP API Server
# This file exposes REST endpoints for the frontend.
# Frontend uploads media → API runs detection models → returns JSON results
# ENDPOINTS: /image, /video, /audio (for different media types)
# ============================================

app = FastAPI(title="AI Generated Media Detector", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# MODEL LOADING: Initialize detection models at startup
ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT_DIR / "dataset"
MODEL_FILE = DATASET_DIR / "media_detector.pt"
VIDEO_MODEL_FILE = DATASET_DIR / "video_cnn_lstm.pt"

# ============================================
# IMAGE DETECTION: ResNet50 backbone loaded here
# LIBRARY: TorchVision, PyTorch
# ALGORITHM: ResNet50 with MC Dropout for uncertainty estimation
# ============================================
try:
    PACK = build_inference_pack(model_dir=str(DATASET_DIR), arch="resnet50")
    PACK_MTIME = MODEL_FILE.stat().st_mtime if MODEL_FILE.exists() else 0.0
    logger.info("✓ Image model loaded successfully (ResNet50)")
except Exception as e:
    logger.error(f"Failed to load image model: {e}")
    PACK = None
    PACK_MTIME = 0.0

# ============================================
# VIDEO DETECTION: CNN-LSTM model loaded here
# LIBRARY: PyTorch, TorchVision
# ALGORITHM: CNN (ResNet50) + LSTM for temporal video understanding
# LOCATION: predict_video_with_cnnlstm() in core.py
# ============================================
VIDEO_MODEL = load_video_model(str(VIDEO_MODEL_FILE), device=PACK.device if PACK else None)
VIDEO_MTIME = VIDEO_MODEL_FILE.stat().st_mtime if VIDEO_MODEL_FILE.exists() else 0.0

# ============================================
# AUDIO DETECTION: AudioClassifier model loaded here
# LIBRARY: PyTorch, librosa (audio processing)
# LOCATION: predict_audio_file() in core.py
# ============================================
AUDIO_MODEL_FILE = DATASET_DIR / "audio_classifier.pt"
AUDIO_MODEL = load_audio_model(str(AUDIO_MODEL_FILE), device=PACK.device if PACK else None)
AUDIO_MTIME = AUDIO_MODEL_FILE.stat().st_mtime if AUDIO_MODEL_FILE.exists() else 0.0

# PERFORMANCE: In-memory result caching to avoid reprocessing same files
# Cache key = model hash + file hash, value = (timestamp, detection result)
RESULT_CACHE: Dict[str, Tuple[float, dict]] = {}
CACHE_TTL_SEC = 120.0


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _confidence_range(out: dict) -> Tuple[float, float]:
    # Prefer explicit uncertainty interval if present.
    interval = out.get("uncertainty_interval")
    if isinstance(interval, (list, tuple)) and len(interval) == 2:
        return float(interval[0]), float(interval[1])
    c = float(out.get("confidence", 0.5))
    s = float(out.get("uncertainty_std", 0.0))
    if s > 0:
        return max(0.0, c - s), min(1.0, c + s)
    return c, c


def _metrics_ranges_for_media(media_kind: str) -> dict:
    # Return model-level metric ranges for display in the UI.
    if media_kind == "video":
        met = _load_json(DATASET_DIR / "video_metrics.json")
        val = met.get("val_metrics", {}) if isinstance(met, dict) else {}
        f1_val = float(val.get("f1", 0.0)) if val else 0.0
        best_f1 = float(met.get("best_f1", f1_val)) if isinstance(met, dict) else f1_val
        recall_val = float(val.get("recall", 0.0)) if val else 0.0
        return {
            "f1": (min(f1_val, best_f1), max(f1_val, best_f1)),
            "recall": (recall_val, recall_val),
            "source": "video_metrics.json",
        }
    if media_kind == "image":
        met = _load_json(DATASET_DIR / "metrics.json")
        un = met.get("val_uncalibrated", {}) if isinstance(met, dict) else {}
        ca = met.get("val_calibrated", {}) if isinstance(met, dict) else {}
        f1_vals = [float(v) for v in [un.get("f1", 0.0), ca.get("f1", 0.0)]]
        rc_vals = [float(v) for v in [un.get("recall", 0.0), ca.get("recall", 0.0)]]
        return {
            "f1": (min(f1_vals), max(f1_vals)),
            "recall": (min(rc_vals), max(rc_vals)),
            "source": "metrics.json",
        }
    return {"f1": None, "recall": None, "source": None}


def _deepfake_type(media_kind: str, filename: str, out: dict) -> str:
    # Heuristic classification based on media kind and label.
    label = (out.get("label") or "").lower()
    is_fake = "ai" in label or "fake" in label
    if not is_fake:
        return "Not AI (Real)"
    if media_kind == "audio":
        return "AI voice cloning"
    if media_kind == "image":
        return "GAN-generated image"
    if media_kind == "video":
        name = (filename or "").lower()
        if any(k in name for k in ["lip", "sync", "dub", "voice", "talk"]):
            return "Lip-sync manipulation"
        return "Face swap"
    return "Unknown"



def _refresh_if_models_changed() -> None:
    """Reload models automatically when weight files change on disk."""
    global PACK, PACK_MTIME, VIDEO_MODEL, VIDEO_MTIME, AUDIO_MODEL, AUDIO_MTIME, RESULT_CACHE
    
    m = MODEL_FILE.stat().st_mtime if MODEL_FILE.exists() else 0.0
    if PACK is None or m != PACK_MTIME:
        # Image model was updated (for example after training).
        try:
            PACK = build_inference_pack(model_dir=str(DATASET_DIR), arch="resnet50")
            PACK_MTIME = m
            RESULT_CACHE = {}
            logger.info("✓ Reloaded image model")
        except Exception as e:
            logger.error(f"Failed to reload image model: {e}")

    vm = VIDEO_MODEL_FILE.stat().st_mtime if VIDEO_MODEL_FILE.exists() else 0.0
    if vm != VIDEO_MTIME:
        # Video model was updated (or removed/added).
        VIDEO_MODEL = load_video_model(str(VIDEO_MODEL_FILE), device=PACK.device if PACK else None)
        VIDEO_MTIME = vm
        RESULT_CACHE = {}
        logger.info(f"✓ Reloaded video model" if VIDEO_MODEL else "✓ Video model not available")

    am = AUDIO_MODEL_FILE.stat().st_mtime if AUDIO_MODEL_FILE.exists() else 0.0
    if am != AUDIO_MTIME:
        # Audio model was updated (or removed/added).
        AUDIO_MODEL = load_audio_model(str(AUDIO_MODEL_FILE), device=PACK.device if PACK else None)
        AUDIO_MTIME = am
        RESULT_CACHE = {}
        logger.info(f"✓ Reloaded audio model" if AUDIO_MODEL else "✓ Audio model not available")


def _fuse_video_scores(base: dict, seq: dict) -> None:
    """
    ALGORITHM: Score fusion for video detection
    PURPOSE: Combine frame-level (ResNet50) and sequence-level (CNN-LSTM) predictions
    METHOD: Weighted average with threshold-based decision
    RESULT: More robust video deepfake detection
    """
    p_frame = float(base["fake_probability"])
    p_video = float(seq["fake_probability"])
    probs = base.get("temporal", {}).get("frame_probs", [])
    n = len(probs) or 1
    med = sorted(probs)[n // 2] if probs else p_frame
    high_ratio = sum(v >= 0.6 for v in probs) / n if probs else float(p_frame >= 0.6)
    
    # Give the video model more weight so AI-generated videos are less likely to be missed.
    p_fake = 0.55 * p_frame + 0.45 * p_video

    # Trigger fake if either signal is confidently high.
    trigger_fake = (p_video >= 0.62) or (p_frame >= 0.70) or (med >= 0.60 and high_ratio >= 0.35)
    cls = 1 if trigger_fake else int(p_fake >= float(base.get("decision_threshold", 0.5)))
    
    base.update({
        "label_id": cls,
        "label": PACK.class_names[cls],
        "confidence": abs(p_fake - 0.5) * 2,
        "fake_probability": p_fake,
        "fusion": {
            "frame_model_p_fake": p_frame,
            "video_model_p_fake": p_video,
            "frame_median_p_fake": med,
            "frame_high_ratio": high_ratio,
        },
    })



def _analyze_video(raw: bytes, suffix: str) -> dict:
    """
    FUNCTION: VIDEO DETECTION Pipeline
    STEP 1: ResNet50 frame-level analysis (detect manipulation per frame)
    STEP 2: CNN-LSTM temporal analysis (detect temporal inconsistencies)
    STEP 3: Score fusion (combine frame + sequence predictions)
    LIBRARY: OpenCV (video decoding), PyTorch (inference)
    """
    # Write uploaded bytes to a temporary file because video decoders work with file paths.
    with tempfile.NamedTemporaryFile(delete=True, suffix=suffix or ".mp4") as f:
        f.write(raw)
        f.flush()
        # Frame-level model prediction (ResNet-based).
        out = predict_video_file(PACK, f.name, n_frames=20)
        if VIDEO_MODEL is not None:
            # Optional sequence model prediction + fusion for stronger temporal reasoning.
            seq = predict_video_with_cnnlstm(VIDEO_MODEL, PACK.device, f.name, n_frames=12)
            _fuse_video_scores(out, seq)
        return out


def _analyze_audio(raw: bytes, suffix: str) -> dict:
    """
    FUNCTION: AUDIO DETECTION Pipeline
    ALGORITHM: AudioClassifier CNN on Mel-spectrograms
    PROCESSING: Convert audio file to Mel-spectrogram → Run CNN inference
    LIBRARY: librosa (audio processing), PyTorch (inference)
    """
    # Use delete=False to prevent file from being deleted before librosa finishes processing.
    import atexit
    temp_f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".wav")
    temp_f.write(raw)
    temp_f.close()
    
    try:
        device = PACK.device if PACK else None
        result = predict_audio_file(temp_f.name, model_path=AUDIO_MODEL, device=device)
        return result
    finally:
        # Clean up temp file after processing
        try:
            Path(temp_f.name).unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to delete temp audio file: {e}")


def _payload(out: dict, h: str, media_kind: str, filename: str) -> dict:
    # Keep response format stable for the frontend.
    c0, c1 = _confidence_range(out)
    met = _metrics_ranges_for_media(media_kind)
    f1_range = met.get("f1")
    recall_range = met.get("recall")
    return {
        "result": out.get("label", "Unknown"),
        "confidence": round(float(out.get("confidence", 0.5)), 4),
        "fake_probability": round(float(out.get("fake_probability", 0.5)), 4),
        "uncertainty_std": round(float(out.get("uncertainty_std", 0.0)), 4),
        "uncertainty_interval": out.get("uncertainty_interval"),
        "confidence_range": [round(c0, 4), round(c1, 4)],
        "f1_range": [round(float(f1_range[0]), 4), round(float(f1_range[1]), 4)] if f1_range else None,
        "recall_range": [round(float(recall_range[0]), 4), round(float(recall_range[1]), 4)] if recall_range else None,
        "metrics_source": met.get("source"),
        "deepfake_type": _deepfake_type(media_kind, filename, out),
        "hash_sha256": h,
        "cached": False,
        "details": out,
    }


@app.get("/health")
def health():
    # Small health endpoint so we can quickly check server status.
    return {
        "ok": PACK is not None,
        "model": str(MODEL_FILE.name),
        "version": "3.0.0 (ResNet50)",
    }


@app.post("/analyze")
async def analyze(media: UploadFile = File(...)):
    """
    API ENDPOINT: /analyze
    PURPOSE: Main detection endpoint for images, videos, and audio
    ROUTE LOGIC:
      - IMAGE: ResNet50 + MC Dropout confidence estimation
      - VIDEO: ResNet50 frames + CNN-LSTM temporal + score fusion
      - AUDIO: Mel-spectrogram + AudioClassifier CNN
    CACHING: Results cached by model version + file hash
    RESPONSE: JSON with classification, confidence, uncertainty
    """
    # 1) Make sure latest models are in memory.
    _refresh_if_models_changed()
    
    if PACK is None:
        return JSONResponse({"result": "Model not loaded", "error": "Server model failed to load"}, status_code=503)

    # 2) Read uploaded file bytes.
    raw = await media.read()
    if not raw:
        return JSONResponse({"result": "Invalid file", "error": "empty upload"}, status_code=400)

    # 3) Detect media type (image/video/audio)
    ctype = (media.content_type or "").lower()
    suffix = Path(media.filename or "").suffix.lower()
    is_video = ctype.startswith("video/") or suffix in {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    media_kind = "video" if is_video else ("image" if ctype.startswith("image/") else ("audio" if ctype.startswith("audio/") else "other"))

    # 4) Generate file hash for caching
    h = provenance_hash(raw)
    key = f"{PACK_MTIME}:{VIDEO_MTIME}:{h}"
    now = time.time()

    # 5) Return cached result if we already analyzed same file with same model version.
    if key in RESULT_CACHE and (now - RESULT_CACHE[key][0] <= CACHE_TTL_SEC):
        out = dict(RESULT_CACHE[key][1])
        out["cached"] = True
        return out

    try:
        # 6) Route to the correct detection pipeline by MIME type.
        if ctype.startswith("image/"):
            # IMAGE DETECTION: ResNet50 inference
            res = predict_image_bytes(PACK, raw)
        elif is_video:
            # VIDEO DETECTION: Frame-based + temporal analysis
            res = _analyze_video(raw, suffix)
        elif ctype.startswith("audio/"):
            # AUDIO DETECTION: Mel-spectrogram + AudioClassifier
            res = _analyze_audio(raw, suffix)
        else:
            return JSONResponse({"result": "Unsupported media type", "error": ctype}, status_code=415)
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return JSONResponse({"result": "Analysis failed", "error": str(e)}, status_code=500)

    if "fake_probability" in res:
        p_fake = float(res.get("fake_probability", 0.0))
        threshold = float(res.get("decision_threshold", PACK.threshold))
        logger.info(f"p_fake={p_fake:.4f} threshold={threshold:.4f}")

    payload = _payload(res, h, media_kind, media.filename or "")
    RESULT_CACHE[key] = (now, payload)

    # 7) Keep cache small in long-running sessions.
    if len(RESULT_CACHE) > 300:
        stale = [k for k, (ts, _) in RESULT_CACHE.items() if now - ts > CACHE_TTL_SEC]
        for k in stale:
            RESULT_CACHE.pop(k, None)

    # 8) Save last result for debugging / UI display if needed.
    DATASET_DIR.mkdir(exist_ok=True)
    (DATASET_DIR / "last_result.json").write_text(json.dumps(payload, indent=2))
    
    logger.info(f"Analysis: {payload['result']} (confidence: {payload['confidence']})")
    return payload


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=3000, reload=False)
