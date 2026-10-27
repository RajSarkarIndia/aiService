import io
import logging
import os
import sys
import tempfile

import av
import numpy as np
import pillow_heif
import torch
import torch.nn as nn
from fastapi import APIRouter, FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from huggingface_hub import snapshot_download
from PIL import Image, ImageOps, ImageSequence, UnidentifiedImageError
from pydantic import BaseModel, Field
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModel,
    AutoModelForAudioClassification,
    AutoTokenizer,
    PreTrainedModel,
)

# Enable HEIC / HEIF / AVIF support in Pillow
pillow_heif.register_heif_opener()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai-detector")

# ---------------- Application ----------------

app = FastAPI(title="AI Content Detection API")
router = APIRouter(prefix="/api/ai", tags=["AI Detection"])

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
VIDEO_SAMPLE_FRAMES = int(os.getenv("VIDEO_SAMPLE_FRAMES", "8"))

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

TEXT_MODEL_ID = os.getenv("TEXT_MODEL_ID", "wasitaigeneratedcom/ai-text-detector-small")
IMAGE_MODEL_ID = os.getenv("IMAGE_MODEL_ID", "abacusai/Smaug-AI-Image-Detector")
AUDIO_MODEL_ID = os.getenv("AUDIO_MODEL_ID", "MelodyMachine/Deepfake-audio-detection-V2")


# ---------------- Image & Video Frame Model (Smaug / EmbeddingGemma-2) ----------------

logger.info("Loading image/video detector: %s", IMAGE_MODEL_ID)
image_model_path = snapshot_download(IMAGE_MODEL_ID)
if image_model_path not in sys.path:
    sys.path.insert(0, image_model_path)

from smaug_image_detect import SmaugImageDetect  # noqa: E402

image_detector = SmaugImageDetect.from_pretrained(image_model_path)


# ---------------- Text Model Architecture (DeBERTa-v3 4-class) ----------------

class AIDetectionModel(PreTrainedModel):
    """Model architecture documented by the tropa-mini model publisher."""

    config_class = AutoConfig
    _tied_weights_keys = []  # Transformers 5 compatibility

    @property
    def all_tied_weights_keys(self):
        return {}

    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModel.from_config(config)
        number_of_labels = getattr(config, "detector_num_labels", 1)
        self.classifier = nn.Linear(config.hidden_size, number_of_labels)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        output = self.model(input_ids, attention_mask=attention_mask)
        hidden = output[0]
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return self.classifier(pooled)


logger.info("Loading text detector: %s on %s", TEXT_MODEL_ID, DEVICE)
text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_ID)
text_detector = AIDetectionModel.from_pretrained(TEXT_MODEL_ID)
text_detector.to(DEVICE)
text_detector.eval()


# ---------------- Audio Model (Wav2Vec2 Deepfake Detector) ----------------

logger.info("Loading audio detector: %s on %s", AUDIO_MODEL_ID, DEVICE)
audio_extractor = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL_ID)
audio_detector = AutoModelForAudioClassification.from_pretrained(AUDIO_MODEL_ID)
audio_detector.to(DEVICE)
audio_detector.eval()


# ---------------- DTOs & Helpers ----------------

class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=100_000)


async def read_upload(file: UploadFile) -> bytes:
    data = await file.read(MAX_FILE_SIZE + 1)
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File exceeds 50 MB limit")
    return data


# ---------------- Text Scoring Helper (Transformers 5 Compatible) ----------------

def calculate_text_ai_score(text: str) -> float:
    """Return 100 * (1 - P(human)), averaging 768-token chunks for long text."""

    token_ids = text_tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        raise ValueError("Text contains no valid tokens")

    # 766 content tokens + [CLS] + [SEP] = 768 max sequence length
    chunk_size = 766
    chunks = [token_ids[i : i + chunk_size] for i in range(0, len(token_ids), chunk_size)]

    cls_id = text_tokenizer.cls_token_id
    sep_id = text_tokenizer.sep_token_id
    pad_id = text_tokenizer.pad_token_id if text_tokenizer.pad_token_id is not None else 0

    human_probability_sum = 0.0
    total_weight = 0
    batch_size = 4

    for start in range(0, len(chunks), batch_size):
        chunk_batch = chunks[start : start + batch_size]

        sequences = []
        for chunk in chunk_batch:
            seq = []
            if cls_id is not None:
                seq.append(cls_id)
            seq.extend(chunk)
            if sep_id is not None:
                seq.append(sep_id)
            sequences.append(seq)

        max_len = max(len(seq) for seq in sequences)
        input_ids = torch.full((len(sequences), max_len), pad_id, dtype=torch.long, device=DEVICE)
        attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long, device=DEVICE)

        for idx, seq in enumerate(sequences):
            seq_len = len(seq)
            input_ids[idx, :seq_len] = torch.tensor(seq, dtype=torch.long, device=DEVICE)
            attention_mask[idx, :seq_len] = 1

        with torch.inference_mode():
            logits = text_detector(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(logits, dim=-1)

        for index, chunk in enumerate(chunk_batch):
            chunk_weight = len(chunk)
            human_probability_sum += probabilities[index, 0].item() * chunk_weight
            total_weight += chunk_weight

    if total_weight == 0:
        raise ValueError("Text contains no valid tokens")

    average_human_probability = human_probability_sum / total_weight
    ai_score = 1.0 - average_human_probability
    return round(max(0.0, min(100.0, ai_score * 100.0)), 2)


# ---------------- Image Scoring Helper ----------------

def score_image_bytes(data: bytes) -> float:
    with Image.open(io.BytesIO(data)) as source:
        n_frames = getattr(source, "n_frames", 1)
        if n_frames > 1:
            # Sample up to 4 frames for animated GIF / WebP / APNG
            indices = np.linspace(0, n_frames - 1, num=min(4, n_frames), dtype=int)
            frame_scores = []
            for idx in indices:
                source.seek(int(idx))
                frame = ImageOps.exif_transpose(source).convert("RGB")
                score = float(image_detector(frame))
                if 0.0 <= score <= 1.0:
                    frame_scores.append(score)
            if not frame_scores:
                raise ValueError("Failed to score animated image frames")
            final_prob = float(np.mean(frame_scores))
        else:
            image = ImageOps.exif_transpose(source).convert("RGB")
            final_prob = float(image_detector(image))

    if not 0.0 <= final_prob <= 1.0:
        raise ValueError("Invalid image detector score")

    return round(final_prob * 100.0, 2)


# ---------------- Video Scoring Helper ----------------

def extract_video_frames(data: bytes, max_frames: int = VIDEO_SAMPLE_FRAMES) -> list[Image.Image]:
    """Decode evenly spaced RGB frames from any video container using PyAV."""

    def _decode_from_container(container: av.container.InputContainer) -> list[Image.Image]:
        if not container.streams.video:
            raise ValueError("No video stream found")

        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        total_frames = stream.frames or 0
        decoded_frames: list[Image.Image] = []

        # If total frames are known and small enough, decode and sample uniformly
        if 0 < total_frames <= 450:
            target_indices = set(
                np.linspace(0, total_frames - 1, num=min(max_frames, total_frames), dtype=int).tolist()
            )
            for idx, frame in enumerate(container.decode(stream)):
                if idx in target_indices:
                    decoded_frames.append(frame.to_image().convert("RGB"))
                if len(decoded_frames) >= max_frames:
                    break
            if decoded_frames:
                return decoded_frames

        # For longer videos or unknown frame counts, seek across duration or decode keyframes
        duration = stream.duration
        time_base = float(stream.time_base) if stream.time_base else None
        if duration and time_base and duration > 0:
            timestamps = np.linspace(0.05, 0.95, num=max_frames) * duration
            for ts in timestamps:
                try:
                    container.seek(int(ts), any_frame=False, backward=True, stream=stream)
                    for frame in container.decode(stream):
                        decoded_frames.append(frame.to_image().convert("RGB"))
                        break
                except Exception:
                    continue
            if decoded_frames:
                return decoded_frames

        # Fallback: stream decode with stride
        container.seek(0)
        all_sampled: list[Image.Image] = []
        for idx, frame in enumerate(container.decode(stream)):
            if idx % 10 == 0:
                all_sampled.append(frame.to_image().convert("RGB"))
            if len(all_sampled) >= max_frames * 3:
                break

        if not all_sampled:
            raise ValueError("Unable to decode frames from video")

        indices = np.linspace(0, len(all_sampled) - 1, num=min(max_frames, len(all_sampled)), dtype=int)
        return [all_sampled[int(i)] for i in indices]

    try:
        with av.open(io.BytesIO(data)) as container:
            return _decode_from_container(container)
    except Exception:
        # Fallback for containers requiring seekable disk file descriptors
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
            tmp.write(data)
            tmp.flush()
            with av.open(tmp.name) as container:
                return _decode_from_container(container)


def score_video_bytes(data: bytes) -> float:
    frames = extract_video_frames(data, max_frames=VIDEO_SAMPLE_FRAMES)
    if not frames:
        raise ValueError("No valid video frames extracted")

    scores = []
    for frame in frames:
        prob = float(image_detector(frame))
        if 0.0 <= prob <= 1.0:
            scores.append(prob)

    if not scores:
        raise ValueError("Failed to score video frames")

    # Blend mean and 75th percentile to reliably catch AI segments across video clips
    scores_arr = np.array(scores, dtype=np.float64)
    combined_prob = 0.6 * float(np.mean(scores_arr)) + 0.4 * float(np.percentile(scores_arr, 75))
    return round(max(0.0, min(100.0, combined_prob * 100.0)), 2)


# ---------------- Audio Scoring Helper ----------------

def score_audio_bytes(data: bytes) -> float:
    target_sr = getattr(audio_extractor, "sampling_rate", 16000) or 16000

    with av.open(io.BytesIO(data)) as container:
        if not container.streams.audio:
            raise ValueError("No audio stream found")

        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=target_sr)

        chunks: list[np.ndarray] = []
        total_samples = 0
        max_samples = target_sr * 60  # Cap analysis to 60 seconds

        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                arr = resampled.to_ndarray()[0]
                chunks.append(arr)
                total_samples += len(arr)
            if total_samples >= max_samples:
                break

    if not chunks:
        raise ValueError("Audio file contains no decodable samples")

    waveform = np.concatenate(chunks, axis=0)[:max_samples]

    # Split into 10-second windows for stable Wav2Vec2 inference
    window_size = target_sr * 10
    windows = [
        waveform[i : i + window_size]
        for i in range(0, len(waveform), window_size)
        if len(waveform[i : i + window_size]) >= int(target_sr * 0.5)
    ]
    if not windows:
        windows = [waveform]

    # Resolve which label index represents "fake" / AI
    id2label = getattr(audio_detector.config, "id2label", {}) or {}
    fake_idx = 0
    for idx, label in id2label.items():
        if str(label).lower() in {"fake", "spoof", "ai", "synthetic"}:
            fake_idx = int(idx)
            break

    fake_probs = []
    for win in windows:
        inputs = audio_extractor(
            win,
            sampling_rate=target_sr,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.inference_mode():
            logits = audio_detector(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            fake_probs.append(probs[0, fake_idx].item())

    avg_fake = float(np.mean(fake_probs))
    return round(max(0.0, min(100.0, avg_fake * 100.0)), 2)


# ---------------- Routes ----------------

@router.post("/image/detect")
async def detect_image(file: UploadFile = File(...)):
    content_type = (file.content_type or "application/octet-stream").lower()
    try:
        if not (content_type.startswith("image/") or content_type == "application/octet-stream"):
            raise HTTPException(status_code=415, detail=f"Unsupported image type: {content_type}")

        data = await read_upload(file)
        ai_score = await run_in_threadpool(score_image_bytes, data)
        return {"aiScore": ai_score}

    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid or corrupted image file") from exc
    except Exception as exc:
        logger.exception("Image detection failed")
        raise HTTPException(status_code=500, detail="Image detection failed") from exc
    finally:
        await file.close()


@router.post("/video/detect")
async def detect_video(file: UploadFile = File(...)):
    content_type = (file.content_type or "application/octet-stream").lower()
    try:
        if not (content_type.startswith("video/") or content_type == "application/octet-stream"):
            raise HTTPException(status_code=415, detail=f"Unsupported video type: {content_type}")

        data = await read_upload(file)
        ai_score = await run_in_threadpool(score_video_bytes, data)
        return {"aiScore": ai_score}

    except HTTPException:
        raise
    except (av.AVError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid or unreadable video file") from exc
    except Exception as exc:
        logger.exception("Video detection failed")
        raise HTTPException(status_code=500, detail="Video detection failed") from exc
    finally:
        await file.close()


@router.post("/audio/detect")
async def detect_audio(file: UploadFile = File(...)):
    content_type = (file.content_type or "application/octet-stream").lower()
    try:
        if not (
            content_type.startswith("audio/")
            or content_type in {"video/webm", "video/mp4", "application/octet-stream"}
        ):
            raise HTTPException(status_code=415, detail=f"Unsupported audio type: {content_type}")

        data = await read_upload(file)
        ai_score = await run_in_threadpool(score_audio_bytes, data)
        return {"aiScore": ai_score}

    except HTTPException:
        raise
    except (av.AVError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid or unreadable audio file") from exc
    except Exception as exc:
        logger.exception("Audio detection failed")
        raise HTTPException(status_code=500, detail="Audio detection failed") from exc
    finally:
        await file.close()


@router.post("/text/detect")
async def detect_text(request: TextRequest):
    try:
        ai_score = await run_in_threadpool(calculate_text_ai_score, request.text)
        return {"aiScore": ai_score}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Text detection failed")
        raise HTTPException(status_code=500, detail="Text detection failed") from exc


app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8001")))
