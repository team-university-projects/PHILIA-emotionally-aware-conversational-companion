"""
config.py — PHILIA Global Configuration

Single source of truth for:
  - Model paths / identifiers
  - Confidence fusion weights
  - TTS defaults
  - Audio / video capture settings
  - LLM settings
"""

from dataclasses import dataclass, field
from pathlib import Path


ROOT_DIR = Path(__file__).parent


@dataclass
class AudioConfig:
    sample_rate: int = 16_000       # Hz — required by most speech/emotion models
    channels: int = 1               # Mono
    chunk_size: int = 1_024         # Frames per buffer
    device_index: int | None = None # None = system default


@dataclass
class VideoConfig:
    device_index: int = 0           # Webcam index
    frame_width: int = 640
    frame_height: int = 480
    capture_fps: int = 15


@dataclass
class ModelConfig:
    # Inference device for CTranslate2-based models (Whisper).
    # faster-whisper has its own CUDA kernels — no system cuDNN needed.
    device: str = "cuda"

    # Inference device for PyTorch-based emotion models (Wav2Vec2, ViT, BERT).
    # Set to "cuda" only if cuDNN 9.x is properly installed system-wide.
    # If you see "Could not load symbol cudnnGetLibConfig" crashes, keep as "cpu".
    # CPU inference is slower (~3s/clip) but completely stable.
    emotion_device: str = "cpu"

    # Speech-to-text
    whisper_model: str = "distil-large-v3"  # Best accuracy/speed for accents

    # Audio emotion (Wav2Vec2 — HuggingFace model ID)
    audio_emotion_model_name: str = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"

    # Facial emotion (ViT — HuggingFace model ID)
    facial_emotion_model_name: str = "mo-thecreator/vit-Facial-Expression-Recognition"

    # Text emotion (GoEmotions BERT — 28 fine-grained labels)
    text_emotion_model_name: str = "monologg/bert-base-cased-goemotions-original"

    # LLM
    llm_provider: str = "gemini"         # openai | ollama | gemini
    llm_model: str = "gemini-2.5-flash"
    llm_api_key_env: str = "GEMINI_API_KEY"
    llm_max_tokens: int = 512
    llm_temperature: float = 0.8


@dataclass
class FusionConfig:
    # Weights must sum to 1.0
    audio_weight: float = 0.30
    facial_weight: float = 0.30
    text_weight: float = 0.40


@dataclass
class TTSConfig:
    provider: str = "pyttsx3"        # pyttsx3 | elevenlabs | gtts
    default_rate: int = 175          # Words per minute
    default_pitch: float = 1.0       # Multiplier
    default_volume: float = 1.0


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    log_level: str = "INFO"

    # ── Evaluation ─────────────────────────────────────────────────────────────
    # Directory (relative to project root) where benchmark JSON + PNG outputs land.
    # Each run appends a new timestamped file — old results are never overwritten.
    eval_results_dir: str = "eval_results"

    @classmethod
    def load(cls) -> "Config":
        """
        Load configuration. Reads .env from the project root (if present)
        so that API keys set there are available via os.environ.
        Future: parse from config.yaml / environment variables.
        """
        from dotenv import load_dotenv
        from pathlib import Path
        # Load .env from the project root (the directory containing config.py)
        env_path = Path(__file__).parent / ".env"
        load_dotenv(dotenv_path=env_path, override=False)
        return cls()
