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
    # Speech-to-text
    whisper_model: str = "base"     # tiny | base | small | medium | large

    # Audio emotion (Wav2Vec-based or CNN — swap path as needed)
    audio_emotion_model_path: str = "models/audio_emotion"

    # Facial emotion (CNN / ResNet)
    facial_emotion_model_path: str = "models/facial_emotion"

    # Text emotion (Transformer)
    text_emotion_model_name: str = "j-hartmann/emotion-english-distilroberta-base"

    # LLM
    llm_provider: str = "openai"    # openai | ollama | gemini
    llm_model: str = "gpt-4o-mini"
    llm_api_key_env: str = "OPENAI_API_KEY"
    llm_max_tokens: int = 512
    llm_temperature: float = 0.8


@dataclass
class FusionConfig:
    # Weights must sum to 1.0
    audio_weight: float = 0.35
    facial_weight: float = 0.35
    text_weight: float = 0.30


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

    @classmethod
    def load(cls) -> "Config":
        """
        Load configuration. Currently returns defaults.
        Future: parse from config.yaml / environment variables.
        """
        return cls()
