# Global configuration dataclasses for PHILIA.

from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).parent


@dataclass
class AudioConfig:
    sample_rate: int = 16_000
    channels: int = 1
    chunk_size: int = 1_024
    device_index: int | None = None


@dataclass
class VideoConfig:
    device_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    capture_fps: int = 15


@dataclass
class ModelConfig:
    device: str = "cuda"
    emotion_device: str = "cpu"
    whisper_model: str = "distil-large-v3"
    audio_emotion_model_name: str = "models/fine_tuned/audio_emotion"
    facial_emotion_model_name: str = "models/fine_tuned/facial_emotion"
    text_emotion_model_name: str = "models/fine_tuned/text_emotion"
    llm_provider: str = "ollama"
    llm_model: str = "llama3.2"
    llm_api_key_env: str = "GEMINI_API_KEY"
    llm_max_tokens: int = 512
    llm_temperature: float = 0.85


@dataclass
class FusionConfig:
    audio_weight: float = 0.15
    facial_weight: float = 0.60
    text_weight: float = 0.25


@dataclass
class TTSConfig:
    provider: str = "edge-tts"
    default_rate: int = 175
    default_pitch: float = 1.0
    default_volume: float = 1.0


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    log_level: str = "INFO"
    eval_results_dir: str = "eval_results"

    @classmethod
    def load(cls) -> "Config":
        from dotenv import load_dotenv
        from pathlib import Path
        load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=False)
        return cls()
