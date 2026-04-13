# PHILIA

PHILIA is an emotionally aware conversational companion that runs locally as a desktop application. It detects the user's emotional state in real time across three modalities — speech, facial expression, and text — fuses them into a single prediction, and generates an empathetic response delivered through a natural neural voice.

---

## How It Works

Each push-to-talk turn runs through the following stages:

1. **Capture** — Microphone and webcam are recorded simultaneously while the spacebar is held.
2. **Transcription** — Audio is transcribed to text via faster-whisper (distil-large-v3, GPU-accelerated).
3. **Emotion Detection** — Three models run in parallel:
   - Audio: Wav2Vec2 fine-tuned on MELD
   - Facial: ViT fine-tuned on FER2013 / AffectNet, applied to sampled video frames via MediaPipe BlazeFace
   - Text: BERT fine-tuned on GoEmotions (28 labels)
4. **Fusion** — Confidence-weighted late fusion (audio 15%, facial 60%, text 25%) produces a single canonical emotion label.
5. **Response Generation** — The fused emotion, tone descriptor, and last 6 turns of conversation history are injected into a personality-keyed system prompt, then sent to a local Ollama LLM (llama3.2).
6. **TTS** — The response is synthesised via edge-tts (Microsoft neural voices) with prosody adjusted per emotion, and played back through the default audio device.

---

## Architecture

```
main.py
  |
  +-- ui/                  Desktop GUI (CustomTkinter)
  |     setup_screen.py    First-run wizard (name, avatar, voice, microphone)
  |     chat_window.py     Chat interface with emotion HUD and animated avatar
  |     loading_screen.py  Splash screen with health-check progress
  |     interface.py       Window lifecycle and thread-safe UI API
  |
  +-- capture/             Push-to-talk recording
  |     audio_capture.py   Microphone via sounddevice
  |     video_capture.py   Webcam via OpenCV (kept warm between turns)
  |
  +-- speech/
  |     transcriber.py     faster-whisper ASR
  |     tts.py             edge-tts synthesis + pyttsx3 offline fallback
  |
  +-- emotion/
  |     audio_emotion.py   Wav2Vec2 inference
  |     facial_emotion.py  ViT inference + MediaPipe face detection
  |     text_emotion.py    GoEmotions BERT inference
  |     fusion.py          Confidence-weighted late fusion
  |
  +-- llm/
  |     prompt_builder.py  System prompt construction with emotion + history
  |     response_generator.py  Ollama / OpenAI / Gemini backend routing
  |
  +-- tone/
  |     tone_mapper.py     Maps emotion label to TTS prosody parameters
  |
  +-- bot/
  |     bot_profile.py     Persona config (name, avatar, voice, mic) saved to JSON
  |
  +-- utils/
        health_check.py    Pre-launch checks (Ollama, models, webcam, mic)
        logger.py          Rotating file + UTF-8 console logger
```

---

## Models

All emotion models are fine-tuned versions stored under `models/fine_tuned/`:

| Modality | Base Model | Dataset |
|---|---|---|
| Audio | Wav2Vec2 (xlsr-53) | MELD |
| Facial | ViT (google/vit-base-patch16-224) | FER2013 + AffectNet |
| Text | BERT (bert-base-cased GoEmotions) | MELD |

Fusion weights were determined via ablation study on the MELD test set.

---

## Setup

**Requirements:** Python 3.10+, CUDA 12 (optional, for GPU acceleration), [Ollama](https://ollama.com) running locally.

```bash
# Install dependencies
pip install -r requirements.txt

# Pull the LLM
ollama pull llama3.2

# Run
python main.py
```

On first launch, the setup wizard collects your companion's name, avatar, voice, and microphone. The profile is saved to `assets/bot_profile.json` and reused on subsequent launches.

**Push-to-talk:** Hold `Space` to record, release to process.

---

## Configuration

All defaults are in `config.py`. Key settings:

| Setting | Default | Description |
|---|---|---|
| `models.llm_provider` | `ollama` | LLM backend (`ollama`, `openai`, `gemini`) |
| `models.llm_model` | `llama3.2` | Model name passed to the provider |
| `models.device` | `cuda` | Device for Whisper (falls back to CPU if CUDA 12 not found) |
| `models.emotion_device` | `cpu` | Device for PyTorch emotion models |
| `fusion.audio_weight` | `0.15` | Fusion weight for audio modality |
| `fusion.facial_weight` | `0.60` | Fusion weight for facial modality |
| `fusion.text_weight` | `0.25` | Fusion weight for text modality |

---

## Building

A PyInstaller spec and PowerShell build script are provided under `build/`:

```powershell
.\build\build.ps1
```

Output is placed in `dist/PHILIA/` as a portable Windows directory bundle.

---

## Project Structure

```
assets/         Avatars, voice catalog, saved bot profile
build/          PyInstaller spec and build script
capture/        Audio and video capture
emotion/        Three emotion models and fusion
eval/           Evaluation scripts
fine_tuning/    Model fine-tuning scripts
llm/            Prompt builder and LLM backends
models/         Fine-tuned model weights
speech/         ASR and TTS
tone/           Emotion-to-prosody mapping
ui/             Desktop GUI
utils/          Logger and health checks
```