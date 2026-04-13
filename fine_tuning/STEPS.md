# PHILIA Fine-Tuning — Steps Guide

## What you have
| File | Trains | Dataset | Expected time (Colab T4) |
|---|---|---|---|
| `01_audio_emotion_finetune.ipynb` | wav2vec2 | MELD audio | ~90 min |
| `02_text_emotion_finetune.ipynb`  | RoBERTa  | MELD text  | ~20 min |
| `03_facial_emotion_finetune.ipynb`| ViT      | FER2013    | ~40 min |

---

## Before You Start (do once)

1. Go to [drive.google.com](https://drive.google.com) and create this folder structure:
   ```
   MyDrive/
   └── PHILIA/
       └── models/          ← models will be auto-saved here
   ```
2. Open [colab.research.google.com](https://colab.research.google.com)

---

## Running Each Notebook

### Step 1 — Upload the notebook
- In Colab: **File → Upload notebook** → select the `.ipynb` file

### Step 2 — Enable GPU
- **Runtime → Change runtime type → T4 GPU → Save**

### Step 3 — Run all cells in order
- **Runtime → Run all** (or press `Ctrl+F9`)
- The notebook will:
  1. Install dependencies
  2. Mount your Google Drive
  3. Download the dataset
  4. Train the model (~see times above)
  5. Evaluate and print accuracy/F1
  6. **Auto-save model to your Drive**

### Step 4 — Download the model folder
After each notebook finishes, go to Google Drive:
- Navigate to `PHILIA/models/audio_emotion/` (or text/facial)
- Right-click the folder → **Download**
- This downloads a `.zip` file

### Step 5 — Extract into PHILIA
Extract the downloaded zip into your PHILIA project:
```
d:\PHILIA\
└── models\
    └── fine_tuned\
        ├── audio_emotion\      ← extract audio_emotion.zip here
        │   ├── config.json
        │   ├── model.safetensors
        │   └── preprocessor_config.json
        ├── text_emotion\       ← extract text_emotion.zip here
        │   ├── config.json
        │   ├── model.safetensors
        │   └── tokenizer.json
        └── facial_emotion\     ← extract facial_emotion.zip here
            ├── config.json
            ├── model.safetensors
            └── preprocessor_config.json
```

### Step 6 — Update config.py
Open `d:\PHILIA\config.py` and update the model paths:
```python
# Before (pre-trained models):
audio_emotion_model_name:  str = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
text_emotion_model_name:   str = "monologg/bert-base-cased-goemotions-original"
facial_emotion_model_name: str = "mo-thecreator/vit-Facial-Expression-Recognition"

# After (your fine-tuned models):
audio_emotion_model_name:  str = "models/fine_tuned/audio_emotion"
text_emotion_model_name:   str = "models/fine_tuned/text_emotion"
facial_emotion_model_name: str = "models/fine_tuned/facial_emotion"
```

### Step 7 — Re-run benchmarks
```powershell
# In d:\PHILIA
.venv\Scripts\python.exe -m eval.run_audio_benchmark
.venv\Scripts\python.exe -m eval.run_text_benchmark
.venv\Scripts\python.exe -m eval.run_facial_benchmark
```
Results will be saved to `eval_results/` — compare the new JSONs against the pre-trained baselines.

---

## Expected Results After Fine-Tuning

| Modality | Pre-trained baseline | Expected after fine-tuning |
|---|---|---|
| Audio | acc=31%, macro-F1=0.28 | acc ~55-65% |
| Text  | acc=67%, macro-F1=0.60 | acc ~70-78% |
| Facial| acc=91%, macro-F1=0.87 | acc ~72-80%* |

*Facial accuracy may appear to drop because we're switching from a 5-class model (which "cheated" by never predicting disgust/fear) to an honest 7-class model. The per-class F1 for disgust and fear will be non-zero for the first time.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA out of memory` | Reduce `per_device_train_batch_size` by half in the training cell |
| Colab disconnects | Re-run from the training cell — checkpoints are saved each epoch |
| Audio download is slow | The MELD raw archive is ~1.3GB. Let it run, it only downloads once |
| `FileNotFoundError` for wav files | Some MELD clips may fail ffmpeg extraction — this is normal and handled automatically |
