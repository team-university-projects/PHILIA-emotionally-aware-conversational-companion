"""
eval/ — Benchmarking & Evaluation Framework for PHILIA.

Modules:
    dataset_loader   — HuggingFace dataset loaders (RAVDESS, GoEmotions, FER2013)
    metrics          — accuracy, precision, recall, F1 (macro + per-class)
    confusion_matrix — matplotlib confusion matrix PNG generation
    benchmark_runner — generic runner that wires loaders → predictor → metrics → disk
"""
