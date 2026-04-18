# Extrinsic Evaluation: Lexical Normalization → Emotion Classification

## Dataset: UIT-VSMEC (7 emotion classes)

## Tổng Hợp Kết Quả (Macro F1 trên Test Set)

> Train on RAW → Eval on RAW test vs. Train on NORM → Eval on NORM test

| Model | F1 (Raw→Raw) | F1 (Norm→Norm) | ΔF1 |
| :--- | :---: | :---: | :---: |
| **PhoBERT-base-v2** | 60.26% | 54.82% | -5.44% 🔴 |
| **BiLSTM** | 45.60% | 48.45% | **+2.85%** 🟢 |
| **GRU** | 49.20% | 50.17% | **+0.97%** 🟢 |

---

## Per-Class F1 Comparison

### PhoBERT-base-v2

| Emotion | F1 (Raw) | F1 (Norm) | ΔF1 |
| :--- | :---: | :---: | :---: |
| Enjoyment | 47.9% | 33.3% | -14.6% |
| Sadness | 67.9% | 61.0% | -6.9% |
| Anger | 72.9% | 68.7% | -4.2% |
| Fear | 62.2% | 62.5% | +0.3% |
| Disgust | 58.1% | 50.6% | -7.5% |
| Surprise | 67.5% | 64.8% | -2.7% |
| Other | 45.3% | 42.9% | -2.4% |

### BiLSTM

| Emotion | F1 (Raw) | F1 (Norm) | ΔF1 |
| :--- | :---: | :---: | :---: |
| Enjoyment | 26.5% | 37.5% | +11.0% |
| Sadness | 49.7% | 50.7% | +1.0% |
| Anger | 56.6% | 59.8% | +3.2% |
| Fear | 56.2% | 57.4% | +1.2% |
| Disgust | 36.6% | 38.3% | +1.7% |
| Surprise | 51.3% | 54.8% | +3.5% |
| Other | 42.3% | 40.6% | -1.6% |

### GRU

| Emotion | F1 (Raw) | F1 (Norm) | ΔF1 |
| :--- | :---: | :---: | :---: |
| Enjoyment | 36.1% | 38.3% | +2.2% |
| Sadness | 50.0% | 44.1% | -5.9% |
| Anger | 57.3% | 61.6% | +4.3% |
| Fear | 60.7% | 58.2% | -2.4% |
| Disgust | 44.5% | 42.7% | -1.8% |
| Surprise | 59.6% | 55.5% | -4.1% |
| Other | 36.1% | 50.7% | +14.6% |
