# 📌 Fine-tuning mDeBERTaV3 & ModernBERT for Subjectivity Detection (Final Version)

## CLEF 2025 CheckThat! Lab – Task 1

Detecting **subjectivity**—distinguishing subjective vs. objective sentences—is crucial for tasks like **fake-news detection** and **fact-checking**. Our final system supports five languages: **Arabic, German, English, Italian, and Bulgarian**.

---

## 🔍 Approaches

### 1. **BERT-Like Models**

* **mDeBERTaV3‑base** ([Hugging Face link](https://huggingface.co/microsoft/mdeberta-v3-base)) for multilingual support.
* **ModernBERT‑base** ([Hugging Face link](https://huggingface.co/answerdotai/ModernBERT-base)) specifically for English language given the results on NLU tasks comparable with DeBERTaV3.
* Fine‑tuned per language, with **sentiment-aware augmentation** for enhanced detection.

### 2. **Large Language Models (LLMs)**

* **Llama 3.2‑1B** ([Hugging Face link](https://huggingface.co/meta-llama/Llama-3.2-1B))
* Evaluated out-of-the-box to benchmark against BERT-like systems.

---

## 📊 Final Results

* **BERT-like models outperformed** LLM baselines across all languages.
* **Sentiment integration** yielded notable improvements in SUBJ F1 for English and Italian.
* **Calibrated decision thresholds** proved critical to managing class imbalance effectively.

---

### 🏅 Challenge Results — Top 3 Scores per Category

| **Setting**                 | **Participant**           | **Macro F1** | **Position** |
|----------------------------|---------------------------|-------------:|-------------:|
| **Monolingual – Arabic**   | aelboua                   | 0.69         | 1st          |
|                            | tomasbernal01             | 0.59         | 2nd          |
|                            | **AI Wizards**           | **0.56**     | **5th**      |
| **Monolingual – English**  | msmadi                    | 0.81         | 1st          |
|                            | kishan_g                  | 0.80         | 2nd          |
|                            | **AI Wizards**           | **0.66**     | **19th**     |
| **Monolingual – German**   | smollab                   | 0.85         | 1st          |
|                            | cepanca_UNAM              | 0.83         | 2nd          |
|                            | **AI Wizards**           | **0.77**     | **5th**      |
| **Monolingual – Italian**  | aelboua                   | 0.69         | 1st          |
|                            | Sumitjais                 | 0.67         | 2nd          |
|                            | **AI Wizards**           | **0.63**     | **4th**      |
| **Zeroshot – Greek**       | **AI Wizards**           | **0.51**     | **1st**      |
|                            | smollab                   | 0.49         | 2nd          |
|                            | KnowThySelf               | 0.49         | 3rd          |
| **Zeroshot – Polish**      | aelboua                   | 0.69         | 1st          |
|                            | Sumitjais                 | 0.67         | 2nd          |
|                            | **AI Wizards**           | **0.63**     | **4th**      |
| **Zeroshot – Romanian**    | msmadi                    | 0.81         | 1st          |
|                            | KnowThySelf               | 0.80         | 2nd          |
|                            | **AI Wizards**           | **0.75**     | **7th**      |
| **Zeroshot – Ukrainian**   | KnowThySelf               | 0.64         | 1st          |
|                            | Ather‑Hashmi              | 0.64         | 2nd          |
|                            | **AI Wizards**           | **0.64**     | **4th**      |
| **Multilingual**           | Bharatdeep_Hazarika       | 0.75         | 1st          |
|                            | kishan_g                  | 0.75         | 1st          |
|                            | **AI Wizards**           | **0.24**     | **15th**     |

Due to a submission error during the challenge phase, our official **multilingual run** was accidentally low (Macro F1 = 0.24, 15th place). However, after the fact, we corrected the submission issue offline and achieved a **corrected Macro F1 of 0.68**, which would have placed **us 9th overall**.

### ⚠️ Threshold Overfitting in the English Model

Upon reviewing the results, we suspect that our English model may have **overfit the decision threshold on the dev set**, which did not generalize well to the test data. This misalignment in threshold calibration likely caused the drop in performance—illustrating a common pitfall where **checkpoint and threshold selection on English dev data fails to translate effectively to unseen test sets**, especially in multilingual tasks.

---

## 📁 Repository Layout

* `baseline/` – baseline model (paraphrase-multilingual-MiniLM-L12-v2) from Sentence Transformers.
* `data/` – task datasets.
* `img/` – images with model pipeline schema and figures.
* `scorer/` – evaluation utilities (`evaluate.py` supports F1 and threshold tuning).
* `requirements.txt` – full Python environment dependencies if using `pip`.
* `pyproject.toml` – uv dependencies if using [`uv`](https://github.com/astral-sh/uv).

---

## 🏗️ System Architecture

![Model Pipeline Schema](img/model_pipeline_schema1.svg)

---

## 💻 Usage Guide

### Setup

```bash
git clone https://github.com/MatteoFasulo/clef2025-checkthat.git
cd clef2025-checkthat
pip install -r requirements.txt
```

we recommend using a [`uv`](https://github.com/astral-sh/uv) environment for better dependency management. After installing `uv`, run:

```bash
uv sync
```

### Run Evaluation (English dev set)

The evaluation script requires the ground truth and predicted results in TSV format. The expected columns are `sentence_id`, and `label` (SUBJ or OBJ).

```bash
python scorer/evaluate.py \
  -g data/english/dev_en.tsv \
  -p results/dev_english_predicted.tsv
```

### Evaluate Sentiment-Augmented Variant

```bash
python scorer/evaluate.py \
  -g data/english/dev_en.tsv \
  -p results/dev_english_sentiment_predicted_.tsv
```

evaluation can be performed on any language by changing the `-g` and `-p` paths accordingly and providing the appropriate ground truth and predictions.

---

## 🔐 License

Distributed under **CC–BY 4.0**. See [LICENSE](LICENSE) for details.
