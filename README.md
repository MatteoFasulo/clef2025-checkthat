# Enhancing Transformer-Based Embeddings with Sentiment for Subjectivity Detection in News

**AI Wizards @ CLEF 2025 CheckThat! Lab – Task 1**
Code and materials for our submission to **Subjectivity Detection in News Articles**, now available on arXiv: [2507.11764](https://arxiv.org/abs/2507.11764) ([arXiv][1]).

---

## Overview

Our system identifies whether a sentence is **subjective** (e.g., opinion-laden) or **objective**—a key component in combating misinformation, improving fact-checking pipelines, and supporting journalists.

We evaluated our approach across:

* **Monolingual** settings (Arabic, German, English, Italian, Bulgarian)
* **Zero-shot transfer** (Greek, Polish, Romanian, Ukrainian)
* **Multilingual** training

We benchmarked:

* **mDeBERTaV3‑base** (multilingual)
* **ModernBERT‑base** (English-focused)
* **Llama 3.2‑1B** (zero-shot LLM baseline)

Our innovation: *augmenting transformer embeddings with sentiment signals*, resulting in **consistent performance gains**, particularly in **subjective F1 score**. Additionally, we implemented robust **decision threshold calibration** to counter class imbalances.

---

## Key Contributions

1. **Sentiment-Augmented Fine-Tuning**
   We enrich typical embedding-based models (mDeBERTaV3 and ModernBERT) by integrating sentiment scores from an auxiliary model, significantly improving subjective sentence detection.

2. **Diverse Model Coverage**

   * **Multilingual** BERT variants across all official CLEF languages
   * **ModernBERT** as a strong English alternative
   * **Llama 3.2‑1B** as a zero-shot baseline

3. **Threshold Calibration for Imbalance**
   A simple yet effective method to tune decision thresholds on each language’s dev data to enhance macro-F1 performance.

---

## 📈 Highlights of Results

* **Sentiment-enhanced models consistently outperform** their non-augmented counterparts in monolingual settings, particularly for the subjective class in English and Italian.
* **BERT-like models surpass LLM baselines** in all scenarios.
* **Zero-shot success**: Topped Greek zero-shot, ranking 1st with Macro F1 = 0.51.
* We achieved **1st–4th place** in most monolingual settings; **corrected results** later showed a 0.68 multilingual run (top-10 ranking), after resolving an official submission quirk.

---

## 🏆 Final Official Results

| Track                 | AI Wizards Macro-F1 | Rank (Corrected) |
| --------------------- | ------------------: | ---------------: |
| Arabic (Mono)         |                0.56 |              5th |
| English (Mono)        |                0.66 |             19th |
| German (Mono)         |                0.77 |              5th |
| Italian (Mono)        |                0.63 |              4th |
| Greek (Zero-shot)     |                0.51 |          **1st** |
| Polish (Zero-shot)    |                0.63 |              4th |
| Romanian (Zero-shot)  |                0.75 |              7th |
| Ukrainian (Zero-shot) |                0.64 |              4th |
| Multilingual          |    0.68 (corrected) |              9th |

> [!NOTE]
> Due to a mistake in our submission process, we inadvertently used a custom train/dev mix instead of the provided dev_test split. As a result, the class distribution was skewed, leading to under-calibrated decision thresholds and a low official multilingual Macro F1 (0.24, 15th place). After re-evaluating with the correct data split, our actual score was Macro F1 = 0.68, which would have placed us 9th overall in the challenge.

---

## 📂 Repository Structure

```clef2025-checkthat/
├── baseline/                # Baseline paraphrase-miniLM model
├── data/                    # Task datasets
├── draw-io/                 # Diagrams for model architecture
├── img/                     # Model diagrams
├── notebooks/               # Notebooks for EDA and training
├── results/                 # Output TSVs & performance logs
├── scorer/                  # Evaluation script
├── src/
│   ├── const.py             # Constants and configurations
│   ├── custom_deberta.py    # Custom DeBERTa model with sentiment
│   ├── custom_modernbert.py # Custom ModernBERT model with sentiment
│   ├── custom_trainer.py    # Custom Trainer with class weights and decision threshold calibration
│   ├── utils/               # Utility functions
│   └── main.py              # Main script for training and evaluation
├── report/                  # Challenge writeups & supplementary docs
├── requirements.txt         # Python dependencies
└── pyproject.toml           # uv lib dependencies
```

---

## 🎯 Getting Started

Clone and install:

```bash
git clone https://github.com/MatteoFasulo/clef2025-checkthat.git
cd clef2025-checkthat
pip install -r requirements.txt
```

or using `uv`:

```bash
uv sync
```

Run English sentiment-augmented model:

```bash
python src/main.py \
  --model_family deberta \
  --language english
```

> [!Tip]
> You can also run the model with sentiment augmentation by adding the `--use_sentiment` flag. Running with `--verbose` will provide additional logging information from Hugging Face Hub and the model.

Evaluate predictions:

The evaluation script requires the ground truth and predicted results in TSV format. All the predictions exported by the `main.py` script are in the results/ folder and can be used directly with the scorer.

```bash
python scorer/evaluate.py \
  -g data/english/test_en_labeled.tsv \
  -p results/test_english_sentiment_predicted.tsv
```

> [!Tip]
> Evaluation can be performed on any language by changing the -g and -p paths accordingly and providing the appropriate ground truth and predictions.

[1]: https://arxiv.org/pdf/2507.11764 "AI Wizards at CheckThat! 2025: Enhancing Transformer-Based ..."
