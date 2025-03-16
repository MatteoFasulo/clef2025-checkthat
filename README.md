# 📌 **Fine-tuning of mDeBERTaV3 & ModernBERT for Subjectivity Detection**

[**CheckThat! Lab 2025 - Task 1**](https://gitlab.com/checkthat_lab/clef2025-checkthat-lab/-/tree/main/task1)  

This project tackles the problem of **subjectivity detection in natural language** 🌐—a fundamental task for applications like **fake news detection** ❌📰 and **fact-checking** ✅. The goal is to classify sentences as **subjective (SUBJ)** or **objective (OBJ)** across various languages: **Arabic, German, English, Italian, and Bulgarian**.

---

## 🔍 **Approaches**  
We employ **two primary approaches** for subjectivity detection:

### 1. **BERT-like Models (mDeBERTaV3 & ModernBERT)**  
- [mDeBERTaV3-base](https://huggingface.co/microsoft/mdeberta-v3-base) 📖
- [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) 🔍
- Fine-tuned on **language-specific datasets** with integrated **sentiment information** 💬 for enhanced performance.

### 2. **Large Language Models (LLMs)**  
- [Llama3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) 🦙
- Evaluated on its ability to capture subjectivity from general knowledge representations.

---

## 📊 **Key Findings**  
- **BERT-like models** exhibit superior performance in capturing nuanced information compared to LLMs.  
- **Incorporating sentiment information** improves the **subjective F1 score** significantly for **English** and **Italian**; less so for other languages.  
- **Decision threshold calibration** is essential for improving performance when handling **imbalanced label distributions**.  

---

## 📁 **Project Structure**  
- **Data Preparation:** 📂 Data augmentation using sentiment scores, tokenization, and preprocessing.  
- **Model Training:** 🔧 Fine-tuning mDeBERTaV3, ModernBERT, and Llama3.2-1B.  
- **Evaluation:** 📈 Evaluation metrics include **macro-average F1 score** and **SUBJ F1 score** with focus on **threshold optimization**.

---

## 🏗️ **Architecture Overview**  
The architecture of the proposed system is illustrated below:

<p align="center">
  <img src="img/model_pipeline_schema1.svg" width="300" />
</p>

---

## 💻 **Requirements**  
- Python 3.x 🐍  
- PyTorch 🔥  
- Hugging Face Transformers 🤗  
- Dependencies specified in `requirements.txt` 📋

---

## 📦 **Installation**  
1. Clone the repository:  
```bash
 git clone https://github.com/MatteoFasulo/clef2025-checkthat.git
 cd clef2025-checkthat
```

2. Install dependencies:  
```bash
pip install -r requirements.txt
```

---

## 🔬 **Evaluation**  
To evaluate the model performance on the development set for **English**, use:
```bash
python scorer/evaluate.py -g data/english/dev_en.tsv -p results/dev_english_predicted.tsv
```

To evaluate the **sentiment-enhanced model**:
```bash
python scorer/evaluate.py -g data/english/dev_en.tsv -p results/dev_english_sentiment_predicted_.tsv
```

---

## 🔗 **External Resources**  
- [GitHub Repository](https://github.com/MatteoFasulo/clef2025-checkthat) 📂  
- [Dataset](https://gitlab.com/checkthat_lab/clef2025-checkthat-lab/-/tree/main/task1/data) 🗃️  

---

## ✅ **Conclusion**  
This project highlights the **effectiveness of BERT-like models** for subjectivity detection and emphasizes the importance of **handling linguistic variability and class imbalance**. Future work will focus on enhancing **LLM performance** and addressing challenges identified in the **error analysis**.

---

## 📜 **License**  
Licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

