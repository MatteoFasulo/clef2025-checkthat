# Fine-tuning of mDeBERTaV3 & ModernBERT for Subjectivity Detection

## Abstract

Detecting subjectivity in natural language is crucial for various NLP tasks, including fake news detection and fact-checking. However, achieving robust subjectivity detection across different languages remains a challenging task due to the complexity of linguistic diversity and cultural differences. 

In this project, we present our system developed for 2025 CheckThat! Lab task 1 on subjectivity detection. We employ two distinct approaches: one using BERT-like architectures ([mDeBERTaV3-base](https://huggingface.co/microsoft/mdeberta-v3-base), [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base)) and the other using Large Language Models (LLMs) ([Llama3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)). 

The results indicate that BERT-like models exhibit superior performance in capturing nuanced information and accurately determining subjectivity compared to LLMs. Furthermore, we find that incorporating sentiment information yields significant improvements in the subjective F1 score for English and Italian languages, whereas the improvement is marginal for the others. Additionally, our decision threshold calibration procedure has a substantial impact on performance for languages with imbalanced label distributions, while providing only marginal gains for more balanced languages.

## Overview
This project focuses on developing a system for subjectivity detection in natural language, specifically for the 2025 CheckThat! Lab Task 1. The goal is to classify sentences from news articles as either subjective (SUBJ) or objective (OBJ) across multiple languages, including Arabic, German, English, Italian, and Bulgarian. We employ two distinct approaches: one using BERT-like architectures (mDeBERTaV3 and ModernBERT) and the other using Large Language Models (LLMs) like Llama3.2-1B.

## Key Findings
- BERT-like models, particularly mDeBERTaV3, outperform LLMs in capturing nuanced information for subjectivity detection.
- Incorporating sentiment information significantly improves the subjective F1 score for English and Italian, while the improvement is marginal for other languages.
- Decision threshold calibration is crucial for enhancing performance in languages with imbalanced label distributions.

## Project Structure
The project is organized as follows:
- **Data Preparation**: Includes data augmentation with sentiment scores and tokenization.
- **Model Training**: Fine-tuning of mDeBERTaV3, ModernBERT, and Llama3.2-1B on language-specific datasets.
- **Evaluation**: Performance metrics include macro-average F1 score and positive class (SUBJ) F1 score, with a focus on threshold optimization.

## Architecture

The architecture of our system is illustrated below:

<p align="center">
  <img src="img/model_pipeline_schema1.svg" width="300" />
</p>

## Requirements

To run this project, you will need:
- Python 3.x
- PyTorch
- Hugging Face Transformers library
- Other dependencies specified in the `requirements.txt` file

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/MatteoFasulo/clef2025-checkthat.git
   cd clef2025-checkthat
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Evaluation

To evaluate the performance of the model on the development set for the english language, run the following command:

```bash
python scorer/evaluate.py -g data/english/dev_en.tsv -p results/dev_english_predicted.tsv
```

if you want to evaluate the performance of the model which encompasses the sentiment information, run the following command:

```bash
python scorer/evaluate.py -g data/english/dev_en.tsv -p results/dev_english_sentiment_predicted_.tsv
```

## Links to External Resources

- [GitHub Repository](https://github.com/MatteoFasulo/clef2025-checkthat)
- [Dataset](https://gitlab.com/checkthat_lab/clef2025-checkthat-lab/-/tree/main/task1/data)

## Conclusion

This project demonstrates the effectiveness of BERT-like models for subjectivity detection and highlights the importance of considering linguistic variability and class imbalance in multilingual settings. Future work will focus on improving LLM performance and addressing the challenges identified in the error analysis.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
