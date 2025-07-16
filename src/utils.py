import os
import csv
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from datasets import Dataset
from transformers import AutoTokenizer


class Subjectivity:
    """
    A class for handling multilingual subjectivity classification datasets.

    This class provides functionality to load, process, and prepare datasets for
    subjective/objective text classification across multiple languages. It supports:
    - Loading and organizing datasets from multiple languages
    - Splitting data into train/dev/test sets
    - Analyzing label distributions
    - Loading pre-trained tokenizers and models
    - Computing class weights for imbalanced datasets

    Attributes:
        seed (int): Random seed for reproducibility
        device (str): Computation device ('cuda' or 'cpu')
        languages (list): List of available languages in the dataset
        dataset (pd.DataFrame): Combined dataset with all languages and splits
        train (pd.DataFrame): Training split of the dataset
        dev (pd.DataFrame): Development split of the dataset
        test (pd.DataFrame): Test split of the dataset
        all_data (dict): Nested dictionary organizing data by language and split
        tokenizer (AutoTokenizer, optional): Hugging Face tokenizer
        model (AutoModelForSequenceClassification, optional): Classification model
    """

    def __init__(self, data_folder: str = "data", seed: int = 42, device: str = "cuda"):
        """
        Initialize the Subjectivity class.

        Args:
            data_folder (str): Directory path containing the dataset files.
            seed (int): Random seed for reproducibility.
            device (str): Device to use for computations ('cuda' or 'cpu').
        """
        self.seed = seed
        self.device = device
        self.languages = [language for language in os.listdir(data_folder)]

        dataset = self.create_dataset(data_folder=data_folder)
        self.dataset = dataset

        train, dev, test = self.get_splits(dataset, print_shapes=True)
        self.train = train
        self.dev = dev
        self.test = test

        self.all_data = self.get_per_lang_dataset()

    def create_dataset(self, data_folder: str = "data"):
        """
        Create a consolidated dataset from files in multiple languages.

        Args:
            data_folder (str): Directory path containing subdirectories for each language.

        Returns:
            pd.DataFrame: Combined dataset with columns for sentence_id, sentence, label,
                          language, and split information.
        """
        dataset = pd.DataFrame(columns=["sentence_id", "sentence", "label", "lang", "split"])
        for language in os.listdir(data_folder):
            for filename in os.listdir(f"{data_folder}{os.sep}{language}"):
                if ".tsv" in filename:
                    abs_path = f"{data_folder}{os.sep}{language}{os.sep}{filename}"
                    df = pd.read_csv(abs_path, sep="\t", quoting=csv.QUOTE_NONE)
                    if "solved_conflict" in df.columns:
                        df.drop(columns=["solved_conflict"], inplace=True)
                    df["lang"] = language
                    df["split"] = Path(filename).stem
                    dataset = pd.concat([dataset, df], axis=0)
        return dataset

    def get_splits(self, dataset: pd.DataFrame, print_shapes: bool = True):
        """
        Split the dataset into training, development, and test sets.

        Args:
            dataset (pd.DataFrame): The combined dataset to split.
            print_shapes (bool): Whether to print the shapes of the resulting splits.

        Returns:
            tuple: A tuple containing three pandas DataFrames (train, dev, test).
        """
        train = dataset[dataset["split"].str.contains("train")].copy()
        dev = dataset[dataset["split"].str.contains("dev_test")].copy()
        test = dataset[dataset["split"].str.contains("_labeled")].copy()

        # encode the target variable to int (0: obj; 1: subj)
        train.loc[:, "label"] = train["label"].apply(lambda x: 0 if x == "OBJ" else 1)
        dev.loc[:, "label"] = dev["label"].apply(lambda x: 0 if x == "OBJ" else 1)
        test.loc[:, "label"] = test["label"].apply(lambda x: 0 if x == "OBJ" else 1)

        # cast to int
        train["label"] = train["label"].astype(int)
        dev["label"] = dev["label"].astype(int)
        test["label"] = test["label"].astype(int)

        if print_shapes:
            print(f"Train: {train.shape}")
            print(f"Dev: {dev.shape}")
            print(f"Test: {test.shape}")

        return train, dev, test

    def get_per_lang_dataset(self):
        """
        Organize the dataset by language and split (train, dev, test).

        Returns:
            dict: A nested dictionary with languages as the outer keys and
                  split names ('train', 'dev', 'test') as inner keys.
                  For example:
                  {
                      'english': {
                          'train': pd.DataFrame,
                          'dev': pd.DataFrame,
                          'test': pd.DataFrame
                      },
                      ...
                  }
        """
        dataset_dict = {}
        for language in self.languages:
            dataset_dict[language] = {}
            # get the train data
            dataset_dict[language]["train"] = self.train[self.train["lang"] == language].copy()
            # get the dev data
            dataset_dict[language]["dev"] = self.dev[self.dev["lang"] == language].copy()
            # get the test data
            dataset_dict[language]["test"] = self.test[self.test["lang"] == language].copy()
        return dataset_dict

    def print_label_distrib(self, dataset: pd.DataFrame):
        """
        Print the normalized distribution of labels in the dataset.

        Args:
            dataset (pd.DataFrame): The dataset containing a 'label' column.

        Returns:
            None: Prints the percentage distribution of each label.
        """
        print(dataset["label"].value_counts(normalize=True))

    def get_class_weights(self, dataset: pd.DataFrame):
        """
        Compute class weights for imbalanced datasets.

        Args:
            dataset (pd.DataFrame): Dataset containing a 'label' column.

        Returns:
            numpy.ndarray: Array of class weights where the index corresponds to the class label.
        """
        class_weights = compute_class_weight("balanced", classes=np.unique(dataset["label"]), y=dataset["label"])
        return class_weights


def tokenize_text(texts: Dataset, tokenizer: AutoTokenizer):
    """
    Tokenize text data using the current tokenizer.

    Args:
        texts (Dataset): Dataset containing text data with a 'sentence' field

    Returns:
        dict: Dictionary with tokenized text features including input_ids,
              attention_mask, and potentially token_type_ids
    """
    return tokenizer(texts["sentence"], padding=True, truncation=True, max_length=256, return_tensors="pt")


def evaluate_metrics(eval_pred):
    """
    Calculate evaluation metrics for subjectivity classification models.

    This function computes various performance metrics for classification results:
    - Accuracy: Overall correctness of predictions
    - Macro-averaged precision, recall, and F1: Averages across both classes with equal weight
    - Class-specific metrics: Precision, recall, and F1 specifically for the subjective class

    Args:
        eval_pred (tuple): Tuple containing (predictions, labels) where:
            - predictions: Raw model outputs/logits with shape (n_samples, n_classes)
            - labels: Ground truth labels with shape (n_samples,)

    Returns:
        dict: Dictionary containing the following metrics:
            - macro_F1: Macro-averaged F1 score across all classes
            - macro_P: Macro-averaged precision across all classes
            - macro_R: Macro-averaged recall across all classes
            - SUBJ_F1: F1 score for the subjective class (label 1)
            - SUBJ_P: Precision for the subjective class
            - SUBJ_R: Recall for the subjective class
            - accuracy: Overall accuracy
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, predictions)
    m_prec, m_rec, m_f1, m_s = precision_recall_fscore_support(labels, predictions, average="macro", zero_division=0)
    p_prec, p_rec, p_f1, p_s = precision_recall_fscore_support(labels, predictions, labels=[1], zero_division=0)

    return {
        "macro_F1": m_f1,
        "macro_P": m_prec,
        "macro_R": m_rec,
        "SUBJ_F1": p_f1[0],
        "SUBJ_P": p_prec[0],
        "SUBJ_R": p_rec[0],
        "accuracy": acc,
    }


def save_predictions(test_data, predictions, filename: str, save_dir: str = "results"):
    """
    Save model predictions to a TSV file with sentence IDs and predicted labels.

    Args:
        test_data: Dataset containing the 'sentence_id' field to match with predictions
        predictions: Array of binary predictions (0 for OBJ, 1 for SUBJ)
        filename: Name of the output file (should end with .tsv)
        save_dir: Directory to save the predictions file (default: 'results')

    Returns:
        str: Full path to the saved predictions file
    """
    os.makedirs(save_dir, exist_ok=True)
    pred_df = pd.DataFrame()
    pred_df["sentence_id"] = test_data["sentence_id"]
    pred_df["label"] = predictions
    pred_df["label"] = pred_df["label"].apply(lambda x: "OBJ" if x == 0 else "SUBJ")

    predictions_filepath = os.path.join(save_dir, filename)
    pred_df.to_csv(predictions_filepath, index=False, sep="\t")

    print(f"Saved predictions into file:", predictions_filepath)
    return predictions_filepath

def fmt(v):
    # if it’s a NumPy array or scalar, extract the single element
    if isinstance(v, np.ndarray) or hasattr(v, "item"):
        v = v.item()
    return f"{v:.4f}"