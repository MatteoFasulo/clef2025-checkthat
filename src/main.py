import os
import sys
import time
from typing import Literal
import logging
from tabulate import tabulate

from tqdm import tqdm

import numpy as np
import torch

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset
from transformers import DataCollatorWithPadding, TrainingArguments, DebertaV2Config, ModernBertConfig, AutoModelForSequenceClassification, AutoTokenizer

import constants as const
from custom_trainer import CustomTrainer
from custom_deberta import CustomModel
from custom_modernbert import CustomModernBertModel
from utils import Subjectivity, tokenize_text, evaluate_metrics, save_predictions, fmt

logger = logging.getLogger(__name__)

# Store the results and predictions
results = {}
predictions_dict = {}

def run(
    detector: Subjectivity,
    model_family: Literal["deberta", "modernbert"],
    model_card: str,
    pretrained_card: str,
    language: str = "english",
    use_sentiment: bool = False,
):
    # 1) prepare tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_card)
    except Exception as e:
        logger.error("Error loading tokenizer, make sure the model card is correct: %s", e)
        sys.exit(-1)

    # 2) (optional) extract sentiment features
    if use_sentiment:
        for split in ["train", "dev", "test"]:
            # extract sentiment scores for each split
            df = detector.all_data[language][split]

            # batch the texts to avoid memory issues
            batched_results = const.SENT_PIPE(df["sentence"].tolist(), batch_size=const.BATCH_SIZE)

            # separate the results into positive, neutral, and negative scores
            positives, neutrals, negatives = [], [], []
            for result in batched_results:
                # Sort by label to guarantee order, just in case
                result_map = {entry["label"]: entry["score"] for entry in result}
                positives.append(result_map.get("positive", 0.0))
                neutrals.append(result_map.get("neutral", 0.0))
                negatives.append(result_map.get("negative", 0.0))

            # add the sentiment scores to the dataframe
            df["positive"] = positives
            df["neutral"] = neutrals
            df["negative"] = negatives
            detector.all_data[language][split] = df

        # pick the correct HF config
        ConfigClass = {"deberta": DebertaV2Config, "modernbert": ModernBertConfig}[model_family]

        config = ConfigClass.from_pretrained(
            model_card,
            num_labels=2,
            id2label={0: "OBJ", 1: "SUBJ"},
            label2id={"OBJ": 0, "SUBJ": 1},
            output_attentions=False,
            output_hidden_states=False,
        )

        # and use the custom multi-input model
        ModelClass = {"deberta": CustomModel, "modernbert": CustomModernBertModel}[model_family]

        model = ModelClass(config=config, sentiment_dim=3, num_labels=2).from_pretrained(pretrained_card)
    else:
        # no sentiment, just vanilla HF model
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_card,
            num_labels=2,
            id2label={0: "OBJ", 1: "SUBJ"},
            label2id={"OBJ": 0, "SUBJ": 1},
            output_attentions=False,
            output_hidden_states=False,
        )

    logger.info("Loaded model from %s", pretrained_card)
    logger.info("config.name_or_path = %s", model.config._name_or_path)

    # 3) build datasets + tokenize
    splits = {}
    for split in ["train", "dev", "test"]:
        df = detector.all_data[language][split]
        ds = Dataset.from_pandas(df)
        splits[split] = ds.map(tokenize_text, batched=True, fn_kwargs={"tokenizer": tokenizer})

    # 4) weights, data‑collator, training args
    class_weights = detector.get_class_weights(detector.all_data[language]["train"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    output_dir = f"{model_family}-{language}" + ("-sentiment" if use_sentiment else "")

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=const.BATCH_SIZE,
        per_device_eval_batch_size=const.BATCH_SIZE,
        learning_rate=const.LR,
        num_train_epochs=const.EPOCHS,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=splits["train"],
        eval_dataset=splits["dev"],
        data_collator=data_collator,
        compute_metrics=evaluate_metrics,
        class_weights=class_weights,
    )

    # 5) thresholding, predict & save, compute metrics
    best_thr = trainer.compute_best_threshold(dataset=splits["dev"])
    for split in ["dev", "test"]:
        pred_info = trainer.predict(dataset=splits[split], threshold=best_thr)
        save_predictions(
            splits[split],
            pred_info.predictions,
            filename=f"{split}_{language}"
                     + ("_sentiment_predicted.tsv" if use_sentiment else "_predicted.tsv"),
            save_dir=const.RESULTS_PATH,
        )
        if split == "test":
            labels = pred_info.label_ids
            preds = pred_info.predictions
            acc = accuracy_score(labels, preds)
            m_prec, m_rec, m_f1, m_s = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
            p_prec, p_rec, p_f1, p_s = precision_recall_fscore_support(labels, preds, labels=[1], zero_division=0)
            results_key = f"{model_family}-{language}" + ("-sent" if use_sentiment else "") + "-thr"
            results[results_key] = {
                'macro_F1': m_f1,
                'macro_P': m_prec,
                'macro_R': m_rec,
                'SUBJ_F1': p_f1[0],
                'SUBJ_P': p_prec[0],
                'SUBJ_R': p_rec[0],
                'accuracy': acc
            }

    return results[f"{results_key}"]


def main(args):
    # 0) logging setup
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s %(levelname)-5s %(message)s", datefmt="%H:%M:%S",
        level=level
    )

    # Set seed
    logger.debug("Seeding: numpy=%d, torch=%d", args.seed, args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False

    # Initialize Subjectivity class
    logger.info("Loading data from %s …", args.data_folder)
    t0 = time.time()
    detector = Subjectivity(data_folder=args.data_folder, seed=args.seed, device=args.device)
    logger.info("Subjectivity initialized in %.1fs", time.time() - t0)

    for split in ["train","dev","test"]:
        df = detector.all_data[args.language][split]
        counts = df["label"].value_counts(normalize=True).to_dict()
        logger.debug("%5s split: %d examples", split, len(df))
        logger.debug("    class balance: %s", ", ".join(f"{k}={v:.1%}" for k,v in counts.items()))
        if args.verbose and split=="train":
            lengths = df["sentence"].str.len()
            desc = lengths.describe()[["min","mean","max"]].rename({"min":"MinLen","mean":"MeanLen","max":"MaxLen"})
            logger.debug("    lengths: %s", desc.to_dict())

    logger.info("Running with: family=%s, sentiment=%s, lang=%s", args.model_family, args.use_sentiment, args.language)

    model_family = args.model_family
    use_sentiment = args.use_sentiment
    language = args.language

    if model_family == "deberta":
        model_card = "microsoft/mdeberta-v3-base"
        pretrained_card = f"MatteoFasulo/mdeberta-v3-base-subjectivity" + (
            f"-sentiment-{language}" if use_sentiment else f"-{language}"
        )
    else:  # modernbert
        model_card = "answerdotai/ModernBERT-base"
        pretrained_card = f"MatteoFasulo/ModernBERT-base-subjectivity" + (
            f"-sentiment-{language}" if use_sentiment else f"-{language}"
        )

    stats = run(
        detector,
        model_family=model_family,
        model_card=model_card,
        pretrained_card=pretrained_card,
        language=language,
        use_sentiment=use_sentiment,
    )

    table = [[k, fmt(v)] for k, v in stats.items()]
    logger.info("\nFinal test metrics:\n%s", tabulate(table, headers=["Metric","Value"]))

    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Main script for subjectivity classification.")
    parser.add_argument("--data_folder", type=str, default=const.DATA_PATH, help="Path to the dataset folder.")
    parser.add_argument("--seed", type=int, default=const.SEED, help="Random seed for reproducibility.")
    parser.add_argument(
        "--device",
        type=str,
        default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        help="Device to use for computations.",
    )
    parser.add_argument(
        "--model_family", choices=["deberta", "modernbert"], required=True, help="which backbone to use"
    )
    parser.add_argument(
        "--language",
        type=str,
        choices=["arabic", "english", "german", "italian"],
        default="english",
        help="Language of the dataset to use.",
    )
    parser.add_argument("--use_sentiment", action="store_true", help="Whether to use sentiment features in the model.")
    parser.add_argument("--verbose", action="store_true", help="Print label distributions.")

    args = parser.parse_args()
    tqdm.pandas()
    main(args)
