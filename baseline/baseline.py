import argparse
import csv
import logging
import os
import random
from typing import AnyStr

import numpy as np
import pandas as pd
import tensorflow as tf
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

random.seed(0)

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


def to_labels(
        data,
        threshold=0.5
):
    ypred = []
    for pred in data:
        if pred >= threshold:
            ypred.append('SUBJ')
        else:
            ypred.append('OBJ')
    return ypred


def prepare_data(
        X,
        tokenizer,
        max_length,
        y=[]
):
    pad = tf.keras.preprocessing.sequence.pad_sequences  # (seq, padding = 'post', maxlen = maxlen)
    tokenizer = tokenizer
    data_fields = {
        "input_ids": [],
        "token_type_ids": [],
        "attention_mask": [],
        "Label": []
    }
    labels = {
        'SUBJ': 1.0,
        'OBJ': 0.0
    }
    for i in range(len(X)):
        data = tokenizer(X[i])
        padded = pad([data['input_ids'], data['attention_mask'], data['token_type_ids']], padding='post',
                     maxlen=max_length)
        data_fields['input_ids'].append(padded[0])
        data_fields['attention_mask'].append(padded[1])
        data_fields['token_type_ids'].append(padded[-1])
    if len(y):
        data_fields['label'] = list(map(lambda e: labels[e], y))
    else:
        data_fields['label'] = None

    for key in data_fields:
        data_fields[key] = np.array(data_fields[key])

    return [data_fields["input_ids"],
            data_fields["token_type_ids"],
            data_fields["attention_mask"]], data_fields["label"]


def run_sbert_lr_baseline(
        data_dir: AnyStr,
        train_filepath: AnyStr,
        test_filepath: AnyStr
) -> AnyStr:
    train_data = pd.read_csv(train_filepath, sep='\t', quoting=csv.QUOTE_NONE)
    test_data = pd.read_csv(test_filepath, sep='\t', quoting=csv.QUOTE_NONE)

    vect = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    model = LogisticRegression(class_weight="balanced")
    model.fit(X=vect.encode(train_data['sentence'].values), y=train_data['label'].values)

    predictions = model.predict(X=vect.encode(test_data['sentence'].values)).tolist()
    pred_df = pd.DataFrame()
    pred_df['sentence_id'] = test_data['sentence_id']
    pred_df['label'] = predictions

    predictions_filepath = os.path.join(data_dir, 'base_pred_lan.tsv')
    pred_df.to_csv(predictions_filepath, index=False, sep='\t')

    return predictions_filepath


def float_formatter(f) -> str:
    """
    Format a float as a pretty string.
    """
    if f != f or f is None:
        # instead of returning nan, return "" so it shows blank in table
        return ""
    if isinstance(f, int):
        # don't do any rounding of integers, leave them alone
        return str(f)
    if f >= 1000:
        # numbers > 1000 just round to the nearest integer
        s = f'{f:.0f}'
    else:
        # otherwise show 4 significant figures, regardless of decimal spot
        s = f'{f:.4g}'
    # replace leading 0's with blanks for easier reading
    # example:  -0.32 to -.32
    s = s.replace('-0.', '-.')
    if s.startswith('0.'):
        s = s[1:]
    # Add the trailing 0's to always show 4 digits
    # example: .32 to .3200
    if s[0] == '.' and len(s) < 5:
        s += '0' * (5 - len(s))
    return s


def general_formatter(value):
    if isinstance(value, list) or isinstance(value, np.ndarray):
        # apply formatting to each element
        return '[{}]'.format(','.join([float_formatter(item) for item in value]))
    if isinstance(value, dict):
        return '[{}]'.format(','.join(list(value.items())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainpath', '-trp',
                        required=True,
                        type=str, )
    parser.add_argument('--testpath', '-ttp',
                        required=True,
                        type=str, )
    args = parser.parse_args()

    train_filepath = os.path.normpath(args.trainpath)
    assert os.path.isfile(train_filepath), f'Could not find train file. Got {train_filepath}'

    test_filepath = os.path.normpath(args.testpath)
    assert os.path.isfile(test_filepath), f'Could not find test file. Got {test_filepath}'

    data_dir = os.path.dirname(test_filepath)

    logging.info(f"""Running baseline with following configuration: 
                 Train: {train_filepath} 
                 Test: {test_filepath}""")
    run_sbert_lr_baseline(data_dir=data_dir,
                          test_filepath=test_filepath,
                          train_filepath=train_filepath)
