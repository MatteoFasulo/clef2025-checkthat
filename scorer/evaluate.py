import csv

import argparse
import pandas
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def validate_files(pred_file, gold_file):
    pred_data = pandas.read_csv(pred_file, sep='\t', quoting=csv.QUOTE_NONE)
    gold_data = pandas.read_csv(gold_file, sep='\t', quoting=csv.QUOTE_NONE)

    if len(pred_data) != len(gold_data):
        print("ERROR! Different number of instances in the files")
        return False

    if not ('label' in pred_data and
            'sentence_id' in pred_data and 'sentence_id' in gold_data):
        print("ERROR! Wrong columns")
        return False

    pred_values = pred_data['label'].unique()

    if not ((len(pred_values) == 2 and "SUBJ" in pred_values and
             "OBJ" in pred_values) or
            (len(pred_values) == 1 and
             ("OBJ" in pred_values or "SUBJ" in pred_values))):
        print("ERROR! Wrong labels")
        return False

    pred_data.rename(columns={'label': 'pred_label'}, inplace=True)

    whole_data = pandas.merge(pred_data, gold_data, on="sentence_id")

    if len(pred_data) != len(whole_data):
        print("ERROR! Different ids in the two files")
        return False

    print("The file is properly formatted")

    if not ('label' in gold_data):
        print("WARNING: no labels in the gold data file")
        print("Impossible to proceed with evaluation")
        return False

    whole_data.rename(columns={'label': 'gold_label'}, inplace=True)

    return whole_data


def evaluate(whole_data):
    pred_values = whole_data['pred_label'].values
    gold_values = whole_data['gold_label'].values

    acc = accuracy_score(gold_values, pred_values)
    m_prec, m_rec, m_f1, m_s = precision_recall_fscore_support(gold_values, pred_values, average="macro",
                                                               zero_division=0)
    p_prec, p_rec, p_f1, p_s = precision_recall_fscore_support(gold_values, pred_values, labels=["SUBJ"],
                                                               zero_division=0)

    return {
        'macro_F1': m_f1,
        'macro_P': m_prec,
        'macro_R': m_rec,
        'SUBJ_F1': p_f1[0],
        'SUBJ_P': p_prec[0],
        'SUBJ_R': p_rec[0],
        'accuracy': acc
    }


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-file-path", "-g", required=True, type=str,
                        help="Path to file with gold annotations.")
    parser.add_argument("--pred-file-path", "-p", required=True, type=str,
                        help="Path to file with predict class per sentence")

    args = parser.parse_args()

    pred_file = args.pred_file_path
    gold_file = args.gold_file_path

    whole_data = validate_files(pred_file, gold_file)

    if whole_data is not False:
        print("Started evaluating results for task-2...")

        scores = evaluate(whole_data)
        # print(scores)

        print("macro_F1:\t{:.2f}\t\tmacro_P:\t\t{:.2f}\t\tmacro_R:\t\t{:.2f}\t\t".format(scores['macro_F1'],
                                                                                     scores['macro_P'],
                                                                                     scores['macro_R']) +
              "SUBJ_F1:\t{:.2f}\t\tSUBJ_P:\t\t{:.2f}\t\tSUBJ_R:\t\t{:.2f}\t\t".format(scores['SUBJ_F1'], scores['SUBJ_P'],
                                                                                  scores['SUBJ_R']) +
              "accuracy:\t{:.2f}".format(scores['accuracy']))
