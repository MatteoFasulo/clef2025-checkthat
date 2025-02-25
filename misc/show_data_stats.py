import pandas as pd
from pathlib import Path
import logging
import csv

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

if __name__ == '__main__':
    data_path = Path(__file__).resolve().parent.parent.joinpath('data')

    for language_folder in data_path.glob('*'):
        logging.info(f'Language: {language_folder.name}')
        for tsv_file in language_folder.glob('*.tsv'):
            df = pd.read_csv(tsv_file, sep='\t', quoting=csv.QUOTE_NONE)
            logging.info(f'Filename: {tsv_file.name}')
            logging.info(f'Size: {df.shape[0]}')
            logging.info(f'Stats: {df.label.value_counts()}')
        logging.info('*' * 20)