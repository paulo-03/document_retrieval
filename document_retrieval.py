"""
Python script to use for document retrieval using TF-IDF or BM25s model.

Author: Paulo Ribeiro
"""

import pandas as pd
from data_helpers import QueryClean
from models.TFIDF.tf_idf import TFIDFRetriever
from models.BM25s.bm25s import BM25sRetriever
import warnings

warnings.filterwarnings('ignore')


def main(model: str, processing_wanted: str, data_type: str, k1: float):
    """Pipeline to retrieve document choosing specific model"""
    # Load the queries
    query = QueryClean(
        queries_path=f'data/{data_type}.csv',
        processing_wanted=processing_wanted,
        show_progress=False
    )

    # Initiate the list to stack all the matches per language in one .csv file
    match_per_lang = []

    # Perform the pre-processing step chosen
    langs = query.pre_process()

    if model == 'BM25s':
        # Initiate all the BM25s models for each language present in the queries
        bm25s_retrievers = {
            lang: BM25sRetriever(queries_df=query.data_clean[lang],
                                 model_path=f'models/BM25s/bm25s_matrix/{processing_wanted}/k1_{k1}/bm25s_{lang}.pkl',
                                 top_k=10)
            for lang in langs
        }

        # Compute the matching between query and document for each language separately
        for lang in langs:
            bm25s = bm25s_retrievers[lang]
            bm25s.match()
            match_per_lang.append(bm25s.matches)

        # Stack all the pd.Series to create a unified pd.Series with all the matches
        matches = pd.concat(match_per_lang, ignore_index=True)

        # Write on disk a .csv file with the matches
        matches.to_csv(f'bm25s_{processing_wanted}_k1_{k1}_{data_type}_output.csv',
                       index=True,
                       index_label='id')

    elif model == 'TFIDF':
        # Initiate all the TFIDF models for each language present in the queries
        tf_idf_retrievers = {
            lang: TFIDFRetriever(queries_df=query.data_clean[lang],
                                 tf_idf_data_path=f'models/TFIDF/tf_idf_matrix/{processing_wanted}/tf_idf_{lang}.pkl',
                                 lang=f'{lang}',
                                 top_k=10)
            for lang in langs
        }

        for lang in langs:
            tf_idf = tf_idf_retrievers[lang]
            tf_idf.vectorize_query()
            tf_idf.match()
            match_per_lang.append(tf_idf.matches)

        # Stack all the pd.Series to create a unified pd.Series with all the matches
        matches = pd.concat(match_per_lang, ignore_index=True)

        # Write on disk a .csv file with the matches
        matches.to_csv(f'tf_idf_{processing_wanted}_{data_type}_output.csv',
                       index=True,
                       index_label='id')


if __name__ == '__main__':
    processes = ['lc', 'lc_sw', 'lc_sw_l']
    k1s = [1.0]  #[1.2, 1.6, 2.0]

    for process in processes:
        for k1 in k1s:
            main(
                processing_wanted=process,  # 'lc', 'lc_sw' or 'lc_sw_l'
                model='BM25s',  # 'BM25s' or 'TFIDF'
                data_type='dev',  # 'dev' or 'test'
                k1=k1  # Only useful for BM25s model
            )
