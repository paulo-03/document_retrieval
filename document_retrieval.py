"""
Python script to use for document retrieval using TF-IDF or BM25s model.

Author: Paulo Ribeiro
"""

from data_helpers import QueryClean
from models.TFIDF.tf_idf import TFIDFRetriever
from models.BM25s.bm25s import BM25sRetriever
import warnings
warnings.filterwarnings('ignore')


def main(processing_wanted: str):
    """Pipeline to retrieve document choosing specific model"""
    # Load the queries
    query = QueryClean(
        queries_path='data/dev.csv',
        processing_wanted=processing_wanted,
        show_progress=False
    )

    # Variable needed to choose the right model from the pre-process chosen before
    model_type = '_'.join(processing_wanted)

    # Perform the pre-processing step chosen
    langs = query.pre_process()


if __name__ == '__main__':
    # Choose the pre-process to perform over the queries ['lc', 'lc_sw', 'lc_sw_l']
    main(processing_wanted = 'lc',
         model = 'bm25s')