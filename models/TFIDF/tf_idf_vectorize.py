"""
Python script useful to create sparse vectors representation of our corpus documents using TF-IDF.

Author: Paulo Ribeiro
"""

import os
from tf_idf import TFIDFVectorizer


def main(root_corpus_path: str):
    """main function to train all BM25s model from a specific cleaned corpus and hyperparameters choice"""
    # Find all the clean dataset in the root corpus path and the pre-processing steps chosen
    corpus_paths = ["/".join([root_corpus_path, corpus])
                    for corpus in os.listdir(root_corpus_path)
                    if corpus.endswith('.json')]

    # Isolate the pre-processing steps chosen, to store the model in the right place
    pre_process = root_corpus_path.split('/')[-1]

    # Start "training" model for the specific pre-process corpus of each language
    for corpus_path in corpus_paths:

        # Initiate the model
        tf_idf = TFIDFVectorizer(
            corpus_path=corpus_path
        )

        # Create the vocabulary and pre-compute the docs length and average doc length across corpus.
        tf_idf.create_vocabulary()

        # Compute the sparse vectorization representation of the corpus.
        tf_idf.vectorize()

        # Save the model in the bm25s_matrix_score
        tf_idf.save_results(path=f"tf_idf_matrix/{pre_process}")


if __name__ == '__main__':

    processes = ['lc', 'lc_sw', 'lc_sw_l']

    for process in processes:
        main(root_corpus_path=f"../../clean_data/{process}")
