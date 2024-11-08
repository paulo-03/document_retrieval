"""
This python script pre-compute the clean corpus matrix score per language.
"""

import os
from bm25s import BM25sTrainer


def main(root_corpus_path: str, k1: float, b: float):
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
        bm25s = BM25sTrainer(
            corpus_path=corpus_path,
            k1=k1,
            b=b
        )

        # Create the vocabulary and pre-compute the docs length and average doc length across corpus.
        bm25s.create_vocabulary()

        # Compute the full matrix score for every (term, doc) pair.
        bm25s.fit()

        # Save the model in the bm25s_matrix_score
        bm25s.save_results(path=f"bm25s_matrix/{pre_process}")


if __name__ == '__main__':

    k1s = [1.0]  # [1.2, 1.6, 2.0]
    processes = ['lc', 'lc_sw', 'lc_sw_l']

    for process in processes:
        for k1 in k1s:
            main(root_corpus_path=f"../../clean_data/{process}",
                 k1=k1,  # values going from [1.2, 2]
                 b=0.75)
