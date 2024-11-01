# BM25s Matrix

The model pickle files are too large to be included in this repository, so they must be computed locally. First, ensure 
that data preprocessing has been completed by running the `/corpus_preprocessing.py` script. Once preprocessing is done,
run the `/models/BM25s/bm25s_matrix_score.py` script to generate the BM25 matrix scores.

This process will take a few hours, after which you’ll have all the necessary BM25 matrix scores to run the 
`/performance/performance_visualization.ipynb` notebook for performance analysis.
