# TF-IDF Matrix

The model pickle files are too large to be included in this repository, so they must be computed locally. First, ensure 
that data preprocessing has been completed by running the `/corpus_preprocessing.py` script. Once preprocessing is done,
run the `/models/TFIDF/tf_idf_matrix.py` script to generate the TF-IDF matrix scores.

This process will take 30 minutes, after which you’ll have all the necessary TF-IDF matrix scores to run the 
`/performance/performance_visualization.ipynb` notebook for performance analysis.
