# Document Retrieval

This project implements a document retrieval system designed to rank documents based on their relevance to a specific
query. The goal is to search through a collection of documents and return the top-ranked documents most relevant to each
query in the given test set. This project demonstrates the end-to-end process of building an effective retrieval system
from scratch.

***Authors:*** Rami Atassi, Othmane Idrissi Oudghiri & Paulo Ribeiro

## Project Overview

The main objective is to rank and retrieve the top 10 documents based on relevance to a given query from a multilingual collection. Our
retrieval system is evaluated on **Recall@10** and **Computation Time**. We aim
to achieve the highest recall on 2,000 queries across 7 languages while staying within the 10-minute computation time limit.

### Project Requirements

- **Retrieval Methods**: You may use either supervised or unsupervised retrieval techniques. For supervised methods, you
  must train models on the provided labeled data (not pre-trained or publicly available models).
- **Embeddings**: You are permitted to use pre-trained language models for embedding documents and queries.
- **Restrictions**: Use only custom implementations of retrieval models (i.e., do not use implementations from external
  libraries like `sklearn`).

## Project Structure

The repository is structured and evolves as follows:

1. First in [*corpus_analysis.ipynb*](./corpus_analysis.ipynb), we perform a quick data analysis on the raw corpus.
2. Then in [*corpus_preprocessing.py*](./corpus_preprocessing.py), we preprocess the corpus in 3 different ways (lowercase, lowercase + stopwords
   removal, lowercase + stopwords removal + lemmatization) and save the 3 cleaned corpus in [clean_data](./clean_data/).
   We verify and analyze the new versions of the corpus in [*corpus_analysis.ipynb*](./corpus_analysis.ipynb).
3. Using our models implementation (either [*bm25s_matrix_score.py*](./models/BM25s/bm25s_matrix_score.py) or [
   *tf_idf_vectorize.py*](./models/TFIDF/tf_idf_vectorize.py)), we *~encode~* the preprocessed versions of the corpus
   into vectors and save the resulting matrices (in [models/BM25s/bm25s_matrix](./models/BM25s/bm25s_matrix/)
   or [models/TFIDF/tf_idf_matrix](./models/TFIDF/tf_idf_matrix/)).
4. In [*document_retrieval.py*](./document_retrieval.py), we perform the query-document matching with the chosen set of corpus preprocessing, model, query source and hyperparameters.
5. Finally in [*performance/performance_visualization.ipynb*](./performance/performance_visualization.ipynb), we compare the performance of the different models, corpus preprocessing and hyperparameters using the Recall@10 score.
6. [We implement our best version in the [*submission.ipynb*](./submission.ipynb) notebook for submission.]

