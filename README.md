# Document Retrieval

This project implements a document retrieval system designed to rank documents based on their relevance to a specific
query. The goal is to search through a collection of documents and return the top-ranked documents most relevant to each
query in the given test set. This project demonstrates the end-to-end process of building an effective retrieval system
from scratch.

***Authors:*** Rami Atassi, Othmane Idrissi Oudghiri & Paulo Ribeiro

## EPFL Project Description

The main objective of the project is to rank and retrieve the top 10 documents based on relevance to a given query from
a multilingual corpus composed of more than 200,000 documents. The retrieval system is evaluated on **Recall@10** and
**Computation Time**. We aim to achieve the highest recall on 2,000 queries across 7 languages while staying within the
10-minute computation time limit.

### Project Requirements

- **Retrieval Methods**: You may use either supervised or unsupervised retrieval techniques. For supervised methods, you
  must train models on the provided labeled data (not pre-trained or publicly available models).
- **Embeddings**: You are permitted to use pre-trained language models for embedding documents and queries.
- **Restrictions**: Use only custom implementations of retrieval models (i.e., do not use implementations from external
  libraries like `sklearn`).

## Project Structure

The repository is structured and evolves as follows:

1. First in [*corpus_analysis.ipynb*](./corpus_analysis.ipynb), we perform a quick data analysis on the raw corpus.
2. Then in [*corpus_preprocessing.py*](./corpus_preprocessing.py), we preprocess the corpus in 3 different ways (
   lowercase, lowercase + stopwords
   removal, lowercase + stopwords removal + lemmatization) and save the 3 cleaned corpus in [clean_data](./clean_data/).
   We verify and analyze the new versions of the corpus in [*corpus_analysis.ipynb*](./corpus_analysis.ipynb).
3. Using our models implementation, either [*bm25s_matrix_score.py*](./models/BM25s/bm25s_matrix_score.py) or [
   *tf_idf_vectorize.py*](./models/TFIDF/tf_idf_vectorize.py), we *"encode"* the preprocessed versions of the corpus
   into matrix and save them in [models/BM25s/bm25s_matrix](./models/BM25s/bm25s_matrix/)
   or [models/TFIDF/tf_idf_matrix](./models/TFIDF/tf_idf_matrix/) accordingly to the used model.
4. In [*document_retrieval.py*](./document_retrieval.py), we perform the query-document matching with the chosen set of
   corpus preprocessing, model, query source and hyperparameters.
5. Finally in [*performance/performance_visualization.ipynb*](./performance/performance_visualization.ipynb), we compare
   the performance of the different models, corpus preprocessing and hyperparameters using the Recall@10 score.
6. We propose our best version in the [*submission.ipynb*](./submission.ipynb) notebook for kaggle submission.

> **Note**: The `old_versions/` folder store the first BM25 implementation that was accurate but not efficient enough to 
> respect the time constraint of 10 minutes. We decided to let the implementation to show the alternative implementation
> of BM25 without sparse matrix.


# Document Retrieval

This project implements a document retrieval system designed to rank documents based on their relevance to specific queries. The goal is to search a document collection and return the top-ranked documents most relevant to each query in a given test set. This project demonstrates the end-to-end process of building an effective retrieval system from scratch.

***Authors:*** Rami Atassi, Othmane Idrissi Oudghiri & Paulo Ribeiro

## EPFL Project Description

The main objective of this project is to rank and retrieve the top 10 most relevant documents for a given query from a multilingual corpus containing over 200,000 documents. The retrieval system is evaluated based on **Recall@10** and **Computation Time**. The goal is to achieve the highest recall on 2,000 queries across 7 languages while meeting a 10-minute computation time limit.

### Project Requirements

- **Retrieval Methods**: Both supervised and unsupervised retrieval techniques are allowed. For supervised methods, models must be trained on the provided labeled data (pre-trained or publicly available models cannot be used).
- **Embeddings**: Pre-trained language models may be used to embed documents and queries.
- **Restrictions**: Only custom implementations of retrieval models are permitted (external libraries like `sklearn` cannot be used for the retrieval models).

## Project Structure

The repository is organized as follows:

1. First, we perform an initial data analysis on the raw corpus in [*corpus_analysis.ipynb*](./corpus_analysis.ipynb).
2. Next, we preprocess the corpus in [*corpus_preprocessing.py*](./corpus_preprocessing.py) in three ways:
   - Lowercase
   - Lowercase + stopwords removal
   - Lowercase + stopwords removal + lemmatization 
   
   These processed versions are saved in the [clean_data](./clean_data/) directory and verified in [*corpus_analysis.ipynb*](./corpus_analysis.ipynb).

3. Using our models, either [*bm25s_matrix_score.py*](./models/BM25s/bm25s_matrix_score.py) or [*tf_idf_vectorize.py*](./models/TFIDF/tf_idf_vectorize.py), we *encode* each preprocessed corpus version into matrices. These matrices are saved in [models/BM25s/bm25s_matrix](./models/BM25s/bm25s_matrix/) or [models/TFIDF/tf_idf_matrix](./models/TFIDF/tf_idf_matrix/), depending on the model used.
4. In [*document_retrieval.py*](./document_retrieval.py), we perform query-document matching with the selected corpus preprocessing, model, query source, and hyperparameters.
5. In [*performance/performance_visualization.ipynb*](./performance/performance_visualization.ipynb), we evaluate the performance of different models, preprocessing methods, and hyperparameters using the Recall@10 score.
6. Our final, optimized version for submission is documented in [*submission.ipynb*](./submission.ipynb) for Kaggle submission.

> **Note**: The `old_versions/` folder contains an earlier BM25 implementation, which was accurate but could not meet the 10-minute time constraint. This version is retained to showcase an alternative BM25 implementation without sparse matrices.
