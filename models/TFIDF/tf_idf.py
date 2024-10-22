"""
This python script implement the class used to use the TF-IDF method,
to retrieve documents from a query.

Author: Paulo Ribeiro
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import sparse
from collections import Counter
from pickle import dump, load, HIGHEST_PROTOCOL


class TFIDF:
    def __init__(self):
        self.data = None
        self.idf = None
        self.tf_idf_matrix = None
        self.vocabulary = None
        self.vocab_len = None
        self.token_to_id = None

    @staticmethod
    def _tokenize(doc):
        """Efficient tokenization using regex."""
        return re.findall(r'\b\w+\b', doc)

    def _compute_tf(self):
        """TODO"""
        # Prepare lists to construct a sparse matrix
        doc_ids = []
        token_ids = []
        tf_values = []
        doc_number = len(self.data)
        token_to_id = {token: idx for idx, token in enumerate(self.vocabulary)}

        for doc_id, doc in tqdm(enumerate(self.data['text']), total=doc_number,
                                desc='Computing the TF matrix'):
            tokens = self._tokenize(doc)
            token_freq = Counter(tokens)
            total_tokens = len(tokens)

            # Loop over the word frequencies and fill in the indices and values for the sparse matrix
            for token, freq in token_freq.items():
                if token in token_to_id:
                    token_ids.append(token_to_id[token])
                    doc_ids.append(doc_id)
                    tf_values.append(freq / total_tokens)  # Normalized term frequency

        # Create the sparse TF matrix
        return sparse.csr_matrix((tf_values, (doc_ids, token_ids)),
                                 shape=(doc_number, self.vocab_len))

    def _compute_tf_idf(self, tf_matrix):
        """Compute the TF-IDF vector for a given document's TF vector."""
        return sparse.csr_matrix(tf_matrix.multiply(self.idf))


class TFIDFVectorizer(TFIDF):
    def __init__(self, corpus_path: str):
        super().__init__()
        self.lang = None
        self.data = self._load_data(corpus_path)
        self.dict_len = 0

    def _load_data(self, corpus_path):
        """Load data and look the language of the data."""
        print("Loading corpus...")
        corpus = pd.read_json(corpus_path, lines=True)
        lang = corpus['lang'].unique()
        if len(lang) == 1:
            self.lang = lang[0]
            print(f"Corpus loaded successfully! Found {len(corpus)} documents in '{self.lang}'.")
            return corpus
        else:
            raise Warning(f"The corpus you are giving is multilingual ({len(lang)} different languages)."
                          "Please give only a corpus with one languages.")

    def create_vocabulary(self, plot: bool = False):
        """Manually create a dictionary of top words based on frequency."""
        word_count = Counter()

        # Iterating through each document to update the word count
        for text in self.data['text']:
            word_count.update(self._tokenize(text))

        self.vocabulary = sorted(word_count.keys())
        self.vocab_len = len(self.vocabulary)

        if plot:
            # Plot the most frequent words
            sorted_word_count = dict(sorted(word_count.items(), key=lambda item: item[1], reverse=True))
            words = list(sorted_word_count.keys())
            frequencies = list(sorted_word_count.values())

            # Limit to the top 3000 words
            top_n = 3000
            words = words[:top_n]
            frequencies = frequencies[:top_n]

            plt.figure(figsize=(11, 4))

            # Plot the line for word frequencies
            plt.plot(words, frequencies, color='lightblue', label='Word Frequency')

            # Fill the area under the curve (between the curve and y=0)
            plt.fill_between(words, frequencies, color='lightblue', alpha=0.8)

            # Add labels and title
            plt.xlabel('Words')
            plt.ylabel('Frequency')
            plt.title(f'Top {top_n} Words by Frequency')

            # Re-arrange x-axis labels to make them readable for user
            plt.xticks(rotation=90,
                       ticks=range(0, top_n, int(top_n / 30)),
                       labels=[words[i] for i in range(0, top_n, int(top_n / 30))])

            # Display the plot
            plt.legend()
            plt.show()

        print(f"Dictionary created with {len(self.vocabulary)} unique words/tokens !\n")

    def _compute_idf(self, tf_matrix):
        """Compute the inverse document frequency (IDF) for the corpus."""
        N = len(self.data)
        df = tf_matrix.astype(bool).sum(axis=0)
        self.idf = sparse.csr_matrix(np.log((N + 1) / (df + 1)) + 1)

    def vectorize(self):
        """Fit the TF-IDF model to the corpus."""
        # Computation of the sparse TF matrix
        corpus_tf_matrix = self._compute_tf()

        # Computation of the IDF vector
        print("\nComputing IDF...")
        self._compute_idf(tf_matrix=corpus_tf_matrix)
        print("Done.\n")

        # Multiply the TF vectors by the IDF vector
        print("Computing TF-IDF matrix...")
        self.tf_idf_matrix = self._compute_tf_idf(tf_matrix=corpus_tf_matrix)
        print("Done.\n")

        print(f"The TF-IDF matrix has a shape of {self.tf_idf_matrix.shape}.")

    def save_results(self, path: str):
        """Save the TF-IDF matrix and related data."""
        os.makedirs(f'{path}', exist_ok=True)

        # Create a dictionary to store results
        results = {
            'lang': self.lang,
            'id_to_docid': {idx: docid for idx, docid in enumerate(self.data['docid'])},
            'vocabulary': self.vocabulary,
            'idf': self.idf,
            'tf_idf_matrix': self.tf_idf_matrix
        }

        # Save the results using pickle with compression
        with open(f'{path}/tf_idf_{self.lang}.pkl', 'wb') as f:
            dump(results, f, protocol=HIGHEST_PROTOCOL)
        print("TF-IDF results saved successfully using pickle!")


class TFIDFRetriever(TFIDF):
    def __init__(self, queries_df: pd.DataFrame, tf_idf_data_path: str, lang: str, top_k: int = 10):
        super().__init__()
        # Load the saved TF-IFD matrix and other necessary data
        with open(tf_idf_data_path, 'rb') as f:
            results = load(f)
        self.id_to_docid = results['id_to_docid']
        self.tf_idf_matrix = results['tf_idf_matrix']
        self.idf = results['idf']
        self.vocabulary = results['vocabulary']
        self.lang = results['lang']
        del results

        # Create class variables useful for matching
        self.data = queries_df
        self.query_tf_idf_matrix = None
        self.top_k = top_k
        self.vocab_len = len(self.vocabulary)
        self.matches = None

        # Warning to let the user know that he might be performing some cross language matching.
        if lang != self.lang:
            raise Warning(f"You are trying to use TFIDFRetriever for '{lang}' queries, using '{self.lang}' documents.")

    def vectorize_query(self):
        """Vectorize the queries by computing its TF-IDF vector"""
        # Computation of the sparse TF matrix
        query_tf_matrix = self._compute_tf()

        # Multiply the TF vectors by the IDF vector
        print("Computing TF-IDF matrix...")
        self.query_tf_idf_matrix = self._compute_tf_idf(tf_matrix=query_tf_matrix)
        print("Done.\n")

    @staticmethod
    def normalize_sparse_matrix(matrix):
        """Normalize the sparse matrix rows to have unit L2 norm."""
        # Compute L2 norm for each row
        norms = sparse.linalg.norm(matrix, axis=1)

        # Avoid division by zero by setting zero norms to 1 (they will stay zero after normalization)
        norms[norms == 0] = 1

        # Divide each element in the row by the corresponding row's norm
        normalized_matrix = matrix.multiply(1.0 / norms[:, np.newaxis])

        return normalized_matrix

    def match(self):
        """TODO"""
        # First, we need to normalize both matrices
        normalized_queries = self.normalize_sparse_matrix(self.query_tf_idf_matrix)
        normalized_documents = self.normalize_sparse_matrix(self.tf_idf_matrix)

        # Now, we can simply compute the dot product to retrieve the cosine similarity
        cosine_sim_matrix = normalized_queries.dot(normalized_documents.T)

        # Step 3: For each query, get the indices of the top-k highest similarities (document ids)
        top_docids_to_query = []
        for similarity in cosine_sim_matrix:  # Iterate over each query
            # Get the top-k document indices for this query
            top_k_ids = np.argsort(np.abs(similarity.toarray().flatten()))[-self.top_k:][::-1]
            top_k_docids = [self.id_to_docid[ids] for ids in top_k_ids]
            top_docids_to_query.append(top_k_docids)

        # Add the results to the query DataFrame
        self.matches = pd.Series(top_docids_to_query, name='docids')
