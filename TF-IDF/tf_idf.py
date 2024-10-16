"""
This python script implement the class used to use the TF-IDF method,
to retrieve documents from a query.

Author: Paulo Ribeiro
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib
from scipy import sparse
from multiprocessing import Pool, cpu_count
from data_helpers import QueryClean

class TFIDFTrainer:
    def __init__(self, corpus_path: str):
        self.corpus_path = corpus_path
        self.lang = None
        self.corpus = self._load_data()
        self.dictionary = None
        self.idf = None
        self.tf_idf_matrix = None
        self.dict_len = 0

    def _load_data(self):
        """Load data and basic information about the corpus."""
        print("Loading corpus...")
        corpus = pd.read_json(self.corpus_path, lines=True)
        self.lang = corpus['lang'].unique()[0]
        print(f"Corpus loaded successfully! Found {len(corpus)} documents.")
        return corpus

    def create_dictionary(self, top_w: int, plot: bool = True):
        """Manually create a dictionary of top words based on frequency."""
        word_count = {}

        for doc in tqdm(self.corpus['text'], desc="Creating dictionary"):
            for word in doc.split():
                if word in word_count:
                    word_count[word] += 1
                else:
                    word_count[word] = 1

        sorted_word_count = dict(sorted(word_count.items(), key=lambda item: item[1], reverse=True))

        # Get the top `top_w` words for the dictionary
        self.dictionary = list(sorted_word_count.keys())[:top_w]
        self.dict_len = len(self.dictionary)

        if plot:
            # Plot the top words
            words = list(sorted_word_count.keys())
            frequencies = list(sorted_word_count.values())
            plt.figure(figsize=(10, 5))
            plt.bar(words[:top_w * 3], frequencies[:top_w * 3], color='blue')
            plt.axvline(x=top_w, color='red', linestyle=':', label=f'Top {top_w} Words')
            plt.xlabel('Words')
            plt.ylabel('Frequency')
            plt.title(f'Top {top_w} Words by Frequency')
            # Re-arrange x-axis labels to make them readable for user
            plt.xticks(rotation=90,
                       ticks=range(0, top_w * 3, int(top_w * 3 / 10)),
                       labels=[words[i] for i in range(0, top_w * 3, int(top_w * 3 / 5))])

            plt.show()

            print(f"Dictionary created with {self.dict_len} unique words selected !\n"
                  f"(Please notice that {len(words)} unique words where found over all the documents, but for \n"
                  f"computational reasons, only the top {top_w} words where selected into the dictionary.)")

    def _compute_tf(self, doc):
        """Compute the term frequency (TF) for a given document."""
        counter = dict.fromkeys(self.dictionary, 0)
        words = doc.split()
        N = len(words)
        for word in words:
            if word in self.dictionary:
                counter[word] += 1
        return np.array([freq / N for freq in counter.values()])

    def _compute_idf(self):
        """Compute the inverse document frequency (IDF) for the corpus."""
        df = np.zeros(self.dict_len)
        N = len(self.corpus)

        for doc in self.corpus['tf']:
            df += (doc > 0).astype(int)

        # Compute IDF with smoothing
        return np.log((N + 1) / (df + 1)) + 1

    def _compute_tf_idf_for_doc(self, doc_tf):
        """Compute the TF-IDF vector for a given document's TF vector."""
        return sparse.csr_matrix(doc_tf * self.idf)

    def fit(self):
        """Fit the TF-IDF model to the corpus."""
        # Parallel computation of TF and TF-IDF for each document
        print("Computing TF for each document using multiprocessing...")
        with Pool(cpu_count() // 2) as pool:
            self.corpus['tf'] = pool.map(self._compute_tf, self.corpus['text'])
        print("Finished computing TF.\n")

        print("Computing IDF across the corpus...")
        self.idf = self._compute_idf()
        print("Finished computing IDF.\n")

        print("Computing TF-IDF for each document using multiprocessing...")
        with Pool(cpu_count() // 2) as pool:
            self.tf_idf_matrix = pool.map(self._compute_tf_idf_for_doc, self.corpus['tf'])
        print("Finished computing TF-IDF.")

    def save_results(self):
        """Save the TF-IDF matrix and related data."""
        os.makedirs("tf_idf_matrix", exist_ok=True)

        # Create a dictionary to store results
        results = {
            'tf_idf_matrix': sparse.vstack(self.tf_idf_matrix),
            'idf': self.idf,
            'dictionary': self.dictionary
        }

        # Save the results using joblib with compression
        joblib.dump(results, f'tf_idf_matrix/tf_idf_{self.lang}.pkl', compress=True)
        print("TF-IDF results saved successfully!")


class TFIDFRetriever:
    def __init__(self, tf_idf_matrix_path: str):
        # Load the saved matrix and other data
        results = joblib.load(tf_idf_matrix_path)
        self.tf_idf_matrix = results['tf_idf_matrix']
        self.idf = results['idf']
        self.dictionary = results['dictionary']

    def _compute_tf(self, text: list[str]):
        """Compute term frequency (TF) for a query."""
        counter = dict.fromkeys(self.dictionary, 0)
        N = len(text)
        for word in text:
            if word in self.dictionary:
                counter[word] += 1
        return np.array([freq / N for freq in counter.values()])

    @staticmethod
    def _cosine_similarity(query_vect, doc_vect):
        """Compute cosine similarity between two vectors."""
        dot_product = query_vect.dot(doc_vect.T)
        magnitude_query = np.linalg.norm(query_vect)
        magnitude_doc = np.linalg.norm(doc_vect)

        if magnitude_query == 0 or magnitude_doc == 0:
            return 0.0
        return dot_product / (magnitude_query * magnitude_doc)

    def search(self, query: str, lang: str, topk: int = 5):
        """Search for top-k documents based on the query."""
        # Still TODO
        clean_query = clean_sentence(query, lang)
        query_tf = self._compute_tf(clean_query.split())
        query_tf_idf = query_tf * self.idf

        # Compute cosine similarity for each document
        similarities = []
        for i in range(self.tf_idf_matrix.shape[0]):
            doc_vect = self.tf_idf_matrix[i].toarray().flatten()
            similarities.append(self._cosine_similarity(query_tf_idf, doc_vect))

        # Get top-k most similar documents
        top_indices = np.argsort(similarities)[-topk:][::-1]
        return top_indices, np.array(similarities)[top_indices]
