"""
This python script implement the class used to use the TF-IDF method,
to retrieve documents from a query.

Author: Paulo Ribeiro
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm
from scipy import sparse
from data_loader import clean_sentence


class TFIDFTrainer:
    def __init__(self, corpus_path: str):
        self.corpus_path = corpus_path
        self.lang = None
        self.corpus = self._load_data()
        self.dictionary = None
        self.idf = None
        self.tf_idf = {}
        self.dict_len = 0

    def _load_data(self):
        """Load the cleaned data, meaning a transformed documents to have a constant pattern across words,
        meaning no uppercase letter, etc."""
        # Load the data
        print("Loading corpus...")
        corpus = pd.read_json(self.corpus_path, lines=True)
        self.lang = corpus['lang'].unique()[0]
        print("Corpus loaded successfully !\n")

        # Give some basic information about the data
        print("Information about the given corpus\n"
              "###################################\n"
              f"Number of documents: {len(corpus)}\n"
              f"Language (only one language should be displayed): {corpus['lang'].unique()}\n")

        return corpus

    def create_dictionary(self, top_w: int, plot: bool = True):
        """Create the vocabulary containing all words found across all the corpus."""
        # Initiate the word_count dictionary
        word_count = dict()

        # Iterate over the text column only
        for doc in tqdm(self.corpus['text'], desc="Creating dictionary"):
            for word in doc.split():
                if word in word_count:
                    word_count[word] += 1  # Increase count if word already exists
                else:
                    word_count[word] = 1  # Create a new key with count 1 if word doesn't exist

        # Sort the values to know which one to exclude for computational reduction costs
        sorted_word_count = dict(sorted(word_count.items(), key=lambda item: item[1], reverse=True))

        # Extract keys (words) and values (frequencies) for plotting
        words = list(sorted_word_count.keys())
        frequencies = list(sorted_word_count.values())

        # Dictionary will be set to be the top_w words with the highest frequency across documents
        self.dictionary = words[:top_w]
        self.dict_len = len(self.dictionary)  # equals to top_w if no error

        if plot:
            # Plot the bar chart of every word's frequency
            plt.figure(figsize=(10, 5))
            plt.bar(words[:top_w * 3], frequencies[:top_w * 3], color='blue')

            # Add labels and title and the threshold top_w
            plt.xlabel('Words')
            plt.ylabel('Frequency')
            plt.title('Frequency of words across corpus')
            plt.axvline(x=top_w, color='red', linestyle=':', label=f'Top {top_w} Words')

            # Re-arrange x-axis labels to make them readable for user
            plt.xticks(rotation=90,
                       ticks=range(0, top_w * 3, int(top_w * 3 / 10)),
                       labels=[words[i] for i in range(0, top_w * 3, int(top_w * 3 / 10))])

            # Show the legend and display plot
            plt.legend()
            plt.show()

        print(f"Dictionary created with {self.dict_len} unique words selected !\n"
              f"(Please notice that {len(words)} unique words where found over all the documents, but for \n"
              f"computational reasons, only the top {top_w} words where selected into the dictionary.)")

    def _tf(self, text: list[str]):
        """Compute the Term Frequency (TF) of a text."""
        # Initiate the counter and compute the length of document
        counter = dict.fromkeys(self.dictionary, 0)
        N = len(text)

        # Count the words
        for word in text:
            if word in self.dictionary:
                counter[word] += 1

        return np.array([freq / N for _, freq in counter.items()])

    def _idf(self):
        """Compute the Inverse Document Frequency (IDF) of a corpus."""
        # Concatenating arrays in the 'TF' column side by side
        all_tf = np.column_stack(self.corpus['tf'].values)

        # Compute the number of document in the corpus
        N = all_tf.shape[1]

        # Make the array a boolean object and sum for each term the term frequency across the documents
        all_tf = np.sum(all_tf > 0, axis=1)

        # Compute the IDF
        return np.log(N / (1 + all_tf)) + 1  # IDF smooth version selected

    def _tf_idf(self):
        """TODO"""
        # Take every TF for each document and multiply it by the IDF vector and store in a list
        for index, doc in tqdm(self.corpus.iterrows(),
                               total=len(self.corpus),
                               desc="Computation of the TF-IDF vectors for each document"):
            self.tf_idf[doc['docid']] = sparse.csr_matrix(doc['tf'] * self.idf)

    def fit(self):
        """TODO"""
        # Compute the Term Frequency (TF) for each document and store the results
        tqdm.pandas(desc="Computation of Term Frequencies (TF)")
        self.corpus['tf'] = self.corpus.progress_apply(
            lambda doc: self._tf(doc['text'].split()), axis=1
        )

        print(f"\n{'#' * 50}\n")

        # Compute the Inverse Document Frequency (IDF)
        print("Computation of Inverse Document Frequency...")
        self.idf = self._idf()
        print("Finished !\n")

        print(f"{'#' * 50}\n")

        # Create the TF-IDF matrix
        self._tf_idf()

    def save_results(self):
        """TODO"""
        # Create a folder named "tf_idf_matrix"
        os.makedirs("tf_idf_matrix", exist_ok=True)

        # Add the idf vector to the results to store
        results = self.tf_idf
        results['idf'] = self.idf
        results['dictionary'] = self.dictionary

        # Save the sparse dictionary using joblib
        joblib.dump(results, f'tf_idf_matrix/tf_idf_{self.lang}.pkl', compress=True)


class TFIDFRetriever:
    def __init__(self, tf_idf_matrix_path: str):
        self.tf_idf_matrix_path = tf_idf_matrix_path
        self.tf_idf = joblib.load(self.tf_idf_matrix_path)
        self.idf = self.tf_idf['idf']
        self.dictionary = self.tf_idf['dictionary']
        for key in ['idf', 'dictionary']:
            del self.tf_idf[key]

    def _tf(self, text: list[str]):
        """Compute the Term Frequency (TF) of a text."""
        # Initiate the counter and compute the length of document
        counter = dict.fromkeys(self.dictionary, 0)
        N = len(text)

        # Count the words
        for word in text:
            if word in self.dictionary:
                counter[word] += 1

        return np.array([freq / N for _, freq in counter.items()])

    @staticmethod
    def _cos_similarity(query_vect, doc_vect):
        # Compute the dot product
        dot_product = np.dot(query_vect, doc_vect)

        # Compute the magnitudes of the vectors
        magnitude_a = np.linalg.norm(query_vect)
        magnitude_b = np.linalg.norm(doc_vect)

        # Handle the case of zero magnitude to avoid division by zero
        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0  # or raise an exception if preferred

        # Calculate cosine similarity
        cosine_sim = dot_product / (magnitude_a * magnitude_b)

        return cosine_sim

    def search(self, query: str, lang: str, topk: int = 5, present_results: bool = False):
        """TODO"""
        clean_query = clean_sentence(query, lang)
        query_tf = self._tf(clean_query.split())
        query_tf_idf = query_tf * self.idf

        # Compute the cosine similarity between the query and each vector
        similarity = {}
        for docid, doc_tf_idf in self.tf_idf.items():
            similarity[docid] = self._cos_similarity(query_tf_idf, doc_tf_idf.toarray().reshape(len(self.dictionary), ))

        # Return the top k documents with higher similarity
        relevant_docs = dict(sorted(similarity.items(), key=lambda item: item[1], reverse=True)[:topk])

        if present_results:
            print(f"The top {topk} documents are:")
            for docid, sim in relevant_docs.items():
                print(f"{docid}: {sim}")
