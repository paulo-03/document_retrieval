"""
This python script implement the class used to use the TF-IDF method,
to retrieve documents from a query.

Author: Paulo Ribeiro
"""

import pandas as pd
from tqdm import tqdm


class TFIDF:  # Smart move to create a parent class ? Not sure, I will see
    pass


class TFIDFTrainer:
    def __init__(self, corpus_path: str):
        self.corpus_path = corpus_path
        self.corpus = self._load_data()
        self.dictionary = self._create_dictionary()
        self.tf_idf = None

    def _load_data(self):
        """Load the cleaned data and transform it to have a constant pattern across words,
        meaning no uppercase letter, etc."""
        # Load the data
        print("Loading corpus...")
        corpus = pd.read_json(self.corpus_path, lines=True)
        print("Corpus loaded successfully !\n")

        # Give some basic information about the data
        print("Information about the given corpus\n"
              "###################################\n"
              f"Number of documents: {len(corpus)}\n"
              f"Language (only one language should be displayed): {corpus['lang'].unique()}\n")

        return corpus


    def _create_dictionary(self):
        """Create the vocabulary containing all words found across all the corpus."""
        # Initiate the dictionary
        dictionary = set()

        # Iterate over the text column only
        for doc in tqdm(self.corpus['text'], desc="Creating dictionary"):
            # Directly update the set in place
            dictionary.update(doc.split(" "))  # basic tokenization

        print(f"Dictionary created ({len(dictionary)} unique words) !")

        return dictionary


    def _tf(self, text: list[str]):
        """Compute the Term Frequency (TF) of a text."""
        # Initiate the counter and compute th length of document
        counter = dict.fromkeys(self.dictionary, 0)
        N = len(text)

        # Count the words
        for word in text:
            counter[word] += 1

        tf = {word: freq / N for word, freq in counter.items}

    def _idf(self):
        """Compute the Inverse Document Frequency (IDF) of a corpus."""
        pass

    def fit(self):
        """#TODO"""
        for doc in self.corpus.iterrows():
            pass



class TFIDFRetriever:
    def __init__(self, tf_idf_matrix_path: str):
        self.tf_idf_matrix_path = tf_idf_matrix_path
        self.tf_idf = self._load_tf_idf()

    def _load_tf_idf(self):
        """Load the TF-IDF matrix already computed with TFIDFTrainer method"""
        pass
