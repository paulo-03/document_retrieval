import math
from collections import Counter

import pandas as pd
from tqdm import tqdm


class BM25:
    def __init__(self, corpus_path: str, query_path: str, k1=1.5, b=0.75):
        """
        Initializes the BM25 retriever.

        Args:
            corpus_path: Path to the corpus file.
            query_path: Path to the query file.
            k1: Positive tuning parameter for relevance term frequency (best from 1.2 to 2.0)
            b: Positive tuning parameter for scaling the term weight by document length (best from 0 to 1.0)
        """

        self.lang = None
        self.k1 = k1
        self.b = b

        self.corpus_path = corpus_path
        self.query_path = query_path
        self.corpus_df, self.query_df = self._load_data()

        self.corpus_text = self.corpus_df['text'].tolist()
        self.docids = self.corpus_df['docid'].tolist()
        self.corpus_tokenized = [doc.split() for doc in self.corpus_text]

        self.queries = self.query_df['query'].tolist()
        self.queries_ids = self.query_df['query_id'].tolist()

        self.avgdl = sum(len(doc) for doc in self.corpus_tokenized) / len(
            self.corpus_tokenized)  # average document length
        self.doc_freqs = []  # list of term frequency per document
        self.idf = {}  # inverse document frequency for each term
        self._initialize()

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

        # Load the query data
        print("Loading queries...")
        queries = pd.read_csv(self.query_path)
        queries = queries[queries['lang'] == self.lang]
        print("Queries loaded successfully !\n")

        print("Information about the given queries\n"
              "###################################\n"
              f"Number of queries: {len(queries)}\n"
              f"Language (only one language should be displayed): {queries['lang'].unique()}\n")

        return corpus, queries

    def _initialize(self):
        """Precompute document frequencies and IDF for all terms in the corpus."""
        # Compute document frequencies
        doc_count = len(self.corpus_tokenized)
        term_doc_count = Counter()

        for doc in tqdm(self.corpus_tokenized, desc="Processing documents"):
            doc_term_count = Counter(doc)
            self.doc_freqs.append(doc_term_count)
            for term in doc_term_count.keys():
                term_doc_count[term] += 1

        # Compute IDF for each term
        for term, doc_freq in tqdm(term_doc_count.items(), desc="Computing IDF"):
            # Apply smoothing to avoid division by zero
            self.idf[term] = math.log(1 + (doc_count - doc_freq + 0.5) / (doc_freq + 0.5))

    def _score(self, query, document_idx):
        """
        Computes the BM25 score for a document given a query.
        :param query: List of query terms
        :param document_idx: Index of the document in the corpus
        :return: BM25 score for the document
        """
        doc_term_count = self.doc_freqs[document_idx]
        doc_length = len(self.corpus_tokenized[document_idx])
        score = 0.0

        for term in query:
            if term not in doc_term_count:
                continue  # term not in document, skip

            term_freq = doc_term_count[term]
            idf = self.idf.get(term, 0)  # Get IDF, 0 if term not in corpus
            numerator = term_freq * (self.k1 + 1)
            denominator = term_freq + self.k1 * (1 - self.b + self.b * (doc_length / self.avgdl))
            score += idf * (numerator / denominator)

        return score

    def _rank(self, query):
        """
        Ranks all documents in the corpus for a given query.
        :param query: List of query terms
        :return: List of (document_idx, score) tuples sorted by score
        """
        scores = [(idx, self._score(query, idx)) for idx in range(len(self.corpus_tokenized))]
        return sorted(scores, key=lambda x: x[1], reverse=True)

    def top_10_docid_for_all_queries(self) -> dict:
        """
        Retrieve the top 10 documents for each query and return them in a dictionary.
        :return: Dictionary mapping query_id to a list of top 10 docids
        """
        top_10_docids = {}
        for query_id, query in tqdm(zip(self.queries_ids, self.queries), desc="Ranking queries",
                                    total=len(self.queries_ids)):
            query = query.split()
            ranked = self._rank(query)
            top_10_docids[query_id] = [self.docids[idx] for idx, _ in ranked[:10]]

        return top_10_docids
