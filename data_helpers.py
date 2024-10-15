"""
Python script to help the loading of data of all kinds of datasets
"""

import re
import spacy
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from nltk.corpus import stopwords
from multiprocessing import Pool, cpu_count

# Pre-define stopwords for each language (you can keep your STOPWORDS_DICT)
STOPWORDS_DICT = {
    'en': set(stopwords.words('english')),
    'fr': set(stopwords.words('french')),
    'de': set(stopwords.words('german')),
    'es': set(stopwords.words('spanish')),
    'it': set(stopwords.words('italian')),
    'ar': set(stopwords.words('arabic')),
    'ko': None  # Korean stopwords handling may need custom work; an empty set for now
}

# Load spaCy models for different languages
SPACY_MODELS = {
    'en': spacy.load('en_core_web_sm', disable=['parser', 'ner']),
    'fr': spacy.load('fr_core_news_sm', disable=['parser', 'ner']),
    'de': spacy.load('de_core_news_sm', disable=['parser', 'ner']),
    'es': spacy.load('es_core_news_sm', disable=['parser', 'ner']),
    'it': spacy.load('it_core_news_sm', disable=['parser', 'ner']),
    'ar': None,  # No spaCy model for Arabic ('ar'); an empty set for now
    'ko': None  # No spaCy model for Korean ('ko'); an empty set for now
}


class CorpusClean:
    """Class to apply some methods to a corpus for pre-processing purpose"""
    def __init__(self, corpus_path):
        # Load the entire corpus (~3 min with my CPU, MacPro 2016, 2,5 GHz, Intel I7)
        print("Loading corpus... (can take up to 3 minutes)")
        self.corpus = pd.read_json(corpus_path)
        print("Corpus Loaded ! \n")

        self.corpus_clean = None
        self.nlp = None

    def split_per_lang(self):
        """TODO"""
        # Get the unique languages
        langs = self.corpus['lang'].unique()

        # Split the dataframe per language
        self.corpus_clean = {
            lang: self.corpus[self.corpus['lang'] == lang]
            for lang in langs
        }

        # Display the words statistics of current split corpus
        return self._docs_stats_()

    def lower_case(self):
        """TODO"""

        for lang, docs in self.corpus_clean.items():
            # Lower case all the docs
            tqdm.pandas(desc=f"Lower casing '{lang}' docs")
            docs['text'] = docs.progress_apply(lambda row: row['text'].lower(), axis=1)

            # Update the corpus clean variable
            self.corpus_clean[lang] = docs

    def stop_words(self):
        """TODO"""
        for lang, docs in self.corpus_clean.items():
            # Get stopwords for the language
            lang_stopwords = STOPWORDS_DICT.get(lang)

            # Only remove stopwords for languages which we have the stop words list
            if lang_stopwords:
                # Remove all the stop words
                tqdm.pandas(desc=f"Removing stop words for '{lang}' docs")
                docs['text'] = docs.progress_apply(
                    lambda row: ' '.join([word for word in row['text'].split() if word not in lang_stopwords]),
                    axis=1
                )

                # Update the corpus clean variable
                self.corpus_clean[lang] = docs

        # Display the words statistics of current split corpus
        return self._docs_stats_()

    def _lemmatize(self, text):
        """Function to lemmatize a single chunk of texts"""
        return ' '.join([token.lemma_ for token in self.nlp(text)])

    def lemmatization(self):
        """Efficiently lemmatize the documents in the corpus."""
        for lang, docs in self.corpus_clean.items():
            # Get the appropriate spaCy model for the language
            self.nlp = SPACY_MODELS.get(lang)

            if self.nlp:
                print(f"Starting lemmatizing '{lang}' docs!")
                # Use multiprocessing to lemmatize the texts in parallel
                #with Pool(cpu_count() // 3) as pool:
                #    lemmatized_texts = pool.map(self._lemmatize, docs['text'])
                tqdm.pandas(desc=f"Lemmatizing '{lang}' docs")
                docs['text'] = docs.progress_apply(lambda row: self._lemmatize(row['text']), axis=1)

                # Update the 'text' column with the lemmatized text
                #docs['text'] = pd.Series(lemmatized_texts)

                # Update the corpus_clean variable
                self.corpus_clean[lang] = docs

                # Display information to the user
                print(f"Finished lemmatizing '{lang}' docs!\n")

            else:
                print(f"No Lemmatizer found for '{lang}' docs!\n")

        # Display the words statistics of current split corpus
        return self._docs_stats_()

    def store(self, path: str):
        """TODO"""
        for lang, docs in tqdm(self.corpus_clean.items(), desc="Storing current clean dataset into disk"):
            docs.to_json(f'{path}/clean_corpus_{lang}.json', orient='records', lines=True)

    def show_current_docs(self):
        """TODO"""
        print(f"{'-' * 50}\n"
              "Current state of documents (first 300 characters):\n"
              f"{'-' * 50}\n")

        for lang, docs in self.corpus_clean.items():
            doc = docs.iloc[0]
            docid = doc['docid']
            text = doc['text']

            print(f"Language: {lang}\n"
                  f"Docid: {docid}\n"
                  f"{'-' * 50}\n"
                  f"Text:\n{text[:300]}\n")

    def _docs_stats_(self):
        """TODO"""
        # Compute some statistics for the split documents
        split_stats = {
            "lang": [],
            "num_docs": [],
            "max_words": [],
            "min_words": [],
            "mean_words": [],
            "median_words": [],
            "95th_quantile_words": []
        }
        for lang, docs in self.corpus_clean.items():
            # Compute the number of words per documents in a specific languages
            tqdm.pandas(desc=f"Stats for '{lang}' docs")
            docs_length = docs.progress_apply(lambda row: len(row['text'].split()), axis=1)

            # Update the dict
            split_stats['lang'].append(lang)
            split_stats['num_docs'].append(len(docs))
            split_stats['max_words'].append(docs_length.max())
            split_stats['min_words'].append(docs_length.min())
            split_stats['mean_words'].append(docs_length.mean())
            split_stats['median_words'].append(docs_length.median())
            split_stats['95th_quantile_words'].append(docs_length.quantile(0.95))

        # Display the stats
        return pd.DataFrame.from_dict(split_stats)


class CorpusAnalysis:
    """Class to apply some methods to a corpus for analysis purpose"""
    def __init__(self, corpus_path):
        # Load the entire corpus (~3 min with my CPU, MacPro 2016, 2,5 GHz, Intel I7)
        print("Loading corpus... (can take up to 3 minutes)")
        self.corpus = pd.read_json(corpus_path)
        self.N = len(self.corpus)
        print(f"Corpus Loaded ! ({self.N} documents) \n")

    def langs(self):
        """Print the different languages we have in the corpus and the number of doc for each."""
        # Compute the number of document per language
        langs_and_num_docs = self.corpus[['lang', 'docid']].groupby(
            by="lang"
        ).count().rename(
            columns={'docid': 'num_docs'}
        )

        # Compute the ratio of document number that represent each language
        langs_and_num_docs['ratio_full_corpus'] = langs_and_num_docs.apply(
            lambda row: row['num_docs'] / self.N, axis=1
        )

        return langs_and_num_docs

    def text_length(self):
        """Display the distribution of the length, base on the word splitting basis, of our documents."""
        corpus_length = self.corpus.progress_apply(lambda row: len(row['text'].split()), axis=1)

        # Print some statistics
        print(f"Some statistics about the number of words distribution over all the corpus:\n"
              f"max: {corpus_length.max():.2f}\n"
              f"min: {corpus_length.min():.2f}\n"
              f"median: {corpus_length.median():.2f}\n"
              f"95% quantile: {corpus_length.quantile(0.95):.2f}\n"
              f"mean: {corpus_length.mean():.2f}\n"
              f"std: {corpus_length.std():.2f}")

        # Display the histogram distribution of words number
        plt.figure()

        # Plotting the histogram
        sns.histplot(corpus_length)

        # Adding x-label and title
        plt.xlabel('Number of Words')
        plt.title('Distribution of Document Lengths')

        # Display the plot
        plt.show()


class QueryClean:
    def __init__(self):
        pass
