"""
Python script to help the loading of data of all kinds of datasets
"""

import os
import re
import spacy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from nltk.corpus import stopwords

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


class RawCorpusAnalysis:
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
        tqdm.pandas(desc="Splitting the text of each document by words")
        corpus_length = self.corpus.progress_apply(lambda row: len(row['text'].split()), axis=1)

        # Print some statistics
        stats = {
            "max": [corpus_length.max()],
            "min": [corpus_length.min()],
            "mean": [corpus_length.mean()],
            "std": [corpus_length.std()],
            "median": [corpus_length.median()],
            "95th_quantile": [corpus_length.quantile(0.95)]
        }

        stat_df = pd.DataFrame.from_dict(stats)
        stat_df.index = ['words_number']

        # Display the histogram distribution of words number
        plt.figure()

        # Plotting the histogram
        sns.histplot(corpus_length)

        # Adding x-label and title
        plt.xlabel('Number of Words')
        plt.title('Distribution of Document Lengths')

        # Display the plot
        plt.show()

        return stat_df


class CleanCorpusAnalysis:
    """Class to apply some methods to a clean corpus for analysis purpose"""

    def __init__(self, root_clean_corpus_path: str):
        print("Loading clean corpus...")
        self.data_clean = self._load_clean_dataset(root_clean_corpus_path)
        print("Done.")

    @staticmethod
    def _load_clean_dataset(root_clean_corpus_path):
        # Find all the clean dataset in the root corpus path and the pre-processing steps chosen
        corpus_paths = ["/".join([root_clean_corpus_path, corpus])
                        for corpus in os.listdir(root_clean_corpus_path)
                        if corpus.endswith('.json')]

        langs = [corpus_path[-7:-5] for corpus_path in corpus_paths]

        return {
            lang: pd.read_json(corpus_path, lines=True)
            for lang, corpus_path in zip(langs, corpus_paths)
        }

    def docs_stats(self):
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
        for lang, docs in self.data_clean.items():
            # Compute the number of words per documents in a specific languages
            docs_length = docs.apply(lambda row: len(row['text'].split()), axis=1)

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

    def show_docs(self):
        """TODO"""
        print(f"{'-' * 50}\n"
              "Current state of documents (first 500 characters):\n"
              f"{'-' * 50}\n")

        for lang, docs in self.data_clean.items():
            doc = docs.iloc[0]
            docid = doc['docid']
            text = re.sub(r'\n+', ' ', doc['text'][:500])

            print(f"Language: {lang}\n"
                  f"Docid: {docid}\n"
                  f"{'-' * 50}\n"
                  f"Text: \n{text}\n")


# Create parent class to allow data preprocessing for corpus and queries
class TextClean:
    def __init__(self, show_progress: bool = True):
        self.data = None
        self.data_clean = None
        self.nlp = None
        self.show_progress = show_progress  # True if you want to see progress bars while pre-processing.

    def split_per_lang(self):
        """TODO"""
        # Get the unique languages
        langs = self.data['lang'].unique()

        # Split the dataframe per language
        self.data_clean = {
            lang: self.data[self.data['lang'] == lang]
            for lang in langs
        }

    @staticmethod
    def _simple_tokenizer(text):
        """Use regex to find words and punctuation if no SpaCY model is available."""
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
        return tokens

    def lc(self):
        """TODO"""
        for lang, docs in self.data_clean.items():

            # Select the correct SpaCY model
            nlp = SPACY_MODELS.get(lang)

            # Initiate the tqdm progress bar if show_progress to True
            tqdm.pandas(desc=f"LC for '{lang}' texts", disable=not self.show_progress)

            if nlp:
                tokenizer = nlp.tokenizer

                # Lower case all the docs and removes punctuation with spacy
                docs['text'] = docs.progress_apply(
                    lambda row: " ".join(token.text
                                         for token in tokenizer(row['text'].lower())
                                         if not token.is_punct),
                    axis=1)

            else:
                # Take the simple tokenizer since no SpaCY model is founded
                tokenizer = self._simple_tokenizer

                # Lower case all the docs and removes punctuation with spacy
                docs['text'] = docs.progress_apply(
                    lambda row: " ".join(token
                                         for token in tokenizer(row['text'].lower())
                                         if token.isalnum()),
                    axis=1)



    def lc_sw(self):
        """TODO"""
        for lang, docs in self.data_clean.items():

            # Select the correct SpaCY model
            nlp = SPACY_MODELS.get(lang)

            # Initiate the tqdm progress bar if show_progress to True
            tqdm.pandas(desc=f"LC and SW for '{lang}' texts", disable=not self.show_progress)

            if nlp:

                # Retrieve the tokenizer from SpaCY model
                tokenizer = nlp.tokenizer

                # Lower case all the docs and removes punctuation with spacy
                docs['text'] = docs.progress_apply(
                    lambda row: " ".join(token.text
                                         for token in tokenizer(row['text'].lower())
                                         if not token.is_punct and not token.is_stop),
                    axis=1)

            else:
                # Check if stopwords exist in NLTK
                lang_stopwords = STOPWORDS_DICT.get(lang)

                # Take the simple tokenizer since no SpaCY model is founded
                tokenizer = self._simple_tokenizer

                if lang_stopwords:
                    # Lower case all the docs and removes punctuation with SpaCY and stopwords with NLTK
                    docs['text'] = docs.progress_apply(
                        lambda row: " ".join(token
                                             for token in tokenizer(row['text'].lower())
                                             if token.isalnum() and token not in lang_stopwords),
                        axis=1)

                else:
                    # Will be the same file from lc, so just drag it to the folder
                    pass

    def lc_sw_l(self):
        """TODO"""
        for lang, docs in self.data_clean.items():

            # Select the correct SpaCY model
            nlp = SPACY_MODELS.get(lang)

            # Initiate the tqdm progress bar if show_progress to True
            tqdm.pandas(desc=f"LC, SW and L for '{lang}' texts", disable=not self.show_progress)

            if nlp:
                # Lower case all the docs and removes punctuation with spacy
                docs['text'] = docs.progress_apply(
                    lambda row: ' '.join(token.lemma_.lower()
                                         for token in nlp(row['text'])
                                         if not token.is_punct and not token.is_stop),
                    axis=1)

            else:
                # Will be the same file from lc_sw, so just drag it to the folder
                pass


class CorpusClean(TextClean):
    """Class to apply some methods to a corpus for pre-processing purpose"""

    def __init__(self, corpus_path: str, show_progress: bool = True, stat: bool = False):
        super().__init__(show_progress=show_progress)
        # Load the entire corpus (~3 min with my CPU, MacPro 2016, 2,5 GHz, Intel I7)
        print("Loading Corpus... (can take up to 3 minutes)")
        self.data = pd.read_json(corpus_path)
        print("Data Loaded ! \n")

    def store(self, path: str):
        """TODO"""
        # Ensure the folder exists or create it if it doesn't
        os.makedirs(path, exist_ok=True)

        for lang, docs in tqdm(self.data_clean.items(), desc="Storing current clean dataset into disk"):
            docs.to_json(f'{path}/clean_corpus_{lang}.json', orient='records', lines=True)


class QueryClean(TextClean):
    """Class to apply some methods to queries for pre-processing purpose"""

    def __init__(self, queries_path, processing_wanted: str, show_progress: bool = False):
        super().__init__(show_progress=show_progress)

        # Check if the required processes is available
        if processing_wanted in ['lc', 'lc_sw', 'lc_sw_l']:
            pass
        else:
            raise ValueError("One of the processes given are not handle by our implementation.\n\n"
                             "Make sure you choose processes into this list: \n"
                             "['lower_case', 'stop_words', 'lemmatization']")

        # If nothing to raise, we are good to go
        self.process = processing_wanted

        print("Loading queries...")
        # Rename to allow the use of class shared functions from TextClean
        self.data = pd.read_csv(queries_path).rename(columns={'query': 'text'})
        print("Done! \n")

    def pre_process(self):
        # First we split the queries per language
        print("Starting pre-processing queries...")

        # For any processing choice, the first step is to split the documents per languages
        self.split_per_lang()

        if self.process == 'lc':
            self.lc()

        elif self.process == 'lc_sw':
            self.lc_sw()

        elif self.process == 'lc_sw_l':
            self.lc_sw_l()

        print("Pre-processing finished !\n")

        return list(self.data_clean.keys())
