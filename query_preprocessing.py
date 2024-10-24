import string
from typing import Union

import pandas as pd
import spacy
from nltk.corpus import stopwords

# Pre-define stopwords for each language
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


def split_per_lang(df):
    """Split the dataframe per language."""
    langs = df['lang'].unique()
    return {lang: df[df['lang'] == lang] for lang in langs}


def lower_case(dfs):
    """Lower case all the documents."""
    for lang, query_df in dfs.items():
        # tqdm.pandas(desc=f"Lower casing '{lang}' query_df")
        query_df['query'] = query_df.apply(lambda row: row['query'].lower(), axis=1)
        dfs[lang] = query_df
    return dfs


def remove_stop_words(dfs):
    """Remove stop words from all the documents."""
    for lang, query_df in dfs.items():
        lang_stopwords = STOPWORDS_DICT.get(lang)
        if lang_stopwords:
            # tqdm.pandas(desc=f"Removing stop words for '{lang}' query_df")
            query_df['query'] = query_df.apply(
                lambda row: ' '.join([word for word in row['query'].split() if word not in lang_stopwords]),
                axis=1
            )
            dfs[lang] = query_df
    return dfs


def lemmatize(dfs):
    """Lemmatize all the documents."""
    for lang, query_df in dfs.items():
        nlp = SPACY_MODELS.get(lang)
        if nlp:
            # tqdm.pandas(desc=f"Lemmatizing '{lang}' query_df")
            query_df['query'] = query_df.apply(
                lambda row: ' '.join([token.lemma_ for token in nlp(row['query'])]), axis=1)
            dfs[lang] = query_df
    return dfs


def remove_punctuation(dfs):
    """Remove punctuation from all the documents."""
    for lang, query_df in dfs.items():
        # tqdm.pandas(desc=f"Removing punctuation for '{lang}' query_df")
        query_df['query'] = query_df.apply(
            lambda row: row['query'].translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))), axis=1)
        dfs[lang] = query_df
    return dfs


def preprocess_queries(query_source: Union[str, pd.DataFrame]) -> pd.DataFrame:
    if isinstance(query_source, str):
        query_df = pd.read_csv(query_source)
    else:
        query_df = query_source

    dfs = split_per_lang(query_df)
    dfs = lower_case(dfs)
    dfs = remove_stop_words(dfs)
    dfs = remove_punctuation(dfs)
    dfs = lemmatize(dfs)
    return pd.concat(dfs.values())