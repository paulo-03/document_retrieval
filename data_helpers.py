"""
Python script to help the loading of data of all kinds of datasets
"""

import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from nltk.corpus import stopwords

# Enable tqdm for pandas
tqdm.pandas()



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

def clean_sentence(text, lang):
    """
    Cleans the input text by:
    1. Lowercasing the text
    2. Removing stopwords for the specified language
    3. Removing non-alphabetic characters
    """
    # Get stopwords for the language
    lang_stopwords = STOPWORDS_DICT.get(lang, set())

    # Convert to lowercase
    text = text.lower()

    # Check if the language have stopwords or not (i.e. korean)
    if lang_stopwords:
        # Remove non-alphabetic characters and split into words
        words = re.findall(r'\b\w+\b', text)

        # Remove stopwords
        words_cleaned = [word for word in words if word not in lang_stopwords]

        # Join the words back into a sentence
        return ' '.join(words_cleaned)

    else:
        return text


def split_clean_corpus_per_lang(corpus_path: str = 'data/corpus.json/corpus.json'):
    """
    This function splits and cleans a multilingual text corpus by language and saves the cleaned data into separate
    JSON files for each language. It processes each language by filtering the text data, cleaning the sentences
    (removing stopwords and converting to lowercase), and exporting the cleaned data into language-specific files.
    """
    # Load the entire corpus (~3 min with my CPU, MacPro 2016, 2,5 GHz, Intel I7)
    print("Loading corpus... (can take up to 3 minutes)")
    corpus = pd.read_json(corpus_path)
    print("Corpus Loaded ! \n")

    # Get the unique languages
    langs = corpus['lang'].unique()

    for lang in tqdm(langs, desc="Splitting and cleaning the corpus by languages"):

        # Filter the DataFrame for each language
        lang_df = corpus[corpus['lang'] == lang]

        # Apply cleaning function for each row based on its language
        lang_df['text'] = lang_df.apply(lambda row: clean_sentence(row['text'], row['lang']), axis=1)

        # Create a filename based on the language
        filename = f'data/corpus.json/clean_corpus_{lang}.json'

        # Save the filtered DataFrame to a JSON file
        lang_df.to_json(filename, orient='records', lines=True)

        print(f'Saved: {filename}')


def corpus_langs(corpus: pd.DataFrame):
    """Print the different languages we have in the corpus."""
    langs = corpus['lang'].unique()

    print(f"The corpus contains these languages:\n{langs}")


def corpus_text_length(corpus: pd.DataFrame):
    """Display the distribution of the length, base on the word splitting basis, of our documents."""
    corpus_length: pd.Series = corpus.progress_apply(lambda row: len(row['text'].split()), axis=1)

    plt.figure()

    # Plotting the histogram
    sns.histplot(corpus_length)

    # Adding x-label and title
    plt.xlabel('Number of Words')
    plt.title('Distribution of Document Lengths')

    # Display the plot
    plt.show()