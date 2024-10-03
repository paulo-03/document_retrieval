"""
Python script to help the loading of data of all kinds of datasets
"""

import re
import pandas as pd
from tqdm.notebook import tqdm
from nltk.corpus import stopwords


korean_stopwords = [
    "이", "그", "저", "것", "저것", "그것", "그러나", "그래서", "또한", "하지만",
    "그렇지만", "그리고", "또한", "또는", "만약", "만약에", "왜냐하면", "때문에", 
    "따라서", "등", "그들", "그녀", "그녀들", "그들", "있다", "없다", "하다", 
    "되다", "아니다", "이다", "에", "에서", "와", "과", "하고", "나", "너", "너희", 
    "나의", "그의", "그녀의", "우리", "너희의", "그들의", "저희", "나에게", 
    "너에게", "그에게", "우리에게", "너희에게", "이곳", "그곳", "저곳", "이런", 
    "저런", "그런", "이런저런", "모든", "매우", "아주", "좀", "조금", "너무", 
    "많이", "자주", "가끔", "항상", "정말", "진짜", "그냥", "벌써", "이제", 
    "다시", "너무나", "항상", "자꾸", "나중에", "한편", "게다가", "덕분에", 
    "아직", "하여튼", "어쨌든", "바로", "때때로", "한때", "잠시", "잠깐", 
    "갑자기", "확실히", "대개", "거의", "무척", "역시", "정말로", "분명히", 
    "따로", "서로", "마치", "도대체"
]

korean_stopwords = set(korean_stopwords)

# Pre-define stopwords for each language
STOPWORDS_DICT = {
    'en': set(stopwords.words('english')),
    'fr': set(stopwords.words('french')),
    'de': set(stopwords.words('german')),
    'es': set(stopwords.words('spanish')),
    'it': set(stopwords.words('italian')),
    'ar': set(stopwords.words('arabic')),
    'ko': korean_stopwords  # Korean stopwords handling may need custom work; an empty set for now
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
