# Initialize a BM25 model for each language and save them as pickle files.

from bm25 import BM25

# langs = ['fr', 'es', 'de', 'it', 'ar', 'ko']
langs = ['en']
for lang in langs:
    print(f"Initializing BM25 model for {lang}...")
    path = f"../data/corpus.json/clean_corpus_{lang}.json"
    bm25 = BM25(corpus_path=path)
    print(f"BM25 model for {lang} initialized successfully !\n")
    print(f"Saving BM25 model for {lang}...")
    bm25.save_pickle(f"bm25_models/bm25_{lang}.pkl")
    print(f"BM25 model for {lang} saved successfully !\n")
