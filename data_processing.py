"""
Python script to perform the data processing steps easily and store the results.

Author: Paulo Ribeiro
"""

from data_helpers import CorpusClean
import warnings

warnings.filterwarnings('ignore')


def main(processing_wanted: str):
    # First check that the processing step asked is available
    if processing_wanted in ['lc', 'lc_sw', 'lc_sw_l']:
        # Initiate the class CorpusClean to perform easily the processing steps
        corpus = CorpusClean(
            corpus_path="data/corpus.json/corpus.json",
            show_progress=True
        )

        # For any processing choice, the first step is to split the documents per languages
        corpus.split_per_lang()

        if processing_wanted == 'lc':
            corpus.lc()
            corpus.store(path="clean_data/lc")

        elif processing_wanted == 'lc_sw':
            corpus.lc_sw()
            corpus.store(path="clean_data/lc_sw")

        elif processing_wanted == 'lc_sw_l':
            corpus.lc_sw_l()
            corpus.store(path="clean_data/lc_sw_l")

    else:
        error_msg = (f"The processing asked ('{processing_wanted}') is not handle.\n"
                     f"Please choose between this three choices: ['lc', 'lc_sw', 'lc_sw_l']")
        raise ValueError(error_msg)


if __name__ == '__main__':

    for process in ['lc', 'lc_sw', 'lc_sw_l']:
        main(
            processing_wanted=process
        )
