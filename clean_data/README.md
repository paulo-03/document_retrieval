# Clean Data

This folder will contain three processed datasets. Due to data privacy requirements and to keep this GitHub repository 
lightweight, these datasets are not included here and must be created locally by running the `corpus_preprocessing.py` 
script. This script will generate the following `.json` files:

- **lc**: *LowerCase Corpus* – All text is converted to lowercase for consistency.
- **lc_sw**: *LowerCase StopWords* – Text is converted to lowercase, and stop words (based on language) are removed.
- **lc_sw_l**: *LowerCase StopWords Lemmatization* – Text is lemmatized, converted to lowercase, and then stop words 
- are removed.

> **Note**: The corpus is multilingual. Each language is processed separately, generating distinct `.json` files for 
> each language in the `lc`, `lc_sw`, and `lc_sw_l` folders.
