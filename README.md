# Georgia Tech VIP: Unlocking and Analyzing Historical Texts; Project: Topic Modeling of EEBO-TCP data
The xml files in this repository used for topic modeling have been obtained by forking the repository from EEBO-TCP-Collections-Navigations
File distribution for EEBO-TCP Collections: Navigations, a project funded by the National Endowment for the Humanities to select, key, and encode EEBO-TCP texts related to the theme of travel and navigation. 

# Installation
Python 3.10
- `scikit-learn` (1.6.1)
- `numpy` (2.2.2)
- `pandas` (2.2.3)
- `nltk` (3.9.1)
- `spacy` (3.8.4)
  - `python -m spacy download en_core_web_sm`

# Operations
## Preprocessing
### Generate word count vectors & TF-IDF matrix
```sh
python -m preprocess make_vectors -o out/vectors --vector_dim_limit 5000
```
Run `python -m preprocess make_vectors --help` to explore more options.

### (Data Exploration) Identify stop words, compute TF-IDF
Run `stopwords_vectors.ipynb`.
