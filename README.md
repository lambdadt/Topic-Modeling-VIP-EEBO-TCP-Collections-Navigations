# Georgia Tech VIP: Unlocking and Analyzing Historical Texts - Topic Modeling of EEBO-TCP Data

## Installation
Python 3.10
- `scikit-learn` (1.6.1)
- `numpy` (2.2.2)
- `pandas` (2.2.3)
- `nltk` (3.9.1)
- `spacy` (3.8.4)
  - `python -m spacy download en_core_web_sm`

=======
## Introduction
Welcome to the EEBO Topic Modeling Project repository. This project aims to analyze a collection of Early English documents to identify prevalent themes of the time. Additionally, we seek to explore relationships between specific authors, publishers, and topics. For instance, determining if Author A frequently wrote about Topic T1 or if Publisher P significantly published documents on Topic T2. Our methodologies include utilizing algorithms such as Latent Dirichlet Allocation (LDA), Probabilistic Latent Semantic Indexing (PLSI), and BERTopic.

## Background
Understanding the foundational elements of our project is crucial. Below is a summary of key components:

### Early English Books Online (EEBO)
[Early English Books Online (EEBO)](https://proquest.libguides.com/eebopqp) is a comprehensive digital collection of works printed in English from the first book published in the language up to the era of Spenser and Shakespeare. It contains over 146,000 titles, encompassing a wide range of subjects and genres.

### Text Creation Partnership (TCP)
The [Text Creation Partnership (TCP)](https://textcreationpartnership.org/tcp-texts/eebo-tcp-early-english-books-online/) is a collaborative initiative that has produced thousands of accurate, searchable, full-text transcriptions of early printed books. These transcriptions are encoded in SGML/XML, facilitating detailed textual analysis.

### EEBO TCP
EEBO TCP is a project undertaken by the Text Creation Partnership to digitize and transcribe texts from the EEBO collection, making them accessible for research and analysis. This effort has provided researchers with structured, machine-readable formats of early printed books, allowing for computational and linguistic studies of historical texts.

### EEBO-TCP Navigations Collection
This specific subset of the EEBO-TCP focuses on themes related to travel and navigation. It comprises approximately 1,500 documents, providing a focused corpus for analysis. The dataset has been obtained by forking the repository from [EEBO-TCP-Collections-Navigations](https://github.com/Text-Creation-Partnership/EEBO-TCP-Collections-Navigations), a project funded by the National Endowment for the Humanities to select, key, and encode EEBO-TCP texts related to the theme of travel and navigation.

### Header XML Files
The documents in our dataset are provided in header XML format. These XML files contain structured metadata and textual content, which are essential for our computational analysis. More details about TCP production files can be found [here](https://textcreationpartnership.org/about-the-tcp/historical-documentation/tcp-production-files/).

## Installation
This project requires Python 3.10 and the following dependencies:

```bash
pip install spacy==3.8.4 scikit-learn==1.6.1 numpy==2.2.2 pandas==2.2.3
python -m spacy download en_core_web_sm
```

## Operations

### Preprocessing

#### Generate word count vectors & TF-IDF matrix
```sh
python -m preprocess make_vectors -o out/vectors --vector_dim_limit 5000
```
Run `python -m preprocess make_vectors --help` to explore more options.

#### (Data Exploration) Identify stop words, compute TF-IDF
Run the following Jupyter Notebook to identify stop words and compute TF-IDF:

Run `stopwords_vectors.ipynb`.