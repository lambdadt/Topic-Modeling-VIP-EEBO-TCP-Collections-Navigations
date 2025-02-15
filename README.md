# Georgia Tech VIP: Unlocking and Analyzing Historical Texts - Topic Modeling of EEBO-TCP Data

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
pip install spacy==3.8.4 scikit-learn==1.6.1 numpy==2.2.2 pandas==2.2.3 nltk==3.9.1
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

---

### Running PLSI

The `plsi.py` script runs the Probabilistic Latent Semantic Analysis (PLSI) algorithm on preprocessed data.

**Important Notes:**
- The script expects two CSV files in the specified `--input_dir`: `count_vectors.csv` and `tfidf.csv`.  
- Both files must have identical dimensions as they represent different representations (raw counts vs. TF-IDF) of the same document-word matrix.  
- These CSV files can be generated using the `preprocess.py` script as shown above.
- The output files, `PLSI_P_dz_*.csv` and `PLSI_P_zw_*.csv`, are used as inputs to the `visualization.py` script. Ensure they are stored in the expected output directory.

#### Command-line Arguments

- **--topics** *int*  
  Number of topics to discover. Default is `10`.

- **--verbose** *flag*  
  Enable verbose printing for debugging purposes (currently not active).

- **--input_dir** *str*  
  Path to the directory containing the input CSV files (`count_vectors.csv` and `tfidf.csv`).  
  Default is `/out/vectors`.

- **--output_dir** *str*  
  Path to the directory where the output files will be saved.  
  Default is `vectors_in_csv/plsi_vectors`.  
  **Note:** The generated output files will be used as inputs to `visualization.py`.

- **--max_iter** *int*  
  Maximum number of EM iterations to perform. Default is `50`.

- **--tol** *float*  
  Convergence threshold based on the change in log-likelihood between iterations. Default is `1e-5`.

- **--pct_docs** *float*  
  Percentage of documents to use from the input CSV files (range: `0-100`). Default is `100`.  
  Use this parameter to run the algorithm on a subset of the data for faster experimentation.

- **--matrix_type** *str*  
  Specify which matrix to use for running PLSI. Options are `tfidf` (default) or `count`.  
  This selects whether the algorithm processes the TF-IDF matrix or the count vectors.

#### Example Usage

```sh
python plsi.py --topics 10 --input_dir out/vectors --output_dir vectors_in_csv/plsi_vectors --max_iter 100 --tol 1e-5 --pct_docs 100 --matrix_type count
```

In this example, the script runs with 10 topics on the count vectors (using 100% of the documents), with up to 100 iterations and a tolerance of `1e-5`. It runs for approximately 1 minute and 50 seconds (1.09s/it).

---

### Running LSI

The `lsi.py` script runs the Latent Semantic Analysis (LSA) algorithm on preprocessed data.

**Important Notes:**
- The script expects two CSV files in the specified `--input_dir`: `count_vectors.csv` and `tfidf.csv`.
- Both files must have identical dimensions as they represent different representations (raw counts vs. TF-IDF) of the same document-word matrix.
- These CSV files can be generated using the `preprocess.py` script as shown above.
- The output files vary depending on the `--output_type` argument:
  - If `--output_type` is set to `prob` (default), the output files `LSI_P_dz_*.csv` and `LSI_P_zw_*.csv` (probability files) are saved. These are used as inputs to the `visualization.py` script.
  - If `--output_type` is set to `svd`, the output files `LSI_U_*.csv`, `LSI_S_*.csv`, and `LSI_Vt_*.csv` (SVD files) are saved for further inspection.

#### Command-line Arguments

- **--topics** *int*  
  Number of topics to discover. Default is `10`.

- **--verbose** *flag*  
  Enable verbose logging for debugging purposes (currently not active).

- **--input_dir** *str*  
  Path to the directory containing the input CSV files (`count_vectors.csv` and `tfidf.csv`).  
  Default is `out/vectors`.

- **--output_dir** *str*  
  Path to the directory where the output files will be saved.  
  Default is `out/lsi_vectors`.

- **--pct_docs** *float*  
  Percentage of documents to use from the input CSV files (range: `0-100`). Default is `100`.  
  Use this parameter to run the algorithm on a subset of the data for faster experimentation.

- **--matrix_type** *str*  
  Specify which matrix to use for running LSA. Options are `tfidf` (default) or `count`.  
  This selects whether the algorithm processes the TF-IDF matrix or the count vectors.

- **--output_type** *str*  
  Specify the type of output files to save. Options are:
  - `prob` (default): Save probability files (document-topic and topic-word distributions).
  - `svd`: Save SVD files (U, S, and Vt matrices) for further inspection.

#### Example Usage

```sh
python lsi.py --topics 10 --input_dir out/vectors --output_dir vectors_in_csv/lsi_vectors --pct_docs 100 --matrix_type count --output_type prob
```

In this example, the script runs with 10 topics on the count vectors (using 100% of the documents) and saves the probability files. It takes less than 2 seconds.