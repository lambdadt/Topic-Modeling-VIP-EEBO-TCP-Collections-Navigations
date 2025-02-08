###############################################################################
# REFERENCE:
# Latent Semantic Analysis
# Source: https://nlp.stanford.edu/IR-book/html/htmledition/latent-semantic-indexing-1.html
# Author: Taehwan Park
###############################################################################

import argparse
import json
import numpy as np
import pandas as pd
import sys
import os
from tqdm import tqdm  # For progress visualization

def lsa(
    doc_word_matrix: np.ndarray,
    n_topics: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs the LSA algorithm using SVD decomposition.

    Parameters
    ----------
    doc_word_matrix : np.ndarray
        A 2D array of shape (n_docs, n_words) containing counts or TF-IDF values.
    n_topics : int
        Number of latent topics (i.e., number of singular values to retain).

    Returns
    -------
    P_dz : np.ndarray
        Document-topic “distribution” (normalized).
    P_zw : np.ndarray
        Topic-word “distribution” (normalized).
    U_k : np.ndarray
        Truncated U matrix from SVD.
    s_k : np.ndarray
        Truncated singular values.
    Vt_k : np.ndarray
        Truncated V^T matrix from SVD.
    """
    # Compute the SVD of the document-word matrix.
    U, s, Vt = np.linalg.svd(doc_word_matrix, full_matrices=False)

    # Retain only the top n_topics singular values/vectors.
    U_k = U[:, :n_topics]
    s_k = s[:n_topics]
    Vt_k = Vt[:n_topics, :]

    # Weight the matrices with the square roots of the singular values.
    sqrt_s_k = np.sqrt(s_k)
    doc_topic = U_k * sqrt_s_k[np.newaxis, :]      # shape: (n_docs, n_topics)
    topic_word = sqrt_s_k[:, np.newaxis] * Vt_k    # shape: (n_topics, n_words)

    # Take the absolute values to mimic non-negative probabilities.
    doc_topic = np.abs(doc_topic)
    topic_word = np.abs(topic_word)

    # Normalize each document's topic representation so each row sums to 1.
    P_dz = doc_topic.copy()
    row_sums = P_dz.sum(axis=1, keepdims=True)
    P_dz = np.divide(P_dz, row_sums, out=np.zeros_like(P_dz), where=row_sums != 0)

    # Normalize each topic's word representation so each row sums to 1.
    P_zw = topic_word.copy()
    row_sums = P_zw.sum(axis=1, keepdims=True)
    P_zw = np.divide(P_zw, row_sums, out=np.zeros_like(P_zw), where=row_sums != 0)

    return P_dz, P_zw, U_k, s_k, Vt_k


def run_lsa(
    matrix_df: pd.DataFrame,
    vocabulary: list[str],
    n_topics: int,
    matrix_name: str = "Count Vectors",
    verbose: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Wrapper to run LSA on a given DataFrame, then print detailed topic and document analyses.

    Parameters
    ----------
    matrix_df : pd.DataFrame
        Rows = documents, Columns = words.
    vocabulary : list[str]
        List of words aligned with matrix_df columns.
    n_topics : int
        Number of topics to discover.
    matrix_name : str
        Descriptor for logging (e.g., "Count Vectors" or "TF-IDF").
    verbose : bool
        Whether to print verbose output.

    Returns
    -------
    P_dz : np.ndarray
        Document-topic distribution for matrix_df.
    P_zw : np.ndarray
        Topic-word distribution for matrix_df.
    U_k : np.ndarray
        Truncated U matrix.
    s_k : np.ndarray
        Truncated singular values.
    Vt_k : np.ndarray
        Truncated V^T matrix.
    """
    doc_word_matrix = matrix_df.values
    n_docs, n_words = doc_word_matrix.shape

    print(f"\n--- Running LSA on {matrix_name} ---")
    print(f"Data: {n_docs} documents x {n_words} words | Topics: {n_topics}")

    # Run LSA.
    P_dz, P_zw, U_k, s_k, Vt_k = lsa(doc_word_matrix, n_topics)

    return P_dz, P_zw, U_k, s_k, Vt_k


def main():
    parser = argparse.ArgumentParser(description="Run LSA on count or TF-IDF matrices.")
    parser.add_argument("--topics", type=int, default=10,
                        help="Number of topics to discover (default=10).")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging (default=False).")
    parser.add_argument("--output_dir", type=str, default="out/lsi_vectors",
                        help="Path to the output directory (default=out/lsi_vectors).")
    parser.add_argument("--pct_docs", type=float, default=100,
                        help="Percentage of documents to use from CSV files (0-100).")

    args = parser.parse_args()
    verbose = args.verbose
    n_topics = args.topics

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    count_vectors_df = pd.read_csv("./out/vectors/count_vectors.csv", index_col=0)
    tfidf_df = pd.read_csv("./out/vectors/tfidf.csv", index_col=0)

    # Select a subset of documents based on the provided percentage (if < 100).
    if args.pct_docs < 100:
        n_total = count_vectors_df.shape[0]
        n_select = int(n_total * args.pct_docs / 100)
        count_vectors_df = count_vectors_df.iloc[:n_select]
        tfidf_df = tfidf_df.iloc[:n_select]

    with open("./out/vectors/meta.json", "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    # Retrieve vocabulary from DataFrame columns.
    vocabulary = list(count_vectors_df.columns)

    # Run LSA on Count Vectors (uncomment to use).
    # P_dz_count, P_zw_count, U_k_count, s_k_count, Vt_k_count = run_lsa(
    #     matrix_df=count_vectors_df,
    #     vocabulary=vocabulary,
    #     n_topics=n_topics,
    #     matrix_name="Count Vectors",
    #     verbose=verbose
    # )
    # pd.DataFrame(U_k_count).to_csv(f"{args.output_dir}/LSI_U_{n_topics}_count.csv", index=False)
    # pd.DataFrame(s_k_count).to_csv(f"{args.output_dir}/LSI_S_{n_topics}_count.csv", index=False)
    # pd.DataFrame(Vt_k_count).to_csv(f"{args.output_dir}/LSI_Vt_{n_topics}_count.csv", index=False)

    # Run LSA on TF-IDF
    P_dz_tfidf, P_zw_tfidf, U_k_tfidf, s_k_tfidf, Vt_k_tfidf = run_lsa(
        matrix_df=tfidf_df,
        vocabulary=vocabulary,
        n_topics=n_topics,
        matrix_name="TF-IDF",
        verbose=verbose
    )

    # Save U, S, Vt from TF-IDF-based LSA
    pd.DataFrame(U_k_tfidf).to_csv(f"{args.output_dir}/LSI_U_{n_topics}.csv", index=False)
    pd.DataFrame(s_k_tfidf).to_csv(f"{args.output_dir}/LSI_S_{n_topics}.csv", index=False)
    pd.DataFrame(Vt_k_tfidf).to_csv(f"{args.output_dir}/LSI_Vt_{n_topics}.csv", index=False)


if __name__ == "__main__":
    main()
