###############################################################################
# REFERENCE:
# Probabilistic Latent Semantic Analysis
# Source: https://en.wikipedia.org/wiki/Probabilistic_latent_semantic_analysis
# Author: Taehwan Park
###############################################################################

import argparse
import json
import numpy as np
import pandas as pd
import os
import sys
from tqdm import tqdm

def plsi(
    doc_word_matrix: np.ndarray,
    n_topics: int,
    max_iter: int = 50,
    tol: float = 1e-5,
    verbose: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Runs the PLSI algorithm using an EM approach.

    Parameters
    ----------
    doc_word_matrix : np.ndarray
        A 2D array of shape (n_docs, n_words) containing counts or TF-IDF values.
    n_topics : int
        Number of latent topics.
    max_iter : int
        Maximum number of EM iterations.
    tol : float
        Convergence threshold for the change in log-likelihood.
    verbose : bool
        Whether to print additional details.

    Returns
    -------
    P_dz : np.ndarray
        Document-topic distribution (normalized), shape (n_docs, n_topics).
    P_zw : np.ndarray
        Topic-word distribution (normalized), shape (n_topics, n_words).
    """
    n_docs, n_words = doc_word_matrix.shape
    np.random.seed(0)

    # Random initialization of doc-topic (P_dz) and topic-word (P_zw)
    P_dz = np.random.rand(n_docs, n_topics)
    P_zw = np.random.rand(n_topics, n_words)

    # Normalize distributions
    P_dz /= P_dz.sum(axis=1, keepdims=True)
    P_zw /= P_zw.sum(axis=1, keepdims=True)

    # Posterior P(z|d,w)
    P_z_given_d_w = np.zeros((n_docs, n_words, n_topics), dtype=float)

    prev_likelihood = 0.0
    for iteration in tqdm(range(max_iter), desc="EM iterations"):
        # E-step: Calculate P(z|d,w)
        for i in range(n_docs):
            for w in range(n_words):
                topic_probs = P_dz[i, :] * P_zw[:, w]
                total_prob = topic_probs.sum()
                if total_prob > 0:
                    P_z_given_d_w[i, w, :] = topic_probs / total_prob
                else:
                    P_z_given_d_w[i, w, :] = 0.0

        # M-step: Update P_zw
        for z in range(n_topics):
            for w in range(n_words):
                P_zw[z, w] = np.sum(doc_word_matrix[:, w] * P_z_given_d_w[:, w, z])
            row_sum = P_zw[z, :].sum()
            if row_sum > 0:
                P_zw[z, :] /= row_sum

        # M-step: Update P_dz
        for i in range(n_docs):
            for z in range(n_topics):
                P_dz[i, z] = np.sum(doc_word_matrix[i, :] * P_z_given_d_w[i, :, z])
            row_sum = P_dz[i, :].sum()
            if row_sum > 0:
                P_dz[i, :] /= row_sum

        # Compute log-likelihood for convergence checking
        likelihood = 0.0
        for i in range(n_docs):
            for w in range(n_words):
                prob_dw = (P_dz[i, :] * P_zw[:, w]).sum()
                count_dw = doc_word_matrix[i, w]
                if prob_dw > 0:
                    likelihood += count_dw * np.log(prob_dw)

        if iteration > 0 and abs(likelihood - prev_likelihood) < tol:
            print(f"Early stopping at iteration {iteration+1}")
            break

        prev_likelihood = likelihood
        print(f"Iteration {iteration+1}, log-likelihood: {likelihood:.3f}")

    return P_dz, P_zw


def run_plsi(
    matrix_df: pd.DataFrame,
    vocabulary: list[str],
    n_topics: int,
    max_iter: int = 50,
    tol: float = 1e-5,
    matrix_name: str = "Count Vectors",
    verbose: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Wrapper to run PLSI on a given DataFrame, then optionally print detailed topic/document info.

    Parameters
    ----------
    matrix_df : pd.DataFrame
        Rows = documents, Columns = words.
    vocabulary : list[str]
        List of words aligned with matrix_df columns.
    n_topics : int
        Number of topics to discover.
    max_iter : int
        Maximum EM iterations.
    tol : float
        Convergence threshold.
    matrix_name : str
        Descriptor for logging (e.g., "Count Vectors" or "TF-IDF").
    verbose : bool
        Whether to print additional details.

    Returns
    -------
    P_dz : np.ndarray
        Document-topic distribution (normalized).
    P_zw : np.ndarray
        Topic-word distribution (normalized).
    """
    doc_word_matrix = matrix_df.values
    n_docs, n_words = doc_word_matrix.shape

    print(f"\n--- Running PLSI on {matrix_name} ---")
    print(f"Data: {n_docs} documents x {n_words} words | Topics: {n_topics}")

    P_dz, P_zw = plsi(doc_word_matrix, n_topics, max_iter=max_iter, tol=tol, verbose=verbose)
    return P_dz, P_zw


def main():
    parser = argparse.ArgumentParser(description="Run PLSI on count or TF-IDF matrices.")
    parser.add_argument("--topics", type=int, default=10,
                        help="Number of topics to discover (default=10).")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose printing (default=False).")
    parser.add_argument("--output_dir", type=str, default="out/plsi_vectors",
                        help="Path to the output directory (default=out/plsi_vectors).")
    parser.add_argument("--max_iter", type=int, default=50,
                        help="Maximum number of EM iterations (default=50).")
    parser.add_argument("--tol", type=float, default=1e-5,
                        help="Convergence threshold (default=1e-5).")
    parser.add_argument("--pct_docs", type=float, default=100,
                        help="Percentage of documents to use from CSV files (0-100, default=100).")

    args = parser.parse_args()

    verbose = args.verbose
    n_topics = args.topics
    max_iter = args.max_iter
    tol = args.tol

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    count_vectors_df = pd.read_csv("./out/vectors/count_vectors.csv", index_col=0)
    tfidf_df = pd.read_csv("./out/vectors/tfidf.csv", index_col=0)

    # Select a subset of documents based on the provided percentage
    if args.pct_docs < 100:
        n_total = count_vectors_df.shape[0]
        n_select = int(n_total * args.pct_docs / 100)
        count_vectors_df = count_vectors_df.iloc[:n_select]
        tfidf_df = tfidf_df.iloc[:n_select]

    with open("./out/vectors/meta.json", "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    vocabulary = list(count_vectors_df.columns)

    # Example usage: run PLSI on TF-IDF
    P_dz_tfidf, P_zw_tfidf = run_plsi(
        matrix_df=tfidf_df,
        vocabulary=vocabulary,
        n_topics=n_topics,
        max_iter=max_iter,
        tol=tol,
        matrix_name="TF-IDF",
        verbose=verbose
    )

    # Save the normalized distributions P_dz and P_zw to CSV
    dz_filename = f"PLSI_P_dz_{n_topics}topics_{max_iter}iter.csv"
    zw_filename = f"PLSI_P_zw_{n_topics}topics_{max_iter}iter.csv"

    pd.DataFrame(P_dz_tfidf).to_csv(os.path.join(args.output_dir, dz_filename), index=False)
    pd.DataFrame(P_zw_tfidf).to_csv(os.path.join(args.output_dir, zw_filename), index=False)

    # Uncomment the following code to also run on Count Vectors
    # P_dz_count, P_zw_count = run_plsi(
    #     matrix_df=count_vectors_df,
    #     vocabulary=vocabulary,
    #     n_topics=n_topics,
    #     max_iter=max_iter,
    #     tol=tol,
    #     matrix_name="Count Vectors",
    #     verbose=verbose
    # )
    # pd.DataFrame(P_dz_count).to_csv(os.path.join(args.output_dir, f"PLSI_P_dz_{n_topics}topics_{max_iter}iter_count.csv"), index=False)
    # pd.DataFrame(P_zw_count).to_csv(os.path.join(args.output_dir, f"PLSI_P_zw_{n_topics}topics_{max_iter}iter_count.csv"), index=False)


if __name__ == "__main__":
    main()
