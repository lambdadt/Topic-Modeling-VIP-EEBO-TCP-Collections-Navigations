###############################################################################
# REFERENCE:
# Probabilistic Latent Semantic Analysis
# Source: https://en.wikipedia.org/wiki/Probabilistic_latent_semantic_analysis
# Author: Taehwan Park
###############################################################################

import argparse
import json
import os
import sys
from typing import Optional
import contextlib
import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb
from dotenv import load_dotenv


def plsi(
    doc_word_matrix: np.ndarray,
    n_topics: int,
    max_iter: int = 50,
    tol: float = 1e-5,
    verbose: bool = False,
    count_matrix: np.ndarray=None,
    use_wandb: bool=False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Runs the PLSI algorithm using a vectorized EM approach.

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
    count_matrix: np.ndarray
        A 2D array of shape (n_docs, n_words) containing counts. This is used for log-likelihood calculation.
        If not provided, it will by substituted by doc_word_matrix.
    use_wandb: bool
        Will log to Weights & Biases

    Returns
    -------
    P_dz : np.ndarray
        Document-topic distribution (normalized), shape (n_docs, n_topics).
    P_zw : np.ndarray
        Topic-word distribution (normalized), shape (n_topics, n_words).
    """
    n_docs, n_words = doc_word_matrix.shape
    np.random.seed(0)
    if count_matrix is None:
        count_matrix = doc_word_matrix

    # Random initialization for document-topic and topic-word distributions
    P_dz = np.random.rand(n_docs, n_topics)  # P(z|d)
    P_zw = np.random.rand(n_topics, n_words) # P(w|z)

    # Normalize the initial distributions
    P_dz /= P_dz.sum(axis=1, keepdims=True)
    P_zw /= P_zw.sum(axis=1, keepdims=True)

    prev_likelihood = 0.0

    if use_wandb:
        wandb.define_metric('train/iteration')
        wandb.define_metric('train/*', step_metric='train/iteration')

    for iteration in tqdm(range(max_iter), desc="EM iterations"):
        # ---- E-step: Compute P(z|d,w) ----
        # Instead of looping over docs and words, use broadcasting:
        #   P_dz has shape (n_docs, n_topics) and P_zw has shape (n_topics, n_words).
        # We first transpose P_zw to (n_words, n_topics) and then multiply.
        temp = P_dz[:, None, :] * P_zw.T[None, :, :]  # shape: (n_docs, n_words, n_topics)
        total_prob = np.sum(temp, axis=2, keepdims=True)  # shape: (n_docs, n_words, 1)
        # Avoid division by zero by using the 'where' parameter.
        P_z_given_d_w = np.divide(temp, total_prob, out=np.zeros_like(temp), where=total_prob != 0)

        # ---- M-step: Update P_zw ----
        # For each topic z and word w:
        #   P_zw[z, w] = sum_d (doc_word_matrix[d, w] * P_z_given_d_w[d, w, z])
        weighted = doc_word_matrix[:, :, None] * P_z_given_d_w  # shape: (n_docs, n_words, n_topics)
        # Sum over documents (axis=0) gives shape (n_words, n_topics); then transpose to (n_topics, n_words)
        P_zw_new = weighted.sum(axis=0).T
        # Normalize each topic (row) so that they sum to 1
        row_sum = P_zw_new.sum(axis=1, keepdims=True)
        P_zw = np.divide(P_zw_new, row_sum, out=np.zeros_like(P_zw_new), where=row_sum != 0)

        # ---- M-step: Update P_dz ----
        # For each document d and topic z:
        #   P_dz[d, z] = sum_w (doc_word_matrix[d, w] * P_z_given_d_w[d, w, z])
        weighted_dz = doc_word_matrix[:, :, None] * P_z_given_d_w  # shape: (n_docs, n_words, n_topics)
        P_dz_new = weighted_dz.sum(axis=1)  # Sum over words (axis=1) results in shape (n_docs, n_topics)
        # Normalize each document (row) along the topic axes so that they sum to 1
        row_sum_dz = P_dz_new.sum(axis=1, keepdims=True)
        P_dz = np.divide(P_dz_new, row_sum_dz, out=np.zeros_like(P_dz_new), where=row_sum_dz != 0)

        # ---- Log-Likelihood Calculation ----
        # For each document d and word w, the probability is:
        #   prob_dw = sum_z (P_dz[d, z] * P_zw[z, w])
        prob_dw = np.dot(P_dz, P_zw)  # shape: (n_docs, n_words)
        # Compute log-likelihood only where prob_dw is non-zero to avoid log(0)
        mask = prob_dw > 0
        likelihood = np.sum(count_matrix[mask] * np.log(prob_dw[mask]))
        if use_wandb:
            wandb.log({'train/iteration': iteration,
                       'train/loglikelihood': likelihood})

        # Check for convergence
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
    verbose: bool = False,
    count_matrix: pd.DataFrame=None,
    wandb_project: Optional[str]=None,
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
    count_matrix: pd.DataFrame
        Rows = documents, Columns = words. Values must be number of word occurrences.
    wandb_project: Optional[str]
        Will use wandb to log training progress with given project name if set.

    Returns
    -------
    P_dz : np.ndarray
        Document-topic distribution (normalized).
    P_zw : np.ndarray
        Topic-word distribution (normalized).
    """
    doc_word_matrix = matrix_df.values
    n_docs, n_words = doc_word_matrix.shape
    if count_matrix is not None:
        count_matrix = count_matrix.values

    print(f"\n--- Running PLSI on {matrix_name} ---")
    print(f"Data: {n_docs} documents x {n_words} words | Topics: {n_topics}")
    if wandb_project:
        run_name = "{} T{} V{}".format(datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"), n_topics, len(vocabulary))
        print(f"Logging to Weights & Biases (Project: {wandb_project}, Run: {run_name})")
        cm = wandb.init(project=wandb_project, name=run_name,
            config={
                'matrix': matrix_name,
                'vocabulary_size': len(vocabulary),
                'n_topics': n_topics,
                'max_iter': max_iter,
        })
    else:
        cm = contextlib.nullcontext()
    with cm:
        P_dz, P_zw = plsi(doc_word_matrix, n_topics, max_iter=max_iter, tol=tol, verbose=verbose, count_matrix=count_matrix,
                          use_wandb=bool(wandb_project))

    return P_dz, P_zw


def main():
    parser = argparse.ArgumentParser(description="Run PLSI on count or TF-IDF matrices.")
    parser.add_argument("--topics", type=int, default=10,
                        help="Number of topics to discover (default=10).")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose printing (default=False).")
    parser.add_argument("--input_dir", type=str, default="out/vectors",
                        help="Path to the input directory (default=out/vectors).")
    parser.add_argument("--output_dir", type=str, default="vectors_in_csv/plsi_vectors",
                        help="Path to the output directory (default=vectors_in_csv/plsi_vectors).")
    parser.add_argument("--max_iter", type=int, default=50,
                        help="Maximum number of EM iterations (default=50).")
    parser.add_argument("--tol", type=float, default=1e-5,
                        help="Convergence threshold (default=1e-5).")
    parser.add_argument("--pct_docs", type=float, default=100,
                        help="Percentage of documents to use from CSV files (0-100, default=100).")
    parser.add_argument("--matrix_type", type=str, choices=["tfidf", "count", "tfidf_reweighted_count_vectors"], default="tfidf",
                        help="Matrix type to use (default=tfidf).")
    parser.add_argument("--use_wandb", action='store_true',
                        help="Log training progress to Weights & Biases.")
    parser.add_argument("--wandb_project", type=str, default="VIP-Topic-Modelling-pLSI",
                        help="wandb project name that will be applied if --use_wandb is set.")
    
    args = parser.parse_args()

    load_dotenv()

    verbose = args.verbose
    n_topics = args.topics
    max_iter = args.max_iter
    tol = args.tol

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    count_vectors_path = os.path.join(args.input_dir,
        ("count_vectors" if args.matrix_type == 'count' else 'tfidf_reweighted_count_vectors') + ".csv")
    tfidf_path = os.path.join(args.input_dir, "tfidf.csv")

    print("Reading count vectors from '{}'...".format(count_vectors_path))
    count_vectors_df = pd.read_csv(count_vectors_path, index_col=0)
    print("Reading TFIDF from '{}'...".format(tfidf_path))
    tfidf_df = pd.read_csv(tfidf_path, index_col=0)

    # Select a subset of documents based on the provided percentage
    if args.pct_docs < 100:
        n_total = count_vectors_df.shape[0]
        n_select = int(n_total * args.pct_docs / 100)
        count_vectors_df = count_vectors_df.iloc[:n_select]
        tfidf_df = tfidf_df.iloc[:n_select]

    vocabulary = list(count_vectors_df.columns)

    if args.use_wandb:
        print("Using Weights and Biases.")
        wandb.login()

    if args.matrix_type == "tfidf":
        # Run PLSI on TF-IDF
        P_dz, P_zw = run_plsi(
            matrix_df=tfidf_df,
            vocabulary=vocabulary,
            n_topics=n_topics,
            max_iter=max_iter,
            tol=tol,
            matrix_name="TF-IDF",
            verbose=verbose,
            count_matrix=count_vectors_df,
            wandb_project=args.wandb_project if args.use_wandb else None,
        )
    else:
        # Run PLSI on Count Vectors
        P_dz, P_zw = run_plsi(
            matrix_df=count_vectors_df,
            vocabulary=vocabulary,
            n_topics=n_topics,
            max_iter=max_iter,
            tol=tol,
            matrix_name="Count Vectors" if args.matrix_type == 'count' else 'Count Vectors (Reweighted by TF-IDF)',
            verbose=verbose,
            count_matrix=count_vectors_df,
            wandb_project=args.wandb_project if args.use_wandb else None,
        )

    suffix = "_" + args.matrix_type

    # Save the normalized distributions P_dz and P_zw to CSV
    dz_filename = f"PLSI_P_dz_{n_topics}topics_{max_iter}iter{suffix}.csv"
    zw_filename = f"PLSI_P_zw_{n_topics}topics_{max_iter}iter{suffix}.csv"

    pd.DataFrame(P_dz).to_csv(os.path.join(args.output_dir, dz_filename), index=False)
    pd.DataFrame(P_zw).to_csv(os.path.join(args.output_dir, zw_filename), index=False)


if __name__ == "__main__":
    main()