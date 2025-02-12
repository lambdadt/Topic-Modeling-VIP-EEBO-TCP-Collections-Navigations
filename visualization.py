###############################################################################
# REFERENCE:
# Visualization of PLSI's P_dz and P_zw matrices with file output and interactive mode
# Source: 
# Author: 
###############################################################################

import argparse
import json
import numpy as np
import pandas as pd
import os
import sys
from tqdm import tqdm

def visualize(P_dz_tfidf, P_zw_tfidf, index2filename, index2word, doc_lengths, output_file, top_n_words=20):
    """
    Writes three tables into output_file:
      1) Per-topic document counts (and percentages) based on which topic has the highest P_dz value.
      2) For each topic, top N words in P_zw with highest probability/value, plus the average row value,
         plus doc-count info from Table 1.
      3) For each topic, the top 5 documents with the strongest value for that topic, then recall the
         top words of that topic from step #2, plus doc-count info.
    """
    # Dimensions
    d, z = P_dz_tfidf.shape
    _, w = P_zw_tfidf.shape

    # Convert DataFrames to NumPy arrays if needed
    P_dz = P_dz_tfidf.values  # shape (d, z)
    P_zw = P_zw_tfidf.values  # shape (z, w)

    # ============================================================
    # 1) TABLE: Per-topic Doc Counts (Number + Percentage), sorted
    # ============================================================
    doc_topic_assignments = P_dz.argmax(axis=1)  # each doc picks the topic with the highest value
    topic_counts = np.bincount(doc_topic_assignments, minlength=z)

    # Sort topics by descending document counts
    topic_count_pairs = sorted(enumerate(topic_counts), key=lambda x: x[1], reverse=True)

    print("\n" + "#" * 60, file=output_file)
    print("#              TABLE 1: DOC COUNT PER TOPIC             #", file=output_file)
    print("#" * 60 + "\n", file=output_file)
    
    header = f"{'Topic':>10}  |  {'Count':>10}  |  {'Percent':>10}"
    sep = "-" * len(header)
    print(header, file=output_file)
    print(sep, file=output_file)

    for topic_idx, count in topic_count_pairs:
        percent = (count / d) * 100 if d else 0
        print(f"{topic_idx:>10}  |  {count:>10}  |  {percent:>9.2f}%", file=output_file)

    # =================================================
    # 2) TABLE: Top N Words for Each Topic in P_zw
    # =================================================
    print("\n" + "#" * 60, file=output_file)
    print("#   TABLE 2: TOP WORDS PER TOPIC (WITH AVERAGE VALUE)   #", file=output_file)
    print("#" * 60 + "\n", file=output_file)

    row_sums = P_zw.sum(axis=1)
    row_means = P_zw.mean(axis=1)

    for topic_idx in range(z):
        row = P_zw[topic_idx, :]
        doc_count = topic_counts[topic_idx]
        doc_pct   = (doc_count / d) * 100 if d else 0

        top_indices = row.argsort()[::-1][:top_n_words]
        
        print(f"######## Topic {topic_idx} ########", file=output_file)
        print(f"Documents: {doc_count} ({doc_pct:.2f}%)  |  Average row value: {row_means[topic_idx]:.6f}", file=output_file)

        header_topic = f"{'Rank':>4} | {'Word':<15} | {'Value':>10} | {'Pct of Row':>10}"
        sep_topic = "-" * len(header_topic)
        print(header_topic, file=output_file)
        print(sep_topic, file=output_file)
        
        for rank, word_idx in enumerate(top_indices, start=1):
            word_value = row[word_idx]
            pct = (word_value / row_sums[topic_idx] * 100) if row_sums[topic_idx] != 0 else 0.0
            word_str = index2word[word_idx]
            print(f"{rank:>4} | {word_str:<15} | {word_value:>10.6f} | {pct:>9.2f}%", file=output_file)
        
        print("", file=output_file)

    # ==================================================================
    # 3) TABLE: Top 5 Documents for Each Topic & Their Top Topic Words
    # ==================================================================
    print("\n" + "#" * 60, file=output_file)
    print("# TABLE 3: TOP 5 DOCS PER TOPIC & THEIR TOPIC'S BEST WORDS #", file=output_file)
    print("#" * 60 + "\n", file=output_file)

    for topic_idx in range(z):
        doc_count = topic_counts[topic_idx]
        doc_pct   = (doc_count / d) * 100 if d else 0

        topic_col = P_dz[:, topic_idx]
        top_doc_indices = topic_col.argsort()[::-1][:5]

        row = P_zw[topic_idx, :]
        top_word_indices = row.argsort()[::-1][:top_n_words]

        print(f"######## Topic {topic_idx} ########", file=output_file)
        print(f"Documents: {doc_count} ({doc_pct:.2f}%)", file=output_file)

        header_docs = f"{'Rank':>4} | {'DocID':>8} | {'Filename':<60} | {'Length':>8} | {'P_dz':>10}"
        sep_docs = "-" * len(header_docs)
        print(header_docs, file=output_file)
        print(sep_docs, file=output_file)
        
        for rank, doc_id in enumerate(top_doc_indices, start=1):
            doc_val = topic_col[doc_id]
            fname   = index2filename.get(doc_id, f"Doc_{doc_id}")
            dlen    = doc_lengths.get(doc_id, -1)
            print(f"{rank:>4} | {doc_id:>8} | {fname:<60} | {dlen:>8} | {doc_val:>10.6f}", file=output_file)

        print("", file=output_file)
        header_words = f"{'Rank':>4} | {'Word':<15} | {'Value':>10}"
        sep_words = "-" * len(header_words)
        print(header_words, file=output_file)
        print(sep_words, file=output_file)
        
        for rank, w_idx in enumerate(top_word_indices[:5], start=1):
            val = row[w_idx]
            w_str = index2word[w_idx]
            print(f"{rank:>4} | {w_str:<15} | {val:>10.6f}", file=output_file)
        
        print("", file=output_file)


def interactive_loop(P_dz_tfidf, P_zw_tfidf, index2filename, index2word, doc_lengths, top_n_words=20):
    """
    Runs an infinite loop allowing the user to type in a filename. For the chosen document,
    the function displays:
      - The probability distribution across topics (from P_dz).
      - The best topic (highest probability).
      - A table of top words for that topic (similar to Table 2).
    """
    P_dz = P_dz_tfidf.values
    P_zw = P_zw_tfidf.values
    d, z = P_dz.shape

    # Precompute per-document topic assignments and topic counts for convenience
    doc_topic_assignments = P_dz.argmax(axis=1)
    topic_counts = np.bincount(doc_topic_assignments, minlength=z)

    print("\nEntering interactive mode. Type 'exit' to quit.")
    while True:
        user_input = input("\nEnter filename: ").strip()
        if user_input.lower() in ['exit', 'quit']:
            break

        # Try to match the provided filename with one in index2filename (using base name comparison)
        matched_doc_id = None
        base_user = os.path.basename(user_input)
        for doc_id, fname in index2filename.items():
            if os.path.basename(fname) == base_user:
                matched_doc_id = doc_id
                break

        if matched_doc_id is None:
            print(f"File '{user_input}' not found in dataset.")
            continue

        # Display probability distribution for the document
        probabilities = P_dz[matched_doc_id]
        print("\n" + "#" * 60)
        print(f"Topic probabilities for file '{user_input}' (DocID: {matched_doc_id}):")
        print("#" * 60)
        for topic_idx, prob in enumerate(probabilities):
            print(f"Topic {topic_idx}: {prob:.6f}")

        # Identify the best topic for this document
        best_topic = int(np.argmax(probabilities))
        print("\n" + "#" * 60)
        print(f"Best Topic: {best_topic} with probability {probabilities[best_topic]:.6f}")
        print("#" * 60 + "\n")

        # Display a table for the best topic (similar to Table 2)
        row = P_zw[best_topic, :]
        row_sum = row.sum()
        row_mean = row.mean()
        doc_count = topic_counts[best_topic]

        print(f"######## Topic {best_topic} ########")
        print(f"Documents: {doc_count}  |  Average row value: {row_mean:.6f}")
        header = f"{'Rank':>4} | {'Word':<15} | {'Value':>10} | {'Pct of Row':>10}"
        print(header)
        print("-" * len(header))
        top_indices = row.argsort()[::-1][:top_n_words]
        for rank, word_idx in enumerate(top_indices, start=1):
            word_value = row[word_idx]
            pct = (word_value / row_sum * 100) if row_sum != 0 else 0.0
            word_str = index2word[word_idx]
            print(f"{rank:>4} | {word_str:<15} | {word_value:>10.6f} | {pct:>9.2f}%")
        print("\n")


def main():
    parser = argparse.ArgumentParser(description="Visualization of PLSI's P_dz and P_zw matrices")
    parser.add_argument("--input_dir", type=str, default="vectors_in_csv/plsi_vectors",
                        help="Path to the input directory (default=vectors_in_csv/plsi_vectors).")
    parser.add_argument("--dz_filename", type=str,
                        help="Path to the input plsi_P_dz matrix (default=None).")
    parser.add_argument("--zw_filename", type=str,
                        help="Path to the input plsi_P_zw matrix (default=None).")
    parser.add_argument("--output_dir", type=str, default="out/visualizations/plsi",
                        help="Path to the output directory (default=out/visualizations/plsi).")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose printing (default=False).")
    
    args = parser.parse_args()
    verbose = args.verbose

    os.makedirs(args.output_dir, exist_ok=True)

    # Construct full file paths
    dz_path = os.path.join(args.input_dir, args.dz_filename)
    zw_path = os.path.join(args.input_dir, args.zw_filename)

    # Read CSV files
    P_dz_tfidf = pd.read_csv(dz_path)
    P_zw_tfidf = pd.read_csv(zw_path)

    # Create lookup tables
    index2filename = pd.read_csv("./out/vectors/count_vectors.csv").iloc[:, 0].to_dict() # use threshold-5/ for files with larger vocab size
    vocab_list = list(pd.read_csv("./out/vectors/tfidf.csv", index_col=0).columns) # use threshold-5/ for files with larger vocab size
    index2word = {i: word for i, word in enumerate(vocab_list)}

    # Precompute document lengths
    doc_lengths = {}
    for doc_id, fname in index2filename.items():
        if os.path.isfile(fname):
            try:
                with open(fname, 'r', encoding='utf-8', errors='replace') as f:
                    text = f.read()
                doc_lengths[doc_id] = len(text)
            except Exception:
                doc_lengths[doc_id] = -1
        else:
            doc_lengths[doc_id] = -1

    # Create output filename based on the first input file (dz_filename)
    base_name = os.path.splitext(os.path.basename(args.dz_filename))[0]
    output_filename = os.path.join(args.output_dir, base_name + "_visualization.txt")

    with open(output_filename, "w", encoding="utf-8") as out_file:
        visualize(P_dz_tfidf, P_zw_tfidf, index2filename, index2word, doc_lengths, out_file)

    print(f"Visualization output has been written to: {output_filename}")

    # Enter interactive mode
    interactive_loop(P_dz_tfidf, P_zw_tfidf, index2filename, index2word, doc_lengths)


if __name__ == "__main__":
    main()
