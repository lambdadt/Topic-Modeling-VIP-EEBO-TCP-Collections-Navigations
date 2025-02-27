###############################################################################
# Visualization of PLSI's P_dz and P_zw matrices with modular file output
# Refactored by: A Professional Programmer - 10+ Years Experience
# Enhanced by: Grok (xAI) - Added excluded words and V-measure proxy
###############################################################################

import argparse
import json
import numpy as np
import pandas as pd
import os
import sys
import re
from tqdm import tqdm
from scipy.spatial.distance import cosine

def generate_doc_counts_table(P_dz_tfidf, doc_lengths, output_file):
    """Generates document counts and average lengths per topic."""
    P_dz = P_dz_tfidf.values
    d, z = P_dz.shape
    doc_topic_assignments = P_dz.argmax(axis=1)
    topic_counts = np.bincount(doc_topic_assignments, minlength=z)
    topic_count_pairs = sorted(enumerate(topic_counts), key=lambda x: x[1], reverse=True)

    avg_doc_lengths = {}
    for topic_idx in range(z):
        topic_docs = [doc_id for doc_id, assign in enumerate(doc_topic_assignments) if assign == topic_idx]
        lengths = [doc_lengths.get(doc_id, 0) for doc_id in topic_docs]
        avg_doc_lengths[topic_idx] = np.mean(lengths) if lengths else 0

    with open(output_file, 'w', encoding='utf-8') as f:
        print("\n" + "#" * 80, file=f)
        print("#        DOC COUNT AND AVG LENGTH PER TOPIC        #", file=f)
        print("#" * 80 + "\n", file=f)
        
        header = f"{'Topic':>10}  |  {'Count':>10}  |  {'Percent':>10}  |  {'Avg Doc Len':>12}"
        sep = "-" * len(header)
        print(header, file=f)
        print(sep, file=f)

        for topic_idx, count in topic_count_pairs:
            percent = (count / d) * 100 if d else 0
            avg_len = avg_doc_lengths[topic_idx]
            print(f"{topic_idx:>10}  |  {count:>10}  |  {percent:>9.2f}%  |  {avg_len:>12.2f}", file=f)

def generate_cluster_distribution_bar(P_dz_tfidf, output_file):
    """Generates a text-based cluster distribution bar chart."""
    P_dz = P_dz_tfidf.values
    z = P_dz.shape[1]
    doc_topic_assignments = P_dz.argmax(axis=1)
    topic_counts = np.bincount(doc_topic_assignments, minlength=z)
    max_count = max(topic_counts)
    bar_scale = 50  # Max bar length in characters

    with open(output_file, 'w', encoding='utf-8') as f:
        print("\n" + "#" * 80, file=f)
        print("#           CLUSTER DISTRIBUTION BAR CHART           #", file=f)
        print("#" * 80 + "\n", file=f)

        for topic_idx in range(z):
            count = topic_counts[topic_idx]
            bar_length = int((count / max_count) * bar_scale) if max_count > 0 else 0
            bar = '*' * bar_length
            print(f"Topic {topic_idx:>2}: {bar} ({count})", file=f)

def generate_top_words_table(P_zw_tfidf, index2word, output_file, top_n_words=50, initial_candidates=200):
    """Generates top distinct words per topic and lists excluded words."""
    P_zw = P_zw_tfidf.values
    z, w = P_zw.shape
    row_sums = P_zw.sum(axis=1)
    row_means = P_zw.mean(axis=1)

    common_stop_words = {'the', 'of', 'and', 'to', 'in', 'that', 'with', 'which', 'was', 'his', 
                         'by', 'their', 'were', 'they', 'he', 'this', 'for', 'at', 'it', 'all'}
    stop_word_indices = {i for i, word in index2word.items() if word.lower() in common_stop_words}

    top_distinct_words = {}
    excluded_words = {}  # Track excluded words per topic
    used_words = set()

    candidates = {}
    for topic_idx in range(z):
        row = P_zw[topic_idx, :]
        sorted_indices = row.argsort()[::-1]
        top_indices = [idx for idx in sorted_indices if idx not in stop_word_indices][:initial_candidates]
        candidates[topic_idx] = top_indices

    for topic_idx in range(z):
        distinct_words = []
        excluded = []  # Collect excluded words
        candidate_indices = candidates[topic_idx]
        for word_idx in candidate_indices:
            if word_idx not in used_words:
                distinct_words.append(word_idx)
                used_words.add(word_idx)
            else:
                excluded.append(word_idx)  # Word is excluded
            if len(distinct_words) >= top_n_words:
                break
        top_distinct_words[topic_idx] = distinct_words
        excluded_words[topic_idx] = excluded

    with open(output_file, 'w', encoding='utf-8') as f:
        for topic_idx in range(z):
            row = P_zw[topic_idx, :]
            print(f"######## Topic {topic_idx} ########", file=f)
            print(f"Average row value: {row_means[topic_idx]:.6f}", file=f)

            header = f"{'Rank':>4} | {'Word':<15} | {'Value':>10} | {'Pct of Row':>10}"
            sep = "-" * len(header)
            print(header, file=f)
            print(sep, file=f)
            
            for rank, word_idx in enumerate(top_distinct_words[topic_idx], start=1):
                word_value = row[word_idx]
                pct = (word_value / row_sums[topic_idx] * 100) if row_sums[topic_idx] != 0 else 0.0
                word_str = index2word[word_idx]
                print(f"{rank:>4} | {word_str:<15} | {word_value:>10.6f} | {pct:>9.2f}%", file=f)

            # Excluded words section
            print("\nExcluded Words:", file=f)
            excluded_str = [index2word[w_idx] for w_idx in excluded_words[topic_idx][:10]]  # Limit to 10
            print(", ".join(excluded_str) if excluded_str else "None", file=f)
            print("\n", file=f)

def generate_top_docs_table(P_dz_tfidf, index2filename, doc_lengths, output_file, top_n_docs=5):
    """Generates top documents per topic."""
    P_dz = P_dz_tfidf.values
    z = P_dz.shape[1]
    with open(output_file, 'w', encoding='utf-8') as f:
        for topic_idx in range(z):
            topic_col = P_dz[:, topic_idx]
            top_doc_indices = topic_col.argsort()[::-1][:top_n_docs]

            print(f"######## Topic {topic_idx} ########", file=f)
            header_docs = f"{'Rank':>4} | {'DocID':>8} | {'Filename':<60} | {'Length':>8} | {'P_dz':>10}"
            sep_docs = "-" * len(header_docs)
            print(header_docs, file=f)
            print(sep_docs, file=f)
            
            for rank, doc_id in enumerate(top_doc_indices, start=1):
                doc_val = topic_col[doc_id]
                fname = index2filename.get(doc_id, f"Doc_{doc_id}")
                dlen = doc_lengths.get(doc_id, -1)
                print(f"{rank:>4} | {doc_id:>8} | {fname:<60} | {dlen:>8} | {doc_val:>10.6f}", file=f)
            print("\n", file=f)

def generate_similarity_tables(P_dz_tfidf, P_zw_tfidf, output_file):
    """Generates cluster similarity tables (word-based and document-based)."""
    P_dz = P_dz_tfidf.values
    P_zw = P_zw_tfidf.values
    z = P_zw.shape[0]

    similarity_matrix = np.zeros((z, z))
    for i in range(z):
        for j in range(z):
            similarity_matrix[i][j] = 1 - cosine(P_zw[i], P_zw[j]) if i != j else 1.0

    correlation_matrix = np.corrcoef(P_dz.T)

    with open(output_file, 'w', encoding='utf-8') as f:
        print("\n" + "#" * 80, file=f)
        print("#              CLUSTER SIMILARITY TABLES              #", file=f)
        print("#" * 80 + "\n", file=f)

        def print_matrix(matrix, title):
            print(f"\n{title}", file=f)
            header = "    " + " ".join([f"{i:>5}" for i in range(z)])
            print(header, file=f)
            for i in range(z):
                row = [f"{matrix[i][j]:.3f}" for j in range(z)]
                print(f"{i:>2} | {' '.join(row)}", file=f)

        print_matrix(similarity_matrix, "Cosine Similarity (Word-Based)")
        print_matrix(correlation_matrix, "Correlation (Document-Based)")

def generate_v_measure(P_dz_tfidf, output_file):
    """Computes a V-measure-like score using topic assignments."""
    P_dz = P_dz_tfidf.values
    d, z = P_dz.shape

    topic_assignments = P_dz.argmax(axis=1)

    homogeneity = []
    completeness = []
    for topic_idx in range(z):
        cluster_docs = np.where(topic_assignments == topic_idx)[0]
        if len(cluster_docs) == 0:
            continue
        homogeneity.append(np.mean(P_dz[cluster_docs, topic_idx]))
        completeness.append(len(cluster_docs) / d)

    h = np.mean(homogeneity) if homogeneity else 0
    c = np.mean(completeness) if completeness else 0
    v_measure = 2 * (h * c) / (h + c) if (h + c) > 0 else 0

    with open(output_file, 'w', encoding='utf-8') as f:
        print("\n" + "#" * 80, file=f)
        print("#              V-MEASURE-LIKE SCORE              #", file=f)
        print("#" * 80 + "\n", file=f)
        print(f"V-Measure (Proxy): {v_measure:.4f}", file=f)
        print(f"Pseudo-Homogeneity: {h:.4f}", file=f)
        print(f"Pseudo-Completeness: {c:.4f}", file=f)

def main():
    parser = argparse.ArgumentParser(description="Visualization of PLSI's P_dz and P_zw matrices")
    parser.add_argument("--matrix_dir", type=str, default="vectors_in_csv/plsi_vectors",
                        help="Path to the directory containing the probability matrices.")
    parser.add_argument("--data_dir", type=str, default="out/vectors",
                        help="Path to the directory containing data files.")
    parser.add_argument("--dz_filename", type=str,
                        help="Filename of the input PLSI_P_dz matrix.")
    parser.add_argument("--zw_filename", type=str,
                        help="Filename of the input PLSI_P_zw matrix.")
    parser.add_argument("--output_dir", type=str, default="out/vocab100k",
                        help="Base path to the output directory.")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose printing.")
    parser.add_argument("--skip_interactive_mode", action="store_true",
                        help="Skip interactive mode.")
    
    args = parser.parse_args()

    match = re.search(r'PLSI_P_zw_(\d+)topics_(\d+)iter', args.zw_filename)
    if match:
        topics = match.group(1)
        iterations = match.group(2)
        vis_dir = f"vis-t{topics}-i{iterations}"
    else:
        vis_dir = "vis-default"

    output_path = os.path.join(args.output_dir, vis_dir)
    os.makedirs(output_path, exist_ok=True)

    dz_path = os.path.join(args.matrix_dir, args.dz_filename)
    zw_path = os.path.join(args.matrix_dir, args.zw_filename)

    P_dz_tfidf = pd.read_csv(dz_path)
    P_zw_tfidf = pd.read_csv(zw_path)

    index2filename = pd.read_csv(os.path.join(args.data_dir, "count_vectors.csv")).iloc[:, 0].to_dict()
    vocab_list = list(pd.read_csv(os.path.join(args.data_dir, "tfidf.csv"), index_col=0).columns)
    index2word = {i: word for i, word in enumerate(vocab_list)}

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

    # Generate outputs
    generate_doc_counts_table(P_dz_tfidf, doc_lengths, os.path.join(output_path, "doc_counts.txt"))
    generate_cluster_distribution_bar(P_dz_tfidf, os.path.join(output_path, "cluster_bar.txt"))
    generate_top_words_table(P_zw_tfidf, index2word, os.path.join(output_path, "top_words.txt"))
    generate_top_docs_table(P_dz_tfidf, index2filename, doc_lengths, os.path.join(output_path, "top_docs.txt"))
    generate_similarity_tables(P_dz_tfidf, P_zw_tfidf, os.path.join(output_path, "similarity.txt"))
    generate_v_measure(P_dz_tfidf, os.path.join(output_path, "v_measure.txt"))

    print(f"Visualization outputs written to: {output_path}")

    # if not args.skip_interactive_mode:
    #     interactive_loop(P_dz_tfidf, P_zw_tfidf, index2filename, index2word, doc_lengths)

if __name__ == "__main__":
    main()