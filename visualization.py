import argparse
import json
import numpy as np
import pandas as pd
import os
import sys
import re
from tqdm import tqdm
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt

# --- Function Definitions ---

def compute_high_df_words(count_vectors_df, threshold=0.8):
    df = (count_vectors_df > 0).sum(axis=0) / count_vectors_df.shape[0]
    high_df_words = df[df > threshold].index
    return {i for i, word in enumerate(count_vectors_df.columns) if word in high_df_words}

def generate_doc_counts_table(P_dz_tfidf, doc_lengths, topic_counts, output_file):
    P_dz = P_dz_tfidf.values
    d, z = P_dz.shape
    doc_topic_assignments = P_dz.argmax(axis=1)
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

        d_total = P_dz.shape[0]
        for topic_idx, count in topic_count_pairs:
            percent = (count / d_total) * 100 if d_total else 0
            avg_len = avg_doc_lengths[topic_idx]
            print(f"{topic_idx:>10}  |  {count:>10}  |  {percent:>9.2f}%  |  {avg_len:>12.2f}", file=f)

def generate_cluster_distribution_bar(P_dz_tfidf, output_file):
    P_dz = P_dz_tfidf.values
    z = P_dz.shape[1]
    doc_topic_assignments = P_dz.argmax(axis=1)
    topic_counts = np.bincount(doc_topic_assignments, minlength=z)
    max_count = max(topic_counts)
    bar_scale = 50

    with open(output_file, 'w', encoding='utf-8') as f:
        print("\n" + "#" * 80, file=f)
        print("#           CLUSTER DISTRIBUTION BAR CHART           #", file=f)
        print("#" * 80 + "\n", file=f)

        for topic_idx in range(z):
            count = topic_counts[topic_idx]
            bar_length = int((count / max_count) * bar_scale) if max_count > 0 else 0
            bar = '*' * bar_length
            print(f"Topic {topic_idx:>2}: {bar} ({count})", file=f)

def generate_top_words_table(P_zw_tfidf, index2word, high_df_indices, topic_counts, output_file, top_n_words=30):
    P_zw = P_zw_tfidf.values
    z, w = P_zw.shape

    candidates = {}
    excluded = {}
    for topic_idx in range(z):
        row = P_zw[topic_idx, :]
        sorted_indices = row.argsort()[::-1]
        top_words = []
        excluded_words = []
        for idx in sorted_indices:
            if idx not in high_df_indices:
                top_words.append(idx)
                if len(top_words) >= top_n_words:
                    break
            else:
                excluded_words.append(idx)
        candidates[topic_idx] = top_words
        excluded[topic_idx] = excluded_words

    with open(output_file, 'w', encoding='utf-8') as f:
        for topic_idx in range(z):
            row = P_zw[topic_idx, :]
            top_word_idx = row.argsort()[::-1][0]
            top_word = index2word[top_word_idx]
            top_prob = row[top_word_idx]
            doc_count = topic_counts[topic_idx]

            print(f"######## Topic {topic_idx} ########", file=f)
            print(f"Documents: {doc_count}", file=f)
            print(f"Top Word: {top_word} ({top_prob:.6f})", file=f)

            header = f"{'Rank':>4} | {'Word':<15} | {'Value':>10} | {'Pct of Row':>10}"
            sep = "-" * len(header)
            print(header, file=f)
            print(sep, file=f)

            for rank, word_idx in enumerate(candidates[topic_idx], start=1):
                word_value = row[word_idx]
                total_row_sum = row.sum()
                pct = (word_value / total_row_sum) * 100 if total_row_sum != 0 else 0.0
                word_str = index2word[word_idx]
                print(f"{rank:>4} | {word_str:<15} | {word_value:>10.6f} | {pct:>9.2f}%", file=f)

            print("\nTopic-Specific Excluded Words:", file=f)
            excluded_str = [index2word[w_idx] for w_idx in excluded[topic_idx]]
            print(", ".join(excluded_str) if excluded_str else "None", file=f)
            print("\n", file=f)

def generate_top_docs_table(P_dz_tfidf, index2filename, doc_lengths, output_file, top_n_docs=5):
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

def generate_v_measure(P_zw_tfidf, output_file):
    P_zw = P_zw_tfidf.values
    z, w = P_zw.shape  # z topics, w words

    # Predicted clustering: assign words to topics where sum of P(w|z) reaches 0.95
    predicted_assignments = [set() for _ in range(z)]
    for topic_idx in range(z):
        probs = P_zw[topic_idx, :]
        sorted_indices = np.argsort(probs)[::-1]
        cum_prob = 0.0
        for word_idx in sorted_indices:
            cum_prob += probs[word_idx]
            predicted_assignments[topic_idx].add(word_idx)
            if cum_prob >= 0.95:
                break

    # Ground truth: each word assigned to its most probable topic
    true_assignments = P_zw.argmax(axis=0)

    # Contingency table
    contingency = np.zeros((z, z))
    for word_idx in range(w):
        true_topic = true_assignments[word_idx]
        for pred_topic in range(z):
            if word_idx in predicted_assignments[pred_topic]:
                contingency[true_topic, pred_topic] += 1

    # Compute sizes
    n = w
    true_sizes = np.bincount(true_assignments, minlength=z)
    pred_sizes = np.array([len(predicted_assignments[j]) for j in range(z)])
    total_assignments = contingency.sum()

    with open(output_file, 'w', encoding='utf-8') as f:
        print("\n" + "#" * 80, file=f)
        print("#              V-MEASURE SCORE              #", file=f)
        print("#" * 80 + "\n", file=f)
        
        print(f"Number of topics (z): {z}", file=f)
        print(f"Number of words (w): {w}", file=f)
        print("\nPredicted cluster sizes (|C_j|):", file=f)
        for j in range(z):
            print(f"Topic {j}: {pred_sizes[j]} words", file=f)
        print(f"\nTrue cluster sizes (|T_k|):", file=f)
        for k in range(z):
            print(f"Topic {k}: {true_sizes[k]} words", file=f)
        print(f"\nTotal assignments in contingency table: {total_assignments}", file=f)
        
        print("\nContingency Table (True vs Predicted):", file=f)
        header = "    " + " ".join([f"{j:>5}" for j in range(z)])
        print(f"True \\ Pred | {header}", file=f)
        for k in range(z):
            row = " ".join([f"{contingency[k,j]:>5.0f}" for j in range(z)])
            print(f"{k:>2}         | {row}", file=f)

        # H(T)
        H_T = 0
        for k in range(z):
            p = true_sizes[k] / n
            if p > 0:
                H_T -= p * np.log2(p)
        print(f"\nH(T): {H_T:.4f}", file=f)

        # H(C) - Normalize over total assignments
        H_C = 0
        for j in range(z):
            p = pred_sizes[j] / total_assignments if total_assignments > 0 else 0
            if p > 0:
                H_C -= p * np.log2(p)
        print(f"H(C): {H_C:.4f}", file=f)

        # H(T|C) - Use joint probability over total assignments
        H_T_given_C = 0
        for j in range(z):
            if pred_sizes[j] > 0:
                for k in range(z):
                    n_kj = contingency[k, j]
                    if n_kj > 0:
                        p_kj = n_kj / total_assignments  # Joint prob
                        p_k_given_j = n_kj / pred_sizes[j]  # Conditional prob
                        H_T_given_C -= p_kj * np.log2(p_k_given_j)
        print(f"H(T|C): {H_T_given_C:.4f}", file=f)

        # H(C|T)
        H_C_given_T = 0
        for k in range(z):
            if true_sizes[k] > 0:
                for j in range(z):
                    n_kj = contingency[k, j]
                    if n_kj > 0:
                        p_kj = n_kj / total_assignments
                        p_j_given_k = n_kj / true_sizes[k]
                        H_C_given_T -= p_kj * np.log2(p_j_given_k)
        print(f"H(C|T): {H_C_given_T:.4f}", file=f)

        # Homogeneity and Completeness with bounds check
        h = 1 - (H_T_given_C / H_T) if H_T > 0 and H_T_given_C <= H_T else max(0, 1 - (H_T_given_C / H_T))
        c = 1 - (H_C_given_T / H_C) if H_C > 0 and H_C_given_T <= H_C else max(0, 1 - (H_C_given_T / H_C))
        v_measure = (2 * h * c) / (h + c) if (h + c) > 0 else 0

        print(f"\nHomogeneity: {h:.4f}", file=f)
        print(f"Completeness: {c:.4f}", file=f)
        print(f"V-Measure: {v_measure:.4f}", file=f)

def generate_topic_histograms_combined(P_dz_tfidf, output_file, bins=9):
    """
    Create a single image with subplots showing a histogram for each topic.
    Modifications:
      - We trim the histogram range to (0.1, 1.0).
      - We fix the y-axis from 0 to 100.
      - We place x-ticks at 0.1 increments, from 0.1 to 1.0.
      - We add borders to the bars and horizontal grid lines for readability.
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    P_dz = P_dz_tfidf.values
    doc_count, n_topics = P_dz.shape
    
    # We'll assume the user wants 5 columns (since topics is a multiple of 10).
    ncols = 5
    nrows = n_topics // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = axes.flatten()
    
    for topic_idx in range(n_topics):
        ax = axes[topic_idx]
        topic_proportions = P_dz[:, topic_idx]
        
        # Only show the range from 0.1 to 1.0 with borders (edgecolor) and some padding (rwidth)
        ax.hist(topic_proportions, bins=bins, range=(0.1, 1.0), edgecolor='black', rwidth=0.8)
        
        # Set fixed x-axis range
        ax.set_xlim(0.1, 1.0)
        # Set fixed y-axis range to 0..100
        ax.set_ylim(0, 100)
        
        # Set x-ticks at 0.1 intervals
        ax.set_xticks([i / 10 for i in range(1, 11)])  # 0.1 to 1.0
        
        # Add horizontal grid lines
        ax.grid(axis='y', color='gray', linestyle='--', linewidth=0.5)
        
        ax.set_title(f"Topic {topic_idx}")
        ax.set_xlabel("Proportion")
        ax.set_ylabel("Documents")
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def generate_neff_histogram(P_dz_tfidf, output_file):
    """
    Compute the effective number of topics (N_eff) for each document and plot a histogram.
    N_eff is computed as:
         N_eff = exp( - sum(p_i * log(p_i)) )
    where p_i are the topic probabilities for a given document.
    
    The histogram is binned into intervals such that each bin represents values within
    (i - 0.5, i + 0.5] for integer i. For example, both 1.7 and 2.3 will fall into the bin centered at 2.
    """
    P_dz = P_dz_tfidf.values
    # Compute entropy for each row, handling 0 probabilities
    entropy = -np.sum(np.where(P_dz > 0, P_dz * np.log(P_dz), 0), axis=1)
    neff = np.exp(entropy)
    
    n_topics = P_dz.shape[1]
    # Define bin edges so that each bin is centered at an integer
    bin_edges = np.arange(0.5, n_topics + 1, 1)  # edges: 0.5, 1.5, 2.5, ..., n_topics + 0.5
    
    plt.figure()
    plt.hist(neff, bins=bin_edges, edgecolor='black', rwidth=0.8)
    
    # Set x-ticks at the center of each bin with labels as the integer (1,2,3,...)
    tick_positions = (bin_edges[:-1] + bin_edges[1:]) / 2
    tick_labels = [str(int(x)) for x in tick_positions]
    plt.xticks(tick_positions, tick_labels)
    
    plt.xlabel("Effective Number of Topics (N_eff)")
    plt.ylabel("Frequency")
    plt.title("Histogram of N_eff")
    plt.savefig(output_file)
    plt.close()

def interactive_loop(P_dz_tfidf, P_zw_tfidf, index2filename, index2word, doc_lengths):
    pass

# --- Main Function ---

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
        if args.zw_filename.endswith("tfidf_reweighted_count_vectors.csv"):
            vis_dir += "-weighted"
    else:
        vis_dir = "vis-default"

    output_path = os.path.join(args.output_dir, vis_dir)
    os.makedirs(output_path, exist_ok=True)

    dz_path = os.path.join(args.matrix_dir, args.dz_filename)
    zw_path = os.path.join(args.matrix_dir, args.zw_filename)
    P_dz_tfidf = pd.read_csv(dz_path)
    P_zw_tfidf = pd.read_csv(zw_path)

    count_vectors_df = pd.read_csv(os.path.join(args.data_dir, "count_vectors.csv"), index_col=0)
    high_df_indices = compute_high_df_words(count_vectors_df, threshold=0.8)

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

    P_dz = P_dz_tfidf.values
    doc_topic_assignments = P_dz.argmax(axis=1)
    topic_counts = np.bincount(doc_topic_assignments, minlength=P_dz.shape[1])

    # Write out high DF words
    with open(os.path.join(output_path, "stopwords.txt"), 'w', encoding='utf-8') as f:
        print("\n" + "#" * 80, file=f)
        print("#        HIGH DOCUMENT FREQUENCY WORDS EXCLUDED        #", file=f)
        print("#" * 80 + "\n", file=f)
        print("Excluded High DF Words:", file=f)
        high_df_words_list = sorted([index2word[idx] for idx in high_df_indices])
        print(", ".join(high_df_words_list) if high_df_words_list else "None", file=f)

    # Generate outputs
    generate_doc_counts_table(P_dz_tfidf, doc_lengths, topic_counts, os.path.join(output_path, "doc_counts.txt"))
    generate_cluster_distribution_bar(P_dz_tfidf, os.path.join(output_path, "cluster_bar.txt"))
    generate_top_words_table(P_zw_tfidf, index2word, high_df_indices, topic_counts, os.path.join(output_path, "top_words.txt"))
    generate_top_docs_table(P_dz_tfidf, index2filename, doc_lengths, os.path.join(output_path, "top_docs.txt"))
    generate_similarity_tables(P_dz_tfidf, P_zw_tfidf, os.path.join(output_path, "similarity.txt"))
    generate_v_measure(P_zw_tfidf, os.path.join(output_path, "v_measure.txt"))
    
    # Generate a combined histogram image for all topics (trim <0.1, max y=100)
    combined_hist_file = os.path.join(output_path, "hist_all_topics.png")
    generate_topic_histograms_combined(P_dz_tfidf, combined_hist_file)
    
    # Generate histogram of N_eff values
    neff_hist_file = os.path.join(output_path, "hist_neff.png")
    generate_neff_histogram(P_dz_tfidf, neff_hist_file)

    print(f"Visualization outputs written to: {output_path}")

    if not args.skip_interactive_mode:
        interactive_loop(P_dz_tfidf, P_zw_tfidf, index2filename, index2word, doc_lengths)

if __name__ == "__main__":
    main()
