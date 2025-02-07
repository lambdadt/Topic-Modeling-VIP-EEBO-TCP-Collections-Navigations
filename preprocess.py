import numpy as np
import pandas as pd
from tqdm import tqdm
import nltk

from pathlib import Path
import sys
import argparse
import json
import re
import os
import os.path as osp
import math
import collections


def make_vectors():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output_dir', '-o', required=True)
    ap.add_argument('--parsed_texts_dir', default="Navigations_headed_xml/Parsed_texts",
                    help=".txt files in this directory will be searched for recursively.")
    ap.add_argument('--stemmer', default='None', choices=['None', 'Porter', 'Snowball', 'WordNetLemmatizer'],
                    help="If not provided (default), WordNet Lemmatizer will be used.")
    ap.add_argument('--rare_word_threshold', default=0.0, type=float, help="Cull words (after lemmatization) used fewer then x%%")
    ap.add_argument('--vector_dim_limit', default=100_000, type=int, help="Rare words will be culled. -1 means no limit.")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.is_dir():
        i = input("Output directory '{}' already exists; countinue (contents may be overwritten)? [Y|n] ".format(output_dir))
        if i.lower() == 'n':
            print("Aborting.")
            exit(0)
    else:
        output_dir.mkdir(parents=True)

    doc_paths = []
    parsed_texts_dir = Path(args.parsed_texts_dir)
    print("Finding docs (.txt) under '{}'...".format(parsed_texts_dir))
    for root, dirs, files in os.walk(parsed_texts_dir, topdown=False, followlinks=True):
        for fn in files:
            path = Path(root, fn)
            # if path.suffix.lower() == ".txt":
            if path.name.lower().endswith("parsed_text.txt"):
            # if path.name.lower().endswith("footnotes.txt"):
                doc_paths.append(path)
    print("Found {} documents.".format(len(doc_paths)))

    doc_infos = []
    for doc_path in tqdm(doc_paths, desc="Reading docs"):
        with open(doc_path, encoding='utf-8') as f:
            doc_infos.append({
                'path': doc_path,
                'text': f.read(),
            })
    n_docs = len(doc_infos)
    print()

    # Tokenize
    pat = re.compile('(?u)\\b\\w\\w+\\b')
    tokens_all = []
    for d_i in tqdm(doc_infos, desc="Tokenizing docs"):
        tokens = re.findall(pat, d_i['text'].lower())
        d_i['tokens'] = tokens
        tokens_all.extend(tokens)
    vocab = set(tokens_all)
    print("#Total tokens: {}".format(len(tokens_all)))
    print("Vocabulary size: {}".format(len(vocab)))
    print()
    
    # Stem/Lemmatize
    is_stemmer = False
    match args.stemmer:
        case 'Porter':
            print("Using Porter Stemmer.")
            lm = nltk.stemmer.PorterStemmer()
            is_stemmer = True
        case 'Snowball':
            print("Using Snowball Stemmer.")
            lm = nltk.stem.SnowballStemmer('english')
            is_stemmer = True
        case 'WordNetLemmatizer':
            print("Using WordNet Lemmatizer.")
            lm = nltk.wordnet.WordNetLemmatizer()
        case 'None':
            print("No stemmer/lemmatizer used.")
            class IdentityLM:
                def lemmatize(self, w):
                    return w
            lm = IdentityLM()
        case _:
            print("Unknown stemmer/lemmitizer: '{}'. Aborting.".format(args.stemmer))
            exit(1)
    
    tokens_all_lm = []
    word2lm = dict()
    vocab_lm = set()
    for word in tqdm(tokens_all, desc="Lemmatizing/Stemming"):
        if is_stemmer:
            word_lm = lm.stem(word)
        else:
            word_lm = lm.lemmatize(word)
        vocab_lm.add(word_lm)
        word2lm[word] = word_lm
        tokens_all_lm.append(word_lm)
        
    print("Vocab size (lemmatized): {}".format(len(vocab_lm)))

    rare_word_threshold = args.rare_word_threshold / 100.0
    if args.vector_dim_limit > 0 or rare_word_threshold > 0.0:
        n_total_words = len(tokens_all_lm)
        rare_word_threshold_count = round(n_total_words * rare_word_threshold)
        print("Words used less than {:.02f}% (#={}) will be culled from the vocabulary.".format(100*rare_word_threshold, rare_word_threshold_count))
        counter = collections.Counter(tokens_all_lm)
        keep_words = set()
        for word in vocab_lm:
            if counter[word] >= rare_word_threshold_count:
                keep_words.add(word)
        vocab_lm = set(keep_words)
        print("Vocab size after culling rare words: {}".format(len(vocab_lm)))

        if args.vector_dim_limit > 0:
            vector_dim_limit = args.vector_dim_limit
            if len(vocab_lm) > vector_dim_limit:
                n_rm_words = len(vocab_lm) - vector_dim_limit
                print("Vector dim limit set: {}; {} least popular words will be culled.".format(vector_dim_limit, n_rm_words))
                tokens_all2 = [t for t in tokens_all_lm if t in vocab_lm]
                counter2 = collections.Counter(tokens_all2)
                keep_words = []
                for i, tup in enumerate(counter2.most_common(vector_dim_limit)):
                    w, c = tup
                    keep_words.append(w)
                    if i == vector_dim_limit - 1:
                        print("Last word kept: '{}' (#={}, {}%)".format(w, c, 100*c/len(tokens_all2)))
                vocab_lm = set(keep_words)

    vocab_lm_size = len(vocab_lm)
    vocab_lm_list = sorted(vocab_lm)
    vocab_lm_indices = {vocab_lm_list[i]: i for i in range(len(vocab_lm_list))}
    print("Final vocabulary size: {}".format(vocab_lm_size))
    print()

    # Create count vectors
    tf_mat = np.zeros((n_docs, vocab_lm_size))
    cvec_mat = np.zeros((n_docs, vocab_lm_size))
    for doc_idx, d_i in enumerate(tqdm(doc_infos, desc="Computing count vectors")):
        cvec = np.zeros(vocab_lm_size)
        for word in d_i['tokens']:
            word_lm = word2lm[word]
            if word_lm not in vocab_lm:
                continue
            word_idx = vocab_lm_indices[word_lm]
            cvec[word_idx] += 1
        cvec_mat[doc_idx] = cvec
        n_words_in_doc = cvec.sum()
        if n_words_in_doc > 0:
            tf_mat[doc_idx] = cvec / n_words_in_doc 
        d_i['count_vector'] = cvec
    print("Computing TF-IDF...")
    woccurrences_a = (cvec_mat > 0).astype(int).sum(axis=0).astype(float)
    assert np.all(woccurrences_a)
    idf_a = np.log10(np.divide(n_docs, woccurrences_a))
    tfidf_mat = np.broadcast_to(idf_a, tf_mat.shape) * tf_mat
    print()

    doc_fp_list = [str(d['path']) for d in doc_infos]
    cvec_df = pd.DataFrame(cvec_mat, columns=vocab_lm_list, index=doc_fp_list)
    tfidf_df = pd.DataFrame(tfidf_mat, columns=vocab_lm_list, index=doc_fp_list)

    meta_outpath = output_dir / "meta.json"
    cvec_outpath = output_dir / "count_vectors.csv"
    tfidf_outpath = output_dir / "tfidf.csv"
    print("Writing metadata to '{}'...".format(meta_outpath))
    with open(meta_outpath, 'w', encoding='utf-8') as f:
        meta = {
            'vocabulary': list(vocab_lm),
            'vocabulary_raw': list(vocab),
            'count_vectors': {
                'save_path': str(cvec_outpath),
            },
            'tfidf': {
                'save_path': str(tfidf_outpath),
            },
        }
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print("Writing count vectors to '{}'...".format(cvec_outpath))
    cvec_df.to_csv(cvec_outpath)
    print("Writing TF-IDF matrix to '{}'...".format(tfidf_outpath))
    tfidf_df.to_csv(tfidf_outpath)
    print("Done")

if __name__ == '__main__':
    main_funs = [make_vectors.__name__]

    if len(sys.argv) <= 1:
        print("Missing operation (must be one of {})".format(main_funs))
        exit(1)

    fun = sys.argv[1]
    if fun not in main_funs:
        print("Invalid operation: '{}' (must be one of {})".format(fun, main_funs))
        exit(1)

    sys.argv = sys.argv[0:1] + sys.argv[2:]

    globals()[fun]()