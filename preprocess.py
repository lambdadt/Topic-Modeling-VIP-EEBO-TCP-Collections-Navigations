import numpy as np
import pandas as pd
from tqdm import tqdm
import nltk
from bs4 import BeautifulSoup

from pathlib import Path
import sys
import argparse
import json
import re
import os
import os.path as osp
import math
import collections
import xml.etree.ElementTree as ET


def decode_xml_texts():
    ap = argparse.ArgumentParser(description="""
    Decode XML encoded texts from https://textcreationpartnership.org/about-the-tcp/historical-documentation/tcp-production-files/ to be used for topic modelling.
    """)
    ap.add_argument('--input_dir', '-i', required=True, help="Directory containing EEBO-TCP XML files (e.g., Navigations_headed_xml/A0-A5/) from which XML files will be read. Not recursive. Can also pass the path of a single file.")
    ap.add_argument('--output_dir', '-o', required=True, help="Directory where decoded text files will be output. Output file naming may vary depending on --method. See Navigations_headed_xml/Parsed_texts/ for reference.")
    ap.add_argument('--metadata_output_dir', required=True, help="Directory where metadata will be saved.")
    ap.add_argument('--method', default='Spring2025', choices=['Fall2024', 'Spring2025'])
    ap.add_argument('--doc_start_num', type=int, default=-1, help="num is '00005' for file 'A00005.headed.xml. Set to value 0 or greater to enable (otherwise all XMLs will be considered without file name based filtering).")
    ap.add_argument('--doc_end_num', type=int, default=-1, help="See --doc_start_num")
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    if input_dir.is_dir():
        all_xml_paths = [p for p in Path(args.input_dir).iterdir() if p.suffix.lower() == ".xml"]
    elif input_dir.is_file():
        all_xml_paths = [Path(input_dir)]
    else:
        print("Input directory is invalid: {}".format(input_dir))
        exit(1)
    all_xml_paths = sorted(all_xml_paths)
    print("{} XML file(s) found under input dir ({}).".format(len(all_xml_paths), args.input_dir))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    print("Outputs will be saved to: {}".format(output_dir))
    metadata_output_dir = Path(args.metadata_output_dir)
    metadata_output_dir.mkdir(exist_ok=True, parents=True)
    print("Metadata outputs will be saved to: {}".format(metadata_output_dir))

    metadata_all = []
    dry_run = args.dry_run

    n_failed = 0
    for ixml, xml_path in enumerate(all_xml_paths):
        print(75 * "=")
        print("Parsing XML [{}/{}]: {} (#failed: {})".format(ixml+1, len(all_xml_paths), xml_path, n_failed))
        doc_prefix_alphabet = None
        doc_num_str = None
        m = re.match(r"^([A-Z])(\d+).headed.xml$", xml_path.name)
        if m:
            doc_prefix_alphabet = m.group(1)
            doc_num_str = m.group(2)
            doc_num = int(doc_num_str)
            if ((args.doc_start_num >= 0 and doc_num < args.doc_start_num) or 
                (args.doc_end_num >= 0 and doc_num > args.doc_end_num)):
                continue

        with open(xml_path, encoding='utf-8') as f:
            xml_text = f.read()
        soup = BeautifulSoup(xml_text, 'lxml-xml')

        idg_e = soup.select_one('IDG[ID]')
        if idg_e is None:
            print("IDG element with 'ID' attribute not found. Skipping.")
            n_failed += 1
            continue
        doc_id = idg_e.attrs['ID']

        meta = {
            'doc_id': doc_id,
            'xml_path': str(xml_path),
        }

        if args.method == 'Spring2025':
            del_tags = ['IDNO', 'AVAILABILITY', 'IDG', 'HEADER', 'FILEDESC', 'PUBLICATIONSTMT',
                        'PUBPLACE', 'PUBLISHER', 'SERIESSTMT', 'NOTESSTMT',
                        'NOTE', 'FOOTNOTE', 'REF', 'FN'] # , 'FIGURE', 'HI'
            for e in soup.find_all(del_tags):
                e.decompose()
            # While this ignores most structures (e.g., pages separated by PB, ITEM, TITLE, BODY, TEXT, DIVx etc.)
            # seems to produce a readable result with reasonable amount of white space between elements.
            text_content = soup.text.strip()
            output_text_path = output_dir / f"{doc_id}_simplified_content.txt"
            if not dry_run:
                with open(output_text_path, 'w', encoding='utf-8') as f:
                    f.write(text_content)
            print(f"Text content saved to: {output_text_path}")
            meta['text_doc_path'] = str(output_text_path)
        elif args.method == 'Fall2024':
            output_text_path = output_dir / f"{doc_id}_parsed_text.txt"
            output_footnotes_path = output_dir / f"{doc_id}_footnotes.txt"
            text_content, footnotes = None, None
            try:
                text_content, footnotes = parse_xml_fall2024(xml_path)
            except ET.ParseError as e:
                print(f"Failed to parse {xml_path}: {e}")
                n_failed += 1
            if text_content:
                if not dry_run:
                    with open(output_text_path, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(text_content))
                print(f"Text content saved to: {output_text_path}")
            if footnotes:
                with open(output_footnotes_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(footnotes))
                print(f"Footnotes content saved to: {output_footnotes_path}")
            meta['text_doc_path'] = str(output_text_path)
        
        # Extract metadata from XML
        soup = BeautifulSoup(xml_text, 'lxml-xml')
        header_e = soup.find('HEADER')
        if header_e is not None:
            eebo_id_e = header_e.select_one("IDNO[TYPE=DLPS]")
            meta['eebo_id'] = eebo_id_e.text.strip() if eebo_id_e is not None else None
            stc_elms = header_e.select("IDNO[TYPE=stc]")
            if len(stc_elms) > 1:
                full_estc_text = stc_elms[1].text.strip()
                meta['estc_number'] = full_estc_text.split('ESTC ')[-1]
            else:
                meta['estc_number'] = None
            title_e = header_e.select_one("TITLESTMT > TITLE[TYPE='245']")
            meta['title'] = title_e.text.strip() if title_e is not None else None
            author_e = header_e.select_one("AUTHOR")
            meta['author'] = author_e.text.strip() if author_e is not None else None

            for ipubstmt, pubstmt in enumerate(header_e.find_all('PUBLICATIONSTMT')):
                child_tags = {child.name for child in pubstmt.find_all(recursive=False)}
                publisher = pubstmt.find('PUBLISHER', recursive=False)
                date = pubstmt.find('DATE', recursive=False)
                pubplace = pubstmt.find('PUBPLACE', recursive=False)
                if (publisher is None or "text creation partnership" in publisher.text.lower()) or \
                   (date is None or 'eebo-tcp' in date.text.lower()) or \
                    pubplace is None:
                   continue
                assert ipubstmt == 1 and len(child_tags) == 3
                meta['publisher'] = getattr(publisher, 'text', "").strip() or None
                meta['date'] = getattr(date, 'text', "").strip() or None
                meta['pubplace'] = getattr(pubplace, 'text', "").strip() or None
        metadata_all.append(meta)
        print()
    metadata_df = pd.DataFrame(metadata_all)
    metadata_csv_save_path = metadata_output_dir / "decoded_documents_metadata.csv"
    if metadata_csv_save_path.is_file():
        print("Metadata CSV already exists ({}). It will be merged.".format(metadata_csv_save_path))
        metadata_df_existing = pd.read_csv(metadata_csv_save_path)
        print("Old metadata CSV has {} entries (current one has {}). Merging...".format(len(metadata_df_existing), len(metadata_df)))
        metadata_df_merged = pd.concat([metadata_df_existing, metadata_df], axis='index')
        print("Merged metadata entries: {}".format(len(metadata_df_merged)))
        metadata_df = metadata_df_merged
    print("Saving metadata csv to: {}".format(metadata_csv_save_path))
    metadata_df.to_csv(metadata_csv_save_path, index=False)
    print("Done")


def parse_xml_fall2024(input_file):
    tree = ET.parse(input_file)
    root = tree.getroot()
    # Initialize variables to store text and footnotes
    text_content = []
    footnotes = []
    # Recursive function to extract text and footnotes
    def extract_content(element):
        for child in element:
            if child.tag.lower() in ['note', 'footnote', 'ref', 'fn']:  # Assuming footnotes are in these tags
                footnotes.append(child.text.strip() if child.text else '')
            else:
                if child.text:
                    text_content.append(child.text)
                extract_content(child)  # Recurse into child elements
            if child.tail:
                text_content.append(child.tail)
    # Start extraction
    extract_content(root)
    return text_content, footnotes


def make_vectors():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output_dir', '-o', required=True)
    ap.add_argument('--parsed_texts_dir', '-i', default="Navigations_headed_xml/Parsed_texts",
                    help=".txt files in this directory will be searched for recursively.")
    ap.add_argument('--stemmer', default='None', choices=['None', 'Porter', 'Snowball', 'WordNetLemmatizer'],
                    help="If not provided (default), WordNet Lemmatizer will be used.")
    ap.add_argument('--rare_word_threshold', default=0.0, type=float, help="Cull words (after lemmatization) used fewer then x%%")
    ap.add_argument('--vector_dim_limit', default=100_000, type=int, help="Rare words will be culled. -1 means no limit.")
    ap.add_argument('--tfidf_reweighted_count_vectors_as_float', action='store_true',
                    help="After reweighting the count vectors with TF-IDF, return the resulting matrix with floats, instead of rounding to nearist integers.")
    ap.add_argument('--filter', choices=['txt', 'parsed_text', 'footnotes'], default='txt',
                    help="If set to `txt` will read all .txt files.")
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
            if ((args.filter == 'txt' and path.suffix.lower() != '.txt') or
                (args.filter == 'parsed_text' and not path.name.lower().endswith("parsed_text.txt")) or
                (args.filter == 'footnotes' and not path.name.lower().endswith("footnotes.txt"))):
                continue
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

    # Create count vectors & TF-IDF
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

    # Reweight count vectors
    print("Computing count vectors reweighted by TF-IDF...")
    rowsum = tfidf_mat.sum(axis=1, keepdims=True)
    tfidf_dist_mat = np.divide(tfidf_mat, rowsum, out=np.zeros_like(tfidf_mat), where=rowsum != 0)
    tfidf_reweighted_cvects = tfidf_dist_mat * cvec_mat.sum(axis=1, keepdims=True)
    if not args.tfidf_reweighted_count_vectors_as_float:
        tfidf_reweighted_cvects = np.round(tfidf_reweighted_cvects).astype(int)

    doc_fp_list = [str(d['path']) for d in doc_infos]
    cvec_df = pd.DataFrame(cvec_mat, columns=vocab_lm_list, index=doc_fp_list)
    tfidf_df = pd.DataFrame(tfidf_mat, columns=vocab_lm_list, index=doc_fp_list)
    tfidf_reweighted_cvec_df = pd.DataFrame(tfidf_reweighted_cvects, columns=vocab_lm_list, index=doc_fp_list)

    meta_outpath = output_dir / "meta.json"
    cvec_outpath = output_dir / "count_vectors.csv"
    tfidf_outpath = output_dir / "tfidf.csv"
    tfidf_reweighted_cvec_outpath = output_dir / "tfidf_reweighted_count_vectors.csv"
    print("Writing metadata to '{}'...".format(meta_outpath))
    with open(meta_outpath, 'w', encoding='utf-8') as f:
        meta = {
            'vocabulary': list(vocab_lm),
            'vocabulary_raw': list(vocab),
            'count_vectors': {
                'vanilla': {
                    'save_path': str(cvec_outpath),
                },
                'reweighted': {
                    'tfidf': {
                        'save_path': str(tfidf_reweighted_cvec_outpath),
                    },
                },
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
    print("Writing count vectors reweighted by TF-IDF to: '{}'...".format(tfidf_reweighted_cvec_outpath))
    tfidf_reweighted_cvec_df.to_csv(tfidf_reweighted_cvec_outpath)
    print("Done")

if __name__ == '__main__':
    main_funs = [make_vectors.__name__, decode_xml_texts.__name__]

    if len(sys.argv) <= 1:
        print("Missing operation (must be one of {})".format(main_funs))
        exit(1)

    fun = sys.argv[1]
    if fun not in main_funs:
        print("Invalid operation: '{}' (must be one of {})".format(fun, main_funs))
        exit(1)

    sys.argv = sys.argv[0:1] + sys.argv[2:]

    globals()[fun]()