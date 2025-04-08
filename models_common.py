import numpy as np
import pandas as pd

import os
from pathlib import Path
import sys
import math
import random
from typing import Literal, Union


def compute_top_p(zw_mat, top_p=0.9) -> dict:
    """
    Retrieves top_p words for each topic, ordered by their frequency.

    Args:
        zw_mat: topic x word matrix
    
    Returns:
        dictionary of topic index -> word indices
    """
    top_p_words = {}

    assert len(zw_mat.shape) == 2
    n_topics, vocab_sz = zw_mat.shape
    zw_mat = zw_mat / zw_mat.sum(axis=1)
    for itopic in range(n_topics):
        word_dist = zw_mat.iloc[itopic] if isinstance(zw_mat, pd.DataFrame) else zw_mat[itopic]
        words_desc_indices = np.argsort(word_dist)[::-1]
        prob_cumsums = np.cumsum(word_dist[words_desc_indices])
        top_p_words[itopic] = words_desc_indices[prob_cumsums < top_p]

    return top_p_words


def compute_proxy_completeness_measure(zw_mat) -> float:
    """
    Compute a proxy 'completeness measure' without requiring ground truth clustering labels for topic models.

    Args:
        zw_mat: topic x word matrix
    """
    assert len(zw_mat.shape) == 2
    n_topics, vocab_sz = zw_mat.shape
    cluster_sz = round(vocab_sz / n_topics)
    total_covered_word_indices = set()
    for itopic in range(n_topics):
        word_dist = zw_mat.iloc[itopic] if isinstance(zw_mat, pd.DataFrame) else zw_mat[itopic]
        words_desc_indices = np.argsort(word_dist)[::-1]
        top_word_indices = words_desc_indices[:cluster_sz]
        total_covered_word_indices.update(top_word_indices)
    return len(total_covered_word_indices) / vocab_sz


def compute_proxy_homogeneity_score(zw_mat, scoring_metric: Literal['binary', 'reciprocal']='binary', return_dict=False) -> Union[float, dict]:
    """
    Compute a proxy 'homogeniety measure' without requiring ground truth clustering labels for topic models.

    Args:
        zw_mat: topic x word matrix
        scoring_metric:
            If 'binary', 1 if word is unique to each topic otherwise 0
            If 'reciprocal', 1 - (# other topics containing word)/(#topics - 1)
    """
    assert len(zw_mat.shape) == 2
    n_topics, vocab_sz = zw_mat.shape
    cluster_sz = round(vocab_sz / n_topics)
    clusters_mat = np.zeros((n_topics, vocab_sz))
    for itopic in range(n_topics):
        word_dist = zw_mat.iloc[itopic] if isinstance(zw_mat, pd.DataFrame) else zw_mat[itopic]
        words_desc_indices = np.argsort(word_dist)[::-1]
        top_word_indices = words_desc_indices[:cluster_sz]
        clusters_mat[itopic, top_word_indices] = 1

    clusters_sums = clusters_mat.sum(axis=0)

    h_scores = []
    for itopic in range(n_topics):
        clusters_mine = clusters_mat[itopic]
        if clusters_mine.sum() == 0:
            continue
        clusters_others_counts = clusters_sums - clusters_mine
        clusters_others_bin = clusters_others_counts > 0
        if scoring_metric == 'binary':
            h_scores.append(np.logical_and(clusters_mine, np.logical_not(clusters_others_bin)).sum() / cluster_sz)
        elif scoring_metric == 'reciprocal':
            h_scores.append((1 - (clusters_others_counts[clusters_mine.astype(bool)] / (n_topics - 1))).sum() / cluster_sz)
        else:
            raise ValueError()
    
    h_score_total = np.mean(h_scores) if len(h_scores) > 0 else 0
    if return_dict:
        return {
            'per_topic_h_scores': h_scores,
            'h_score': h_score_total,
        }
    else:
        return h_score_total


def compute_proxy_v_measure(zw_mat, homogeneity_score_metric: Literal['binary', 'reciprocal']='binary', beta=1.0) -> float:
    """
    Compute a proxy 'v-measure' without requiring ground truth clustering labels for topic models.

    Args:
        zw_mat: topic x word matrix
    """
    completeness_score = compute_proxy_completeness_measure(zw_mat)
    homogeneity_score = compute_proxy_homogeneity_score(zw_mat, scoring_metric=homogeneity_score_metric)
    return ((1 + beta) * homogeneity_score * completeness_score) / (beta * homogeneity_score + completeness_score)
