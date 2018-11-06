from __future__ import print_function, division

import json
import os
import itertools

import numpy as np
import pandas as pd

from pprint import pprint

from data_mining.data import get_news_groups_documents
from data_mining.compare import (
    get_shingles,
    get_hash_set,
    MinHashing,
    LocalitySensitiveHashing,
    get_jsim_and_approx_jsim,
    get_approximation_diffs,
    get_jaccard_similarity,
    get_p_lsh_candidate)
from data_mining.script_utils import timer, mkdirp, DEFAULT_RESULTS_PATH
from matplotlib import pyplot as plt
from tqdm import tqdm

OUTPUT_PATH = os.path.join(DEFAULT_RESULTS_PATH, 'homework_1', 'benchmark')
SHINGLES_SIZE = 10


def run_timing_job(
    n_docs,
    n_rows,
    minhash_size,
    lsh_nbands
):
    """Times the execution of computing Min Hashes and running LSH.
    """
    job_name = 'ndocs={}_nrows={}_minhash_size={}_lsh_nbands={}'.format(
        n_docs,
        n_rows,
        minhash_size,
        lsh_nbands
    )
    job_path = os.path.join(OUTPUT_PATH, job_name)
    mkdirp(job_path)
    result_path = os.path.join(job_path, "result.json")

    if os.path.exists(result_path):
        with open(result_path) as f:
            result = json.load(f)

        return result

    def primary_hash(s):
        """Pythons default hash(str) returns value in range
            [-(sys.maxint + 1):sys.maxint]
        """
        return hash(s) % n_rows

    print("loading documents")
    documents = list(get_news_groups_documents(first_n=n_docs))

    print("extracting shingle hashes")
    with timer() as prep_timer:
        shingle_sequences = [
            get_shingles(doc, k=SHINGLES_SIZE) for doc in documents
        ]
        hash_sets = [
            get_hash_set(ss, hash_f=primary_hash) for ss in shingle_sequences
        ]

    print("computing minhash signatures")
    mh = MinHashing(n_rows=n_rows, n_hash_fs=minhash_size)
    with timer() as minhash_timer:
        signatures = mh(hash_sets)

    print("finding similar documents using LSH")
    lsh = LocalitySensitiveHashing(n_bands=lsh_nbands)
    with timer() as lsh_timer:
        top_similar_index = lsh.get_top_similar_index(signatures)

    top_similarities_index = get_jsim_and_approx_jsim(
        top_similar_index,
        hash_sets,
        signatures
    )
    pprint(top_similarities_index)
    with open(os.path.join(job_path, "top_similarities_index.json"), 'w') as f:
        json.dump(top_similarities_index, f)

    diffs = get_approximation_diffs(top_similarities_index)
    diff_mean = np.mean(diffs)
    diff_std = np.std(diffs)
    f, ax = plt.subplots(1, 1)
    ax.hist(diffs, bins=np.arange(-0.3, 0.31, 0.025))
    ax.set_xlim(-0.3, 0.3)
    f.set_size_inches((8, 6))
    f.savefig(os.path.join(job_path, "error_dist.png"), format="png")

    result = {
        "ndocs": n_docs,
        "nrows": n_rows,
        "minhash_size": minhash_size,
        "lsh_nbands": lsh_nbands,
        "diff_mean": diff_mean,
        "diff_std": diff_std,
        "n_matches": len(diffs),
        "t_prep": prep_timer.elapsed,
        "t_minhash": minhash_timer.elapsed,
        "t_lsh": lsh_timer.elapsed
    }
    with open(result_path, "w") as f:
        json.dump(result, f)

    return result


def plot_time_vs_ndocs(df):
    f, axs = plt.subplots(2, 1)
    df.plot.line(x="ndocs", y="t_minhash", marker="*", ax=axs[0])
    df.plot.line(x="ndocs", y="t_lsh", marker="*", ax=axs[1])
    axs[0].set_ylabel("time [s]")
    axs[1].set_ylabel("time [s]")
    return f, axs


def run_vary_ndocs():
    n_docs_s = [64, 128, 256, 512, 1024, 2048]
    n_rows_s = [100003]
    minhash_size_s = [100]
    lsh_nbands_s = [20]

    run_args = itertools.product(
        n_docs_s,
        n_rows_s,
        minhash_size_s,
        lsh_nbands_s
    )
    results = [run_timing_job(*args) for args in run_args]
    df = pd.DataFrame(results)
    f, axs = plot_time_vs_ndocs(df)
    f.set_size_inches((12, 8))
    f.savefig(os.path.join(OUTPUT_PATH, "time_vs_ndocs.png"), format='png')

    return df


def plot_time_vs_nrows(df, ndocs):
    f, ax = plt.subplots(1, 1)
    df[df.ndocs == ndocs].plot.line(x="nrows", y="t_minhash", marker="*", ax=ax)
    ax.set_ylabel("time [s]")
    return f, ax


def run_vary_nrows():
    n_docs_s = [512]
    n_rows_s = [10007, 33331, 100003, 300007, 1000003]
    minhash_size_s = [100]
    lsh_nbands_s = [20]
    run_args = itertools.product(
        n_docs_s,
        n_rows_s,
        minhash_size_s,
        lsh_nbands_s
    )
    results = [run_timing_job(*args) for args in run_args]
    df = pd.DataFrame(results)
    f, axs = plot_time_vs_nrows(df, ndocs=n_docs_s[0])
    f.set_size_inches((12, 6))
    f.savefig(os.path.join(OUTPUT_PATH, "time_vs_nrows.png"), format='png')

    return df


def run_vary_nbands_and_signature_size():
    run_args = [
        (1024, 100003, 20, 5),
        (1024, 100003, 50, 5),
        (1024, 100003, 100, 5),
        (1024, 100003, 100, 10),
        (1024, 100003, 100, 20),
        (1024, 100003, 100, 50),
        (1024, 100003, 200, 10),
        (1024, 100003, 200, 20),
        (1024, 100003, 200, 50),
    ]
    results = [run_timing_job(*args) for args in run_args]
    df = pd.DataFrame(results)

    return df


def get_jsim_matrix(n_docs, n_rows=None):
    """Computes the full Jaccard similarity matrix for documents.

    If n_rows=None (default) the _true_ shingle set similarities will be used (no
    approximation) otherwise they will first be hashed to an hash space of size
    `n_rows`.
    """
    output_path = os.path.join(
        OUTPUT_PATH,
        "jsim_matrix_ndocs={}_nrows={}.npy".format(
            n_docs,
            n_rows
        )
    )
    if os.path.exists(output_path):
        sim_matrix = np.load(output_path)
        return sim_matrix

    documents = list(get_news_groups_documents(first_n=n_docs))
    shingle_sequences = [
        get_shingles(doc, k=SHINGLES_SIZE) for doc in documents
    ]
    if n_rows is not None:
        def hash_f(s):
            return hash(s) % n_rows
        shingles_sets = [get_hash_set(ss, hash_f) for ss in shingle_sequences]
    else:
        shingles_sets = [set(ss) for ss in shingle_sequences]

    sim_matrix = np.eye(n_docs) * 0.5
    for i in tqdm(range(n_docs)):
        for j in range(i + 1, n_docs):
            sim_matrix[i, j] = get_jaccard_similarity(
                shingles_sets[i],
                shingles_sets[j]
            )

    sim_matrix = sim_matrix + sim_matrix.T
    np.save(output_path, sim_matrix)

    return sim_matrix


def run_accuracy_vs_nrows(n_docs=512):
    true_sim_matrix = get_jsim_matrix(n_docs, None)
    n_rows_s = [10007, 33331, 100003, 300007, 1000003, 10000019]
    sim_matrices = [get_jsim_matrix(n_docs, n_rows) for n_rows in n_rows_s]
    diffs = [(sm - true_sim_matrix).flatten() for sm in sim_matrices]
    f, axs = plt.subplots(len(n_rows_s), 1, sharex=True)
    bins = np.arange(0, 0.1, 0.001)
    for d, n_rows, ax in zip(diffs, n_rows_s, axs):
        ax.hist(d, bins=bins)
        ax.set_ylabel("nrows={}".format(n_rows))
    ax.set_xlabel("Jaccard similarity diff (approximation - true)")
    f.set_size_inches((12, 16))
    f.savefig(os.path.join(OUTPUT_PATH, "accuracy_vs_nrows.png"), format="png")
    df = pd.DataFrame(
        {
            "diff_mean": [np.mean(d) for d in diffs],
            "diffs_std": [np.std(d) for d in diffs],
            "diffs_95th": [np.percentile(d, 95) for d in diffs],
            "diffs_max": [np.max(d) for d in diffs],
        },
        index=n_rows_s
    )

    return df


def measure_lsh_recall(
    n_docs=512,
    n_rows=1000003,
    minhash_size=100,
    lsh_nbands=20,
    jsim_threshold=0.7
):
    """Computes the recall of the LSH algorithm by comparing to "brute force"
    calculation"""
    job_name = 'ndocs={}_nrows={}_minhash_size={}_lsh_nbands={}'.format(
        n_docs,
        n_rows,
        minhash_size,
        lsh_nbands
    )
    job_path = os.path.join(OUTPUT_PATH, job_name)
    mkdirp(job_path)

    top_sim_path = os.path.join(job_path, "top_similarities_index.json")
    if not os.path.exists(top_sim_path):
        run_timing_job(n_docs, n_rows, minhash_size, lsh_nbands)
    with open(top_sim_path) as f:
        top_similarities_index = json.load(f)
    top_similar_set_index = {
        int(doc_idx): set([sim_doc[0] for sim_doc in sim_docs])
        for doc_idx, sim_docs in top_similarities_index.items()
    }
    jsim_matrix = get_jsim_matrix(n_docs, n_rows)
    tps = 0
    fns = 0
    true_similar_index = {}
    for i in range(n_docs):
        similar_docs = set(np.where(jsim_matrix[i] > jsim_threshold)[0]) - {i}
        lsh_similar_docs = top_similar_set_index.get(i, set([]))
        intersection = similar_docs.intersection(lsh_similar_docs)
        missing = similar_docs - intersection
        tps += len(intersection)
        fns += len(missing)
        if len(similar_docs) > 0:
            true_similar_index[i] = similar_docs

    recall = tps / (tps + fns)
    p_candidate_at_threshold = get_p_lsh_candidate(
        jsim=jsim_threshold,
        n_bands=lsh_nbands,
        n_rows_per_band=minhash_size / lsh_nbands
    )
    result = {
        "ndocs": n_docs,
        "nrows": n_rows,
        "minhash_size": minhash_size,
        "lsh_nbands": lsh_nbands,
        "nrows_per_band": minhash_size / lsh_nbands,
        "jsim_threshold": jsim_threshold,
        "p_candidate_at_threshold": p_candidate_at_threshold,
        "support": tps + fns,
        "recall": recall,
    }
    print(result)

    return result


def run_recall_vs_threshold():
    jsims = np.arange(0, 1, 0.005)
    p_candidate_curve = np.array([
        get_p_lsh_candidate(jsim, n_bands=20, n_rows_per_band=5)
        for jsim in jsims
    ])
    f, ax = plt.subplots(1, 1)
    ax.plot(jsims, p_candidate_curve)
    ax.set_ylabel("p(LSH-candidate)")
    ax.set_xlabel("Jaccard similarity")
    f.set_size_inches(12, 6)
    f.savefig(
        os.path.join(OUTPUT_PATH, "p_candidates_curve.png"),
        format="png"
    )
    results = [measure_lsh_recall(jsim_threshold=th) for th in [0.3, 0.5, 0.7, 0.9]]

    return pd.DataFrame(results)


if __name__ == '__main__':
    mkdirp(OUTPUT_PATH)
    res_vary_ndocs = run_vary_ndocs()
    res_vary_nrows = run_vary_nrows()
    res_vary_nbands_and_signature_size = run_vary_nbands_and_signature_size()
    res_accuracy_vs_nrows = run_accuracy_vs_nrows()
    res_recall_vs_threshold = run_recall_vs_threshold()
