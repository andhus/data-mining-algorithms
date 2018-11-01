from __future__ import print_function, division

import json
import os
import itertools

import numpy as np
import pandas as pd

from cPickle import load, dump
from pprint import pprint

from data_mining.data import get_news_groups_documents
from data_mining.compare import (
    get_shingles,
    get_hash_set,
    MinHashing,
    LocalitySensitiveHashing,
    get_jsim_and_approx_jsim,
    get_approximation_diffs
)
from data_mining.script_utils import timer, mkdirp
from matplotlib import pyplot as plt

# compute expected probabilities!

# n_docs = [20, 80, 320]
# n_rows = [1007, ...]
# minhash_size = [10, 50, 100, 500]


# look at execution time of extraction and comparison separately
# look at accuracy

OUTPUT_PATH = os.path.abspath(__file__)[:-3]
SHINGLES_SIZE = 10


def run_job(
    n_docs,
    n_rows,
    minhash_size,
    lsh_nbands
):
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
        hash_sets = [get_hash_set(ss, hash_f=primary_hash) for ss in shingle_sequences]

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


def plot_timings(df, x="ndocs"):
    f, axs = plt.subplots(1, 2)
    df.plot.line(x=x, y="t_minhash", marker="*", ax=axs[0])
    df.plot.line(x=x, y="t_lsh", marker="*", ax=axs[1])
    return f, axs


if __name__ == '__main__':
    mkdirp(OUTPUT_PATH)

    # vary n_docs
    n_docs_s = [20, 80, 320, 640, 1280]
    n_rows_s = [10007]
    minhash_size_s = [100]
    lsh_nbands_s = [20]

    run_args_1 = itertools.product(
        n_docs_s,
        n_rows_s,
        minhash_size_s,
        lsh_nbands_s
    )
    results_1 = [run_job(*args) for args in run_args_1]
    df1 = pd.DataFrame(results_1)

    #
    # # vary n_rows
    # n_docs_s = [1280]
    # n_rows_s = [10007, 33331, 100003]
    # minhash_size_s = [100]
    # lsh_nbands_s = [20]
    #
    # run_args = itertools.product(
    #     n_docs_s,
    #     n_rows_s,
    #     minhash_size_s,
    #     lsh_nbands_s
    # )
    #
    # results_2 = [run_job(*args) for args in run_args]
    #
    # # vary n_bands
    # run_args = [
    #     (1280, 33331, 20, 5),
    #     (1280, 33331, 50, 5),
    #     (1280, 33331, 100, 5),
    #     (1280, 33331, 100, 10),
    #     (1280, 33331, 100, 20),
    #     (1280, 33331, 100, 50),
    #     (1280, 33331, 200, 10),
    #     (1280, 33331, 200, 20),
    #     (1280, 33331, 200, 50),
    # ]
    #
    # results_3 = [run_job(*args) for args in run_args]
