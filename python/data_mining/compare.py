from __future__ import print_function, division

import random
import numpy as np


def get_shingles(text, k=9, pad="_"):
    """Extracts shingles of size k from text.

    # Args
        text (str): The document to extract shingles for.
        k (int): Shingle size.

    # Returns:
        shingles [int]: The extracted shingles.
    """
    if len(text) < k:
        text = text.ljust(k, pad)
    return [text[i:i + k] for i in range(len(text) - k + 1)]


def get_hash_set(sequence, hash_f=hash):
    """TODO

    Args:
        sequence:
        hash_f:

    Returns:

    """
    return set([hash_f(e) for e in sequence])


def get_jaccard_similarity(s, t):
    return len(s.intersection(t)) / len(s.union(t))


class MinHashing(object):

    def __init__(self, n_rows, n_hash_fs=None, hash_fs=None):
        self.n_rows = n_rows
        if hash_fs is not None:
            if n_hash_fs is not None:
                raise ValueError(
                    "`n_hash_fs` should not be specified when `hash_fs` is passed"
                )
            self.hash_fs = hash_fs
            self.n_hash_fs = len(hash_fs)
        else:
            if n_hash_fs is None:
                raise ValueError("`n_hash_fs` (or `hash_fs`) must be specified")
            self.n_hash_fs = n_hash_fs
            ks = [random.randint(1, 10) for _ in range(n_rows)]
            ms = [random.randint(1, n_rows) for _ in range(n_rows)]
            # TODO better default hashes!!
            self.hash_fs = [lambda x: (x * k + m) % n_rows for k, m in zip(ks, ms)]

    def __call__(self, hash_sets):
        """Computes the MinHash signatures for hash_sets.

        Args:
            hash_sets:

        Returns:

        """
        n_columns = len(hash_sets)
        sorted_columns = [sorted(hs) for hs in hash_sets]
        signatures = np.ones((self.n_hash_fs, n_columns)) * np.inf
        cols_parse_idx = np.zeros(n_columns, dtype=int)

        for row in range(self.n_rows):
            hs = [hash_f(row) for hash_f in self.hash_fs]
            for c, col in enumerate(sorted_columns):

                while cols_parse_idx[c] < len(col) and col[cols_parse_idx[c]] < row:
                    cols_parse_idx[c] += 1

                if cols_parse_idx[c] < len(col) and col[cols_parse_idx[c]] == row:
                    for i in range(self.n_hash_fs):
                        signatures[i, c] = min(hs[i], signatures[i, c])

        return [list(signatures[:, c]) for c in range(n_columns)]


def get_estimated_jaccard_similarity_from_signatures(s_sign, t_sign):
    """

    Args:
        s_sign:
        t_sign:

    Returns:

    """
    if not len(s_sign) == len(t_sign):
        raise ValueError("signatures for s and t must be of same length")
    equal = np.array(s_sign) == np.array(t_sign)
    fraction_equal = equal.sum() / len(s_sign)

    return fraction_equal
