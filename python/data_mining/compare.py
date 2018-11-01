from __future__ import print_function, division

import random
from functools import partial
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from data_mining import primes


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

    def __init__(self, n_rows, n_hash_fs=None, hash_fs=None, seed=0):
        self.n_rows = n_rows
        if hash_fs is not None:
            if n_hash_fs is not None:
                raise ValueError(
                    "`n_hash_fs` should not be specified when `hash_fs` is passed"
                )
            self.n_hash_fs = len(hash_fs)
            self.hash_fs = hash_fs

        else:
            if n_hash_fs is None:
                raise ValueError("`n_hash_fs` (or `hash_fs`) must be specified")
            self.n_hash_fs = n_hash_fs

            # TODO verify / select better default hashes
            self.ks = primes.first(self.n_hash_fs)
            np.random.seed(seed)
            self.ms = np.random.permutation(primes.first(self.n_hash_fs))

            def mod_hash(x, k, m):
                return (x * k + m) % self.n_rows

            self.hash_fs = [
                partial(mod_hash, k, m)
                for k, m in zip(self.ks, self.ms)
            ]

    def __call__(self, hash_sets):
        """Computes the MinHash signatures for hash_sets.

        Args:
            hash_sets:

        Returns:

        """
        hash_sets = list(hash_sets)  # loads all in memory
        n_columns = len(hash_sets)
        sorted_columns = [sorted(hs) for hs in hash_sets]
        signatures = np.ones((self.n_hash_fs, n_columns), dtype=int) * self.n_rows + 1
        cols_parse_idx = np.zeros(n_columns, dtype=int)

        for row in tqdm(range(self.n_rows)):
            hfs = [hash_f(row) for hash_f in self.hash_fs]
            for c, col in enumerate(sorted_columns):
                if cols_parse_idx[c] < len(col) and col[cols_parse_idx[c]] == row:
                    for i in range(self.n_hash_fs):
                        signatures[i, c] = min(hfs[i], signatures[i, c])
                    cols_parse_idx[c] += 1

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


class LocalitySensitiveHashing(object):

    def __init__(self, n_bands, hash_f=hash):
        self.n_bands = n_bands
        self.hash_f = hash_f

    def get_band_groups(self, signatures):
        # (should assert all have same length?)
        n_rows = len(signatures[0])
        bands = [  # divide to as even chunks as possible
            slice(a[0], a[-1] + 1)
            for a in np.array_split(range(n_rows), self.n_bands)
        ]
        band_groups = [defaultdict(lambda: []) for _ in bands]
        for doc_idx, signature in enumerate(signatures):
            for b, sl in enumerate(bands):
                band_groups[b][self.hash_f(tuple(signature[sl]))].append(doc_idx)

        return band_groups

    @staticmethod
    def get_top_similar_index_from_band_groups(band_groups):
        doc_idx_to_top_similar = defaultdict(lambda: set([]))
        for groups in band_groups:
            for group in groups.values():
                if len(group) > 1:
                    group_set = set(group)
                    for doc_idx in group:
                        doc_idx_to_top_similar[doc_idx] = \
                            doc_idx_to_top_similar[doc_idx].union(group_set)

        return {
            doc_idx: sorted(top_similar - {doc_idx})
            for doc_idx, top_similar in doc_idx_to_top_similar.iteritems()
        }

    def get_top_similar_index(self, signatures):
        band_groups = self.get_band_groups(signatures)
        return self.get_top_similar_index_from_band_groups(band_groups)


LSH = LocalitySensitiveHashing


def get_jsim_and_approx_jsim(top_similar_index, hash_sets, signatures):
    return {
        doc_idx_a: [
            (doc_idx_b, {
                'jsim': get_jaccard_similarity(
                    hash_sets[doc_idx_a],
                    hash_sets[doc_idx_b]
                ),
                'jsim_approx': get_estimated_jaccard_similarity_from_signatures(
                    signatures[doc_idx_a],
                    signatures[doc_idx_b]
                )
            }) for doc_idx_b in top_similar
        ] for doc_idx_a, top_similar in top_similar_index.iteritems()
    }
