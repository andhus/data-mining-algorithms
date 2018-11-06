from __future__ import print_function, division

from functools import partial
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from data_mining import primes


def get_shingles(text, k=9, pad='_'):
    """Extracts shingles of size k from text.

    Args
        text (str): The document to extract shingles for.
        k (int): Shingle size.

    Returns:
        shingles ([int]): The extracted shingles.
    """
    if len(text) < k:
        text = text.ljust(k, pad)
    return [text[i:i + k] for i in range(len(text) - k + 1)]


def get_hash_set(iterable, hash_f=hash):
    """Extracts the set of hashes of elements in iterable.
    """
    return set([hash_f(e) for e in iterable])


def get_jaccard_similarity(s, t):
    """Computes the Jaccard Similarity of two sets"""
    return len(s.intersection(t)) / len(s.union(t))


class MinHashing(object):
    """Implements the Min-Hashing algorithm [1] for extracting min hash signatures
    from sets.

    Args
        n_rows (int): Size of the hash space of hash sets to encode.
        n_hash_fs (int): Number of hash functions to use to compute the signates,
            i.e. the size of the min hash signatures. Specify _either_ this argument
            or hash_fs bellow.
        hash_fs ([f: int -> int]) the specific hash functions to use. Specify
            _either_ this argument or n_hash_fs bellow.
        seed (int): Seed used for randomization in initialisation of hash functions.


    Example
        >>> mh = MinHashing(n_rows=10, n_hash_fs=3)
        >>> mh([{1,2,3,4,5,6}, {3,4,5,6,7,8}, {5, 9}])
        [[4, 0, 4], [0, 0, 0], [2, 3, 2]]

    References:
        [1] Mining of Massive Datasets, J. Leskovec et al., p. 81

    """
    def __init__(self, n_rows, n_hash_fs=None, hash_fs=None, seed=0):
        self.n_rows = n_rows
        if hash_fs is not None:
            if n_hash_fs is not None:
                raise ValueError(
                    '`n_hash_fs` should not be specified when `hash_fs` is passed'
                )
            self.n_hash_fs = len(hash_fs)
            self.hash_fs = hash_fs

        else:
            if n_hash_fs is None:
                raise ValueError('`n_hash_fs` (or `hash_fs`) must be specified')
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
        """Computes the MinHash signatures the provided for hash sets.

        Args:
            hash_sets ([{int}]): Iterable of hash sets to transform to min hash
                signatures.

        Returns:
            ([[int]]) The minhash signatures.

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

        # Alternative implementation only computing hashes for rows which
        # exists in one of the docs.
        #
        # row = min([col[0] for col in sorted_columns])
        # row_next = self.n_rows
        # while row < self.n_rows:
        #     hfs = [hash_f(row) for hash_f in self.hash_fs]
        #     for c, col in enumerate(sorted_columns):
        #         if cols_parse_idx[c] < len(col):
        #             col_row = col[cols_parse_idx[c]]
        #             if col_row == row:
        #                 for i in range(self.n_hash_fs):
        #                     signatures[i, c] = min(hfs[i], signatures[i, c])
        #                 cols_parse_idx[c] += 1
        #             row_next = min(col_row, row_next)
        #     row = row_next
        #     row_next = self.n_rows

        return [list(signatures[:, c]) for c in range(n_columns)]


def get_estimated_jaccard_similarity_from_signatures(s_sign, t_sign):
    """Computes the estimated Jaccard Similarity based on min hash signatures.

    Args:
        s_sign, t_sign ([int], [int]): The signatures to compare. They must be of
            same length.

    Returns:
        (float) The estimated Jaccard Similarity.

    """
    if not len(s_sign) == len(t_sign):
        raise ValueError('signatures for s and t must be of same length')
    equal = np.array(s_sign) == np.array(t_sign)
    fraction_equal = equal.sum() / len(s_sign)

    return fraction_equal


class LocalitySensitiveHashing(object):
    """Implements the LSH algorithm [1] for fast approximate detection of similar
    vectors (such as MinHash signatures).

    Args:
        n_bands (int): Number of bands to use, each vector is split up to this
            number of sub vectors.
        hash_f: The hash function to apply to each band

    Example
        >>> lsh = LocalitySensitiveHashing(3)
        >>> lsh.get_top_similar_index([
                [1,2,3,4,5,6],
                [1,2,3,5,5,7],
                [0,0,0,0,5,7]])
        {0: [1], 1: [0, 2], 2: [1]}

    References:
        [1] Mining of Massive Datasets, J. Leskovec et al., p. 87

    """
    def __init__(self, n_bands, hash_f=None):
        self.n_bands = n_bands
        self.hash_f = hash_f or hash

    def get_band_groups(self, signatures):
        """Splits signatures into bands and groups them within each band based on
        hash of band elements

        Args:
            signatures ([[int]]): Vectors/MinHash signatures to group.

        Returns:
            ([{<band hash>: <signature indices>}]), where:
                <band hash> (int): Hash of the elements in band.
                <signature indicies> ([int]): the indices of signatures that which
                    band hashed to <band hash>.
        """
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
        """Constructs an index with the most similar signatures for each signature (
        with at least one similar signature) based on band groups.

        Args:
            band_groups: See output of `get_band_groups`.

        Returns:
            ({<signature index>: [<similar signature index>, ...]})
        """

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
        """Constructs an index with the most similar signatures for each signature (
        with at least one similar signature).

        Args:
            signatures ([[int]]): Vectors/MinHash signatures to group.

        Returns:
            ({<signature index>: [<similar signature index>, ...]})
        """
        band_groups = self.get_band_groups(signatures)
        return self.get_top_similar_index_from_band_groups(band_groups)

    @staticmethod
    def get_p_candidate(jsim, n_bands, signature_size):
        """Computes probability that two signatures will be detected as similar
        using the this LSH algorithm.

        NOTE compared to `get_p_lsh_candidate` function below this version computes
        the probability even when n_bans does not evenly divide signature size.

        Args:
            jsim: Actuall/Assumed Jaccard Similarity between the hypothetical
                signatures
            n_bands: Number of bands
            signature_size: size of minhash signatures

        Returns:
            (float) Probability that the hypothetical signatures will be detected as
                similar.
        """
        band_sizes = [  # divide to as even chunks as possible
            len(a) for a in np.array_split(range(signature_size), n_bands)
        ]
        p_missed = 1.
        for band_size in band_sizes:
            p_missed *= (1 - jsim ** band_size)

        return 1. - p_missed

    @classmethod
    def get_n_bands(cls, jsim, signature_size, max_p_missed):
        """Computes the number of bands to choose for a given signature size and
        required maximum probability to miss a similar pair.

        Args:
            jsim: Jaccard similarity
            signature_size: Size of minhash signatures
            max_p_missed: The highest accepted probability that a similar pair is
                missed.

        Returns:
            The minimum number of bands to choose to achieve max_p_missed.
        """
        for n_bands in range(1, signature_size + 1):
            p = cls.get_p_candidate(jsim, n_bands, signature_size)
            if p > (1. - max_p_missed):
                break
        if not p > (1. - max_p_missed):
            raise ValueError(
                'No number of bands gives maximum missed probability of {} '
                'for signature size {} and Jaccard similarity {}'.format(
                    max_p_missed,
                    signature_size,
                    jsim
                )
            )

        return n_bands


LSH = LocalitySensitiveHashing


def get_lsh_top_similar_index(signatures, jsim, max_p_missed=1e-5, lsh_hash_f=None):
    """Helper function that selects the n_bands parameter and runs the LSH algorithm
    based on maximum accepted probability of missing a similar pair.

    Args:
        signatures ([int]): Minhash signatures.
        jsim: Jaccard similarity defining a "similar" pair
        max_p_missed: The highest accepted probability that a similar pair is
            missed.

    Returns:
        See output from `LocalitySensitiveHashing.get_top_similar_index`

    """
    signature_size = len(signatures[0])
    n_bands = LSH.get_n_bands(jsim, signature_size, max_p_missed)
    lsh = LSH(n_bands, hash_f=lsh_hash_f)
    top_similar_index = lsh.get_top_similar_index(signatures)

    return top_similar_index


def get_p_lsh_candidate(jsim, n_bands, n_rows_per_band):
    """Computes probability that two signatures will be detected as similar using the
    LSH algorithm.

    Args:
        jsim: Actuall/Assumed Jaccard Similarity between the hypothetical signatures
        n_bands: Number of bands
        n_rows_per_band: Number of rows per band

    Returns:
        (float) Probability that the hypothetical signatures will be detected as
            similar.
    """
    return 1 - (1 - jsim ** n_rows_per_band) ** n_bands


def get_jsim_and_approx_jsim(top_similar_index, hash_sets, signatures):
    """Adds Jaccard and approximate (based on signatures) Jaccard similarity to a
    "top similar index"

    Args:
        top_similar_index ({<signature index>: [<similar signature index>, ...]}):
            See output of LocalitySensitiveHashing.get_top_similar_index.
        hash_sets ([{int}]): The original hash sets
        signatures ([[int]]): Signatures computed by MinHash algorithm.

    Returns:
        ({<signature index>: ([
            (<similar signature index>, {'jsim': float, 'jsim_approx': float),
            ...
        ]})
    """
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


def get_approximation_diffs(similarities_index):
    """Computes the (flat) array of differences between Jaccard and estimated Jaccard
    similarity based on output of `get_jsim_and_approx_jsim` above.

    Args:
        similarities_index: See output of `get_jsim_and_approx_jsim` above.

    Returns:
        np.array(shape=(<total number of similarities detected>, )])
    """
    flat_sims = sum(similarities_index.values(), [])
    true_jsim = np.array([s[1]['jsim'] for s in flat_sims])
    approx_jsim = np.array([s[1]['jsim_approx'] for s in flat_sims])
    diffs = approx_jsim - true_jsim

    return diffs
