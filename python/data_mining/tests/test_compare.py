from __future__ import print_function, division

from nose.tools import assert_equal


def test_get_shingles():
    from data_mining.compare import get_shingles as test_function

    def verify(test_name, text, k, expected, kwargs):
        shingles = test_function(text, k, **kwargs)
        assert_equal(shingles, expected)

    text = "abcde"
    test_cases = [
        ("basic", text, 3, ["abc", "bcd", "cde"], {}),
        ("len(text) = k", text, len(text), [text], {}),
        ("len(text) < k", text, len(text) + 1, [text + "_"], {"pad": "_"}),
    ]

    for test_name, text, k, expected, kwargs in test_cases:
        yield verify, test_name, text, k, expected, kwargs


def test_get_ordered_hash_set():
    from data_mining.compare import get_hash_set as test_function
    
    def verify(test_name, sequence, expected, kwargs):
        ordered_hash_set = test_function(sequence, **kwargs)
        assert_equal(ordered_hash_set, expected)
    
    def hash_f(e):
        mapping = dict(zip("abcd", range(4)))
        return mapping[e]

    test_cases = [
        ("basic", ["a", "d", "b"], {0, 1, 3}, {"hash_f": hash_f}),
        ("duplicates", ["a", "d", "b", "a"], {0, 1, 3}, {"hash_f": hash_f}),
    ]

    for test_name, sequence, expected, kwargs in test_cases:
        yield verify, test_name, sequence, expected, kwargs


def test_get_jaccard_similarity():
    from data_mining.compare import get_jaccard_similarity as test_function

    s = {0, 1, 2}
    t = {0, 2, 3}
    sim = test_function(s, t)
    sim_expected = 0.5
    assert_equal(sim, sim_expected)


class TestMinHashing(object):

    from data_mining.compare import MinHashing as test_class

    def test_basic(self):
        # test case according Example 3.8 in:
        #   Mining of Massive Datasets - Stanford InfoLab
        #   (http://infolab.stanford.edu/~ullman/mmds/book.pdf)

        hash_sets = [
            {0, 3},
            {2},
            {1, 3, 4},
            {0, 2, 3}
        ]
        hash_fs = [lambda x: (x + 1) % 5, lambda x: (x * 3 + 1) % 5]
        mh = self.test_class(n_rows=5, hash_fs=hash_fs)

        signatures = mh(hash_sets)
        expected_signatures = [[1, 0], [3, 2], [0, 0], [1, 0]]
        assert_equal(signatures, expected_signatures)

    def test_more(self):
        hash_sets = [
            {0, 3, 6, 9},
            {2, 3, 4},
            {1, 3, 2, 7, 9, 0},
            {0, 2, 3, 4}
        ]
        mh = self.test_class(n_rows=10, n_hash_fs=3)

        signatures = mh(hash_sets)


def test_get_estimated_jaccard_similarity_from_signatures():
    from data_mining.compare import (
        get_estimated_jaccard_similarity_from_signatures as test_function
    )
    s_sign = [1, 4, 6, 2, 1]
    t_sign = [1, 0, 6, 5, 3]
    estimated_jsim = test_function(s_sign, t_sign)
    expected_estimated_jsim = 2 / 5
    assert_equal(estimated_jsim, expected_estimated_jsim)
