from __future__ import print_function, division

from nose.tools import assert_equal


def test_get_subsets():
    from data_mining.set_frequency import get_subsets
    item_set = (1, 2, 3, 4)

    subsets = list(get_subsets(item_set, size=2))
    expected_subsets = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    assert_equal(subsets, expected_subsets)

    subsets = list(get_subsets(item_set, size=3))
    expected_subsets = [(1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4)]
    assert_equal(subsets, expected_subsets)

    subsets = list(get_subsets(item_set, size=4))
    expected_subsets = [(1, 2, 3, 4)]
    assert_equal(subsets, expected_subsets)


def test_get_candidate_supersets():
    from data_mining.set_frequency import get_one_larger_supersets
    subsets = [(1, 2), (1, 3), (2, 3)]

    candidate_supersets = get_one_larger_supersets(subsets)
    expected_candidate_supersets = [(1, 2, 3)]
    assert_equal(candidate_supersets, expected_candidate_supersets)


def test_get_frequent_sets():
    from data_mining.set_frequency import get_frequent_sets_counts
    item_sets = [
        (1, 2, 3, 4, 5),
        (1, 2, 3),
        (1, 2, 4),
        (1, 2),
    ]

    frequent_sets, min_count = get_frequent_sets_counts(
        item_sets, size=1, min_support=0.)
    expected_frequent_sets = [{(1,): 4, (2,): 4, (3,): 2, (4,): 2, (5,): 1}]
    assert_equal(frequent_sets, expected_frequent_sets)
    assert_equal(min_count, 0.)

    frequent_sets, min_count = get_frequent_sets_counts(
        item_sets, size=1, min_support=0.3)
    expected_frequent_sets = [{(1,): 4, (2,): 4, (3,): 2, (4,): 2}]
    assert_equal(frequent_sets, expected_frequent_sets)
    assert_equal(min_count, 0.3 * 4)

    frequent_sets, min_count = get_frequent_sets_counts(
        item_sets, size=2, min_support=0.3)
    expected_frequent_sets = [
        {(1,): 4, (2,): 4, (3,): 2, (4,): 2},
        {(1, 2): 4, (1, 3): 2, (1, 4): 2, (2, 3): 2, (2, 4): 2}
    ]
    assert_equal([dict(fs) for fs in frequent_sets], expected_frequent_sets)
    assert_equal(min_count, 0.3 * 4)

    frequent_sets, min_count = get_frequent_sets_counts(
        item_sets, size=3, min_support=0.3)
    expected_frequent_sets = [
        {(1,): 4, (2,): 4, (3,): 2, (4,): 2},
        {(1, 2): 4, (1, 3): 2, (1, 4): 2, (2, 3): 2, (2, 4): 2},
        {(1, 2, 3): 2, (1, 2, 4): 2}
    ]
    assert_equal([dict(fs) for fs in frequent_sets], expected_frequent_sets)
    assert_equal(min_count, 0.3 * 4)
