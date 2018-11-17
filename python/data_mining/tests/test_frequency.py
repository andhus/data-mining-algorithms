from __future__ import print_function, division

from nose.tools import assert_equal


def test_get_subsets():
    from data_mining.frequency import get_subsets

    def verify(item_set, size, expected_subsets):
        subsets = list(get_subsets(item_set, size=size))
        assert_equal(subsets, expected_subsets)

    item_set = (1, 2, 3, 4)

    test_cases = [
        (item_set, 2, [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]),
        (item_set, 3, [(1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4)]),
        (item_set, 4, [(1, 2, 3, 4)])
    ]
    for item_set, size, expected_subsets in test_cases:
        yield verify, item_set, size, expected_subsets


def test_get_subsets_with_complement():
    from data_mining.frequency import get_subsets_with_complement

    def verify(item_set, size, expected_subsets_wc):
        subsets_wc = list(get_subsets_with_complement(item_set, size=size))
        assert_equal(subsets_wc, expected_subsets_wc)

    item_set = (1, 2, 3, 4)

    test_cases = [
        (
            item_set,
            2,
            [
                ((1, 2), (3, 4)),
                ((1, 3), (2, 4)),
                ((1, 4), (2, 3)),
                ((2, 3), (1, 4)),
                ((2, 4), (1, 3)),
                ((3, 4), (1, 2))
            ]
        ),
        (
            item_set,
            3,
            [
                ((1, 2, 3), (4,)),
                ((1, 2, 4), (3,)),
                ((1, 3, 4), (2,)),
                ((2, 3, 4), (1,))
            ]
        ),
        (
            item_set,
            4,
            [
                ((1, 2, 3, 4), tuple([]))
            ]
        )
    ]
    for item_set, size, expected_subsets in test_cases:
        yield verify, item_set, size, expected_subsets


def test_get_one_larger_supersets():
    from data_mining.frequency import get_one_larger_supersets

    def verify(subsets, expected_supersets):
        supersets = get_one_larger_supersets(subsets)
        assert_equal(supersets, expected_supersets)

    test_cases = [
        (
            [(1, 2), (1, 3), (2, 3)],
            [(1, 2, 3)]
        ),
        (
            [(1, 2), (1, 3), (2, 3), (1, 4)],
            [(1, 2, 3)]
        ),
        (
            [(1, 2), (1, 3), (2, 3), (1, 4), (2, 4)],
            [(1, 2, 3), (1, 2, 4)]
        )
    ]
    for subsets, expected_supersets in test_cases:
        yield verify, subsets, expected_supersets


def test_get_frequent_sets():
    from data_mining.frequency import get_frequent_sets_counts

    def verify(
        item_sets, size, min_support,
        expected_frequent_sets, expected_min_count
    ):
        frequent_sets, min_count = get_frequent_sets_counts(
            item_sets, size=size, min_support=min_support)
        assert_equal([dict(fs) for fs in frequent_sets], expected_frequent_sets)
        assert_equal(min_count, expected_min_count)

    item_sets = [
        (1, 2, 3, 4, 5),
        (1, 2, 3),
        (1, 2, 4),
        (1, 2),
    ]

    test_cases = [
        (
            item_sets, 1, 0.,
            [
                {(1,): 4, (2,): 4, (3,): 2, (4,): 2, (5,): 1}
            ],
            0
        ),
        (
            item_sets, 1, 0.3,
            [
                {(1,): 4, (2,): 4, (3,): 2, (4,): 2}
            ],
            0.3 * 4
        ),
        (
            item_sets, 2, 0.3,
            [
                {(1,): 4, (2,): 4, (3,): 2, (4,): 2},
                {(1, 2): 4, (1, 3): 2, (1, 4): 2, (2, 3): 2, (2, 4): 2}
            ],
            0.3 * 4
        ),
        (
            item_sets, 3, 0.3,
            [
                {(1,): 4, (2,): 4, (3,): 2, (4,): 2},
                {(1, 2): 4, (1, 3): 2, (1, 4): 2, (2, 3): 2, (2, 4): 2},
                {(1, 2, 3): 2, (1, 2, 4): 2}
            ],
            0.3 * 4
        ),
    ]

    for (
        item_sets, size, min_support,
        expected_frequent_sets, expected_min_count
    ) in test_cases:
        yield (
            verify,
            item_sets, size, min_support,
            expected_frequent_sets, expected_min_count
        )


def test_get_rules():
    from data_mining.frequency import get_rules

    def verify(subset_counts, confidence, expected_rules):
        rules = get_rules(subset_counts, confidence)
        assert_equal(rules, expected_rules)

    test_cases = [
        (
            [
                {(1,): 10, (2,): 20},
                {(1, 2): 5}
            ],
            0.,
            {
                (1,): [((2,), 0.5)],
                (2,): [((1,), 0.25)],
            }
        ),
        (
            [
                {(1,): 10, (2,): 20},
                {(1, 2): 5}
            ],
            0.3,
            {
                (1,): [((2,), 0.5)],
            }
        ),
        (
            [
                {(1,): 10, (2,): 20, (3,): 8},
                {(1, 2): 5, (1, 3): 4}
            ],
            0.3,
            {
                (1,): [((2,), 0.5), ((3,), 0.4)],
                (3,): [((1,), 0.5)]
            }
        ),
        (
            [
                {(1,): 10, (2,): 20, (3,): 8},
                {(1, 2): 5, (1, 3): 4, (2, 3): 3},
                {(1, 2, 3): 1}
            ],
            0.,
            {
                (1,): [
                    ((2,), 0.5),
                    ((3,), 0.4),
                    ((2, 3), 0.1)
                ],
                (2,): [
                    ((1,), 0.25),
                    ((3,), 0.15),
                    ((1, 3), 0.05)
                ],
                (3,): [
                    ((1,), 0.5),
                    ((2,), 3. / 8),
                    ((1, 2), 1. / 8),
                ],
                (1, 2): [
                    ((3,), 1. / 5)
                ],
                (1, 3): [
                    ((2,), 1. / 4)
                ],
                (2, 3): [
                    ((1,), 1. / 3)
                ]
            }
        )
    ]
    for subset_counts, confidence, expected_rules in test_cases:
        yield verify, subset_counts, confidence, expected_rules
