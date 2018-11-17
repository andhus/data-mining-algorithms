from __future__ import print_function, division

from warnings import warn

from tqdm import tqdm
from collections import defaultdict
import itertools


def get_frequent_sets_counts(
    item_sets,
    size,
    min_support=0.01,
):
    if size == 1:
        counts = defaultdict(lambda: 0)
        num_item_sets = 0
        for item_set in tqdm(item_sets, 'Extracting candidates of size 1'):
            num_item_sets += 1
            for item in item_set:
                counts[item] += 1
        min_count = min_support * num_item_sets
        if min_count <= 1:
            warn('min_count = {} <= 1'.format(min_count))
        counts = {
            (item,): count for item, count in counts.iteritems()
            if count >= min_count
        }
        return [counts], min_count

    subsets_counts, min_count = get_frequent_sets_counts(
        item_sets,
        size=size - 1,
        min_support=min_support
    )
    counts = defaultdict(lambda: 0)
    candidate_sets = set(get_one_larger_supersets(subsets_counts[-1].keys()))
    desc = 'Extracting candidates of size {}'.format(size)
    for item_set in tqdm(item_sets, desc):
        for subset in get_subsets(item_set, size=size):
            if subset in candidate_sets:
                counts[subset] += 1
    counts = {
        set_: count for set_, count in counts.iteritems()
        if count >= min_count
    }
    subsets_counts.append(counts)

    return subsets_counts, min_count


def get_one_larger_supersets(subsets):
    # TODO this can be speed up!
    if len(subsets) == 0:
        return []
    size = len(subsets[0]) + 1
    union = sorted(set(sum(subsets, tuple())))
    subsets = set(subsets)
    supersets = []
    desc = 'checking support for superset fo size {}'.format(size)
    for candidate_superset in tqdm(get_subsets(union, size), desc):
        all_subsets_exists = True
        for subset in get_subsets(candidate_superset, size - 1):
            if subset not in subsets:
                all_subsets_exists = False
                break
        if all_subsets_exists:
            supersets.append(candidate_superset)

    return supersets

# TODO compare alternative approach to find supersets, something like...
# def get_one_larger_supersets_2(subsets):
#     if len(subsets) == 0:
#         return []
#     size = len(subsets[0])
#     one_smaller_to_left_out = defaultdict(lambda: [])
#     for subset in subsets:
#         subsubsets = get_subsets(subset, size - 1)
#         for subsubset in subsubsets:
#             one_smaller_to_left_out[subsubset].append()
# ...


def get_subsets(item_set, size):
    return itertools.combinations(item_set, size)


def get_subsets_with_complement(item_set, size):
    return (
        (subset, tuple(sorted(set(item_set) - set(subset))))
        for subset in get_subsets(item_set, size)
    )


def get_rules(subsets_counts, min_confidence):
    """
    Args:
        subsets_counts: [
            {(int, ): int},
            {(int, int): int},
            {(int, int, int): int}
            ...
        ]
        min_confidence: float \in [0, 1]

    Returns: {(int, ...): [((int, ...), float), ...]}
    """
    rules = defaultdict(lambda: [])
    for rule_set_size in range(2, len(subsets_counts) + 1):
        for rule_set in subsets_counts[rule_set_size - 1]:
            for sub_size in range(1, rule_set_size):
                for subset, complement in get_subsets_with_complement(
                    rule_set,
                    sub_size
                ):
                    confidence = (
                        subsets_counts[rule_set_size - 1][rule_set] /
                        subsets_counts[sub_size - 1][subset]
                    )
                    if confidence > min_confidence:
                        rules[subset].append((complement, confidence))
    return rules
