from __future__ import print_function, division

from warnings import warn

from tqdm import tqdm
from collections import defaultdict
import itertools


def get_frequent_sets_counts(
    item_sets,
    max_size=None,
    min_support=0.01,
):
    """Extracts all set (optionally up to provided max size) with support above
    provided threshold, using the A Priori algorithm [1].

    Args:
        item_sets (Iterable(tuple(int))): Item sets to analyse. NOTE this arg must
            be an iterator that can be called several times - i.e. not a generator
            that is emptied at first pass.
        max_size (int | None): If provided generate sets up to this size only.
        min_support: (float): Minimum support of sets in percentage of total number.

    Returns:
        subsets_counts ([{(int, ): int}, {(int, int): int}, ...]): An array of
            dict:s where the subsets_counts[N-1] holds the mapping from set to count
            for sets with size N.

    References:
        [1] Fast Algorithms for Mining Association Rules, R. Agrawal R. Srikant
            http://www.vldb.org/conf/1994/P487.PDF
    """
    if not (max_size is None or max_size >= 1):
        raise ValueError('size must be >= 1 or None')

    # special handling of size one set
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
    subsets_counts = [counts]

    # sets with size >= 2 extracted based on previous sets
    size = 2
    while max_size is None or size <= max_size:
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
        if not counts:
            break  # no new larger sets were found
        subsets_counts.append(counts)
        size += 1

    return subsets_counts, min_count


def get_subsets(item_set, size):
    """Gets all the subsets of given size that can be formed from the itemset

    Args:
        item_set ((int, ...)): Set of items.
        size (int):

    Returns:
        [(int,...), ...] All subsets of size `size`
    """
    return itertools.combinations(item_set, size)


def get_subsets_with_complement(item_set, size):
    """Gets all the subsets of given size that can be formed from the itemset and
    their respective complement.

    Args:
        item_set ((int, ...)): Set of items.
        size (int):

    Returns:
        [((int,...), (int,)), ...] All subsets of size `size` and their respective
            complement.

    """
    return (
        (subset, tuple(sorted(set(item_set) - set(subset))))
        for subset in get_subsets(item_set, size)
    )


def get_one_larger_supersets(subsets):
    """Extracts all supersets that is one item larger than the provided subsets and
    for which *all subsets are present in subsets*.

    Args:
        subsets ([(int, ...), ...]): Iterable of subsets of which all should have
            the same size.

    Returns:
        [(int, ...), ...] All the formable supersets.
    """
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


def get_rules(subsets_counts, min_confidence):
    """Extracts all association rules given the counts of candidate item sets.

    Args:
        subsets_counts: ([
            {(int, ): int},
            {(int, int): int},
            {(int, int, int): int}
            ...
        ]) An array of dict:s where the subsets_counts[N-1] holds the mapping from
            set to count for sets with size N. I.e. the output from
            `get_frequent_sets_counts`.
        min_confidence (float): The minimum confidence for rules, float \in [0, 1]

    Returns: {(int, ...): [((int, ...), float), ...]} A mapping from item set to
        all other item sets an and the confidence of the corresponding association
        rule.
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
    return dict(rules)
