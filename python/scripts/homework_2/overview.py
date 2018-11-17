from __future__ import print_function, division

import argparse
from pprint import pprint

from data_mining.data import get_frequent_itemset
from data_mining.frequency import get_frequent_sets_counts, get_rules


def get_argument_parser(
    num_baskets_limit=None,
    max_set_size=None,
    min_support=0.005,
    min_confidence=0.10
):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num-baskets-limit',
        type=int,
        default=num_baskets_limit,
        help='the number of (first n) baskets to analyse, leave empty for all')
    parser.add_argument(
        '--max-set-size',
        type=int,
        default=max_set_size,
        help='max size of frequent sets to identify, leave empty for all'
    )
    parser.add_argument(
        '--min-support',
        type=float,
        default=min_support,
        help='minimum support of sets and rules to identify'
    )
    parser.add_argument(
        '--min-confidence',
        type=float,
        default=min_confidence,
        help='minimum confidence for rules to identify'
    )

    return parser


if __name__ == '__main__':
    parser = get_argument_parser()
    args = parser.parse_args()
    print(args)
    baskets = get_frequent_itemset(first_n=args.num_baskets_limit)

    print('\nExtracting frequent item sets...')
    frequent_sets, min_count = get_frequent_sets_counts(
        baskets,
        max_size=args.max_set_size,
        min_support=args.min_support
    )
    print('Done, found following sets (with respective support)')
    pprint(frequent_sets)

    print('\nExtracting rules...')
    rules = get_rules(frequent_sets, min_confidence=args.min_confidence)
    print('Done, found following rules (with respective confidence)')
    pprint(rules)
