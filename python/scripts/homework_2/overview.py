from __future__ import print_function, division

import argparse
from pprint import pprint

from data_mining.data import get_frequent_itemset
from data_mining.set_frequency import get_frequent_sets_counts


def get_argument_parser(
    num_baskets_limit=None,
    set_size=5,
    min_support=0.005
):
    """Helper method for argument parsing for MinHash -> LSH script."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num-baskets-limit',
        type=int,
        default=num_baskets_limit,
        help='the number of baskets (first n) to analyse')
    parser.add_argument(
        '--set-size',
        type=int,
        default=set_size,
        help='(up to) size of frequent sets to identify'
    )
    parser.add_argument(
        '--min-support',
        type=float,
        default=min_support,
        help='minimum support of sets to identify'
    )
    return parser


if __name__ == '__main__':
    parser = get_argument_parser()
    args = parser.parse_args()
    print(args)
    baskets = get_frequent_itemset(
        first_n=args.num_baskets_limit
    )
    frequent_sets, min_count = get_frequent_sets_counts(
        baskets,
        size=args.set_size,
        min_support=args.min_support
    )
    pprint(frequent_sets[-1])
