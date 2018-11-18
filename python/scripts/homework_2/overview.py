from __future__ import print_function, division

from pprint import pprint

from data_mining.data import get_frequent_itemset
from data_mining.frequency import get_frequent_sets_counts, get_rules
from data_mining.script_utils import get_frequent_itemsets_argument_parser


if __name__ == '__main__':
    parser = get_frequent_itemsets_argument_parser(
        num_baskets_limit=None,
        max_set_size=None,
        min_support=0.005,
        min_confidence=0.10,
        print_limit=10
    )
    args = parser.parse_args()
    print('Arguments: {}'.format(args))
    baskets = get_frequent_itemset(first_n=args.num_baskets_limit)

    print('Extracting frequent item sets...')
    frequent_sets, min_count = get_frequent_sets_counts(
        baskets,
        max_size=args.max_set_size,
        min_support=args.min_support
    )
    print('\nDone, found following sets (with respective support)')
    for i, set_counts in enumerate(frequent_sets):
        print('found {} sets of size {}'.format(len(set_counts), i + 1))
        pprint(dict(set_counts.items()[:args.print_limit]))
        print('...')

    print('\nExtracting rules...')
    rules = get_rules(frequent_sets, min_confidence=args.min_confidence)
    print('Done, found {} rules with confidence >= {}'.format(
        len(rules), args.min_confidence))
    pprint(dict(rules.items()[:args.print_limit]))
    print('...')
