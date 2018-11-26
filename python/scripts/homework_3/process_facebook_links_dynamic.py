"""Processes a simulated dynamic stream based on the Facebook Links dataset:
http://socialnetworks.mpi-sws.mpg.de/data/facebook-links.txt.gz using the
TRIEST-FD (Fully Dynamic) algorithm. A given fraction of the added edges will also
be removed at a later point in time.
"""
from __future__ import print_function, division

import random

from data_mining.graph import (
    TriestBase,
    TriestFD
)
from data_mining.data import get_facebook_links
from data_mining.script_utils import get_trieste_args_parser


if __name__ == '__main__':
    arg_parser = get_trieste_args_parser(
        description=__doc__,
        reservoir_size=int(1e4),
        limit=int(1e5),
        seed=123
    )
    arg_parser.add_argument(
        '--fraction-remove',
        type=float,
        default=0.15,
        help='The fraction of links that will be removed again after being added.'
    )
    args = arg_parser.parse_args()
    print('Processing Facebook Links dataset, with args: {}'.format(args))

    # load data (iterator)
    fbl = get_facebook_links(first_n=args.limit, unique=True)

    # sample links to remove
    num_links = len(fbl)
    rm_num_links = int(num_links * args.fraction_remove)
    print('Stream contains {} links, of which {} will be removed'.format(
        num_links, rm_num_links))
    remove_edges = set(random.sample(fbl, rm_num_links))

    def iterate_removed():
        return (edge for edge in fbl if edge not in remove_edges)

    def iterate_dynamic_remove():
        to_remove = set([])
        fbl_list = list(fbl)
        i = 0
        while to_remove or i < len(fbl_list):
            if i > 0 and i % 50000 == 0:
                print('processed {} edges, remove cache size: {}'.format(
                    i, len(to_remove)))
            if random.random() < len(to_remove) / (len(fbl_list) - i + 1):
                edge = random.sample(to_remove, 1)[0]
                to_remove.remove(edge)
                yield (TriestBase.REMOVE, edge)
            else:
                edge = fbl_list[i]
                i += 1
                if edge in remove_edges:
                    to_remove.add(edge)
                yield (TriestBase.ADD, edge)

    # Uncomment below to verify implementation of iterators:
    # print('Verifying stream...')
    # remaining = set([])
    # for op, edge in iterate_dynamic_remove():
    #     if op == TriestBase.ADD:
    #         remaining.add(edge)
    #     elif op == TriestBase.REMOVE:
    #         remaining.remove(edge)
    #     else:
    #         raise ValueError('verification failed, non allowed op')
    # assert set(iterate_removed()) == remaning
    # print('verification ok!')

    # create reference reservoir that has capacity for the full graph
    tb_reference = TriestBase(size=len(fbl))
    print('Processing edge stream exactly for reference...')
    tb_reference(iterate_removed())

    tb = TriestFD(size=args.reservoir_size)
    print('Processing dynamic edge stream for estimation...')
    tb(iterate_dynamic_remove())

    true_number_triangles = tb_reference.get_estimated_num_triangles()
    est_number_triangles = tb.get_estimated_num_triangles()

    print('True number of triangles: {}'.format(true_number_triangles))
    print('Estimated number of triangles: {}'.format(est_number_triangles))
    print('Estimate off by {} %'.format(
        round(est_number_triangles / true_number_triangles - 1, 3) * 100))
