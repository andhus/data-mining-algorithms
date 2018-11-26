"""Processes facebook links dataset:
http://socialnetworks.mpi-sws.mpg.de/data/facebook-links.txt.gz using one of the
TRIEST algorithm versions."""

from __future__ import print_function, division

from data_mining.graph import TriestBase, TriestFD, TriestImpr
from data_mining.data import get_facebook_links
from data_mining.script_utils import get_trieste_args_parser

alg_versions = {
    'base': TriestBase,
    'impr': TriestImpr,
    'fd': TriestFD
}

if __name__ == '__main__':
    arg_parser = get_trieste_args_parser(
        description=__doc__,
        reservoir_size=10000,
        limit=None,
        seed=123
    )
    arg_parser.add_argument(
        '--alg-version',
        type=str,
        choices=alg_versions.keys(),
        default='base',
        help='The version of the TRIEST algorithm to use.'
    )
    args = arg_parser.parse_args()
    print('Processing Facebook Links dataset, with args: {}'.format(args))

    # load data (iterator)
    fbl = get_facebook_links(first_n=args.limit, unique=True)

    print('Stream contains {} links'.format(len(fbl)))
    # create reference reservoir that has capacity for the full graph
    tb_reference = TriestBase(size=len(fbl))
    print('Processing edge stream exactly for reference...')
    tb_reference(fbl)

    alg = alg_versions[args.alg_version](size=args.reservoir_size, seed=args.seed)
    print('Processing edge stream for estimation...')
    alg(
        fbl if args.alg_version != 'fd' else
        ((TriestBase.ADD, edge) for edge in fbl)
    )

    true_number_triangles = tb_reference.get_estimated_num_triangles()
    est_number_triangles = alg.get_estimated_num_triangles()

    print('True number of triangles: {}'.format(true_number_triangles))
    print('Estimated number of triangles: {}'.format(est_number_triangles))
    print('Estimate off by {} %'.format(
        round(est_number_triangles/ true_number_triangles - 1, 3) * 100))
