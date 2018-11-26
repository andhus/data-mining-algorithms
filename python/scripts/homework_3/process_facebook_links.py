from __future__ import print_function, division


from data_mining.graph import TriestBase
from data_mining.data import get_facebook_links
from data_mining.script_utils import get_trieste_args_parser


if __name__ == '__main__':
    arg_parser = get_trieste_args_parser(
        reservoir_size=10000,
        limit=None,
        seed=123
    )
    args = arg_parser.parse_args()
    print('Processing Facebook Links dataset, with args: {}'.format(args))

    # load data (iterator)
    fbl = get_facebook_links(unique=True)

    print('Stream contains {} links'.format(len(fbl)))
    # create reference reservoir that has capacity for the full graph
    tb_reference = TriestBase(size=len(fbl))
    print('Processing edge stream exactly for reference...')
    tb_reference(fbl)

    tb = TriestBase(size=args.reservoir_size)
    print('Processing edge stream for estimation...')
    tb(fbl)

    true_number_triangles = tb_reference.get_estimated_num_triangles()
    est_number_triangles = tb.get_estimated_num_triangles()

    print('True number of triangles: {}'.format(true_number_triangles))
    print('Estimated number of triangles: {}'.format(est_number_triangles))
    print('Estimate off by {} %'.format(
        round(est_number_triangles/ true_number_triangles - 1, 3) * 100))
