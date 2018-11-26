"""
"""
from __future__ import print_function, division

import random
from itertools import combinations

from tqdm import tqdm
import numpy as np

from data_mining.graph import (
    TriestBase,
    TriestImpr
)
from data_mining.script_utils import get_trieste_args_parser


if __name__ == '__main__':
    arg_parser = get_trieste_args_parser(
        reservoir_size=1000,
        limit=5000,
        seed=123
    )
    arg_parser.add_argument(
        '--num-nodes',
        type=int,
        default=200,
        help='the number of nodes in the graph'
    )
    arg_parser.add_argument(
        '--num-estimates',
        type=int,
        default=100,
        help='the number of estimates to run'
    )

    args = arg_parser.parse_args()

    print("Verifying estimation variance for: {}".format(args))

    # generate data
    nodes = range(args.num_nodes)
    edges = list(combinations(nodes, 2))
    if args.limit is not None and len(edges) < args.limit:
        raise ValueError('binomial(num_nodes, 2) must be > limit')
    random.shuffle(edges)
    print('Sampling {} edges (the given limit) out of {} possible for the fully '
          'connected graph'.format(args.limit, len(edges)))

    # create reference reservoir that has capacity for the full graph
    tb_reference = TriestBase(size=len(edges))  # seed does not matter
    tb_reference(edges[:args.limit])

    true_num_tri = tb_reference.get_estimated_num_triangles()

    tb = TriestBase(size=args.reservoir_size, seed=args.seed)
    ti = TriestImpr(size=args.reservoir_size, seed=args.seed)
    num_triangles_est_tb = []
    num_triangles_est_ti = []
    for _ in tqdm(range(args.num_estimates), desc='Running repeated estimations'):
        tb.reset()
        tb(edges[:args.limit])
        num_triangles_est_tb.append(tb.get_estimated_num_triangles())

        ti.reset()
        ti(edges[:args.limit])
        num_triangles_est_ti.append(ti.get_estimated_num_triangles())

    est_mean_tb = np.mean(num_triangles_est_tb)
    est_std_tb = np.std(num_triangles_est_tb)

    est_mean_ti = np.mean(num_triangles_est_ti)
    est_std_ti = np.std(num_triangles_est_ti)


    print('Computing theoretical variance for TriesteBase the given dataset...')
    theoretical_variance_tb = TriestBase.get_variance(
        t=tb.t,
        reservoir_size=tb.size,
        xi_t=tb.xi,
        num_triangles_t=true_num_tri,
        r_t=tb_reference.reservoir.get_r()
    )
    print('Computing theoretical variance for TriesteBase the given dataset...')
    theoretical_variance_ub_ti = TriestImpr.get_variance_upper_bound(
        t=tb.t,
        reservoir_size=tb.size,
        num_triangles_t=true_num_tri,
        r_t=tb_reference.reservoir.get_r()
    )

    print("True number of triangles: {}".format(true_num_tri))

    print('\n---- TriestBase ----')
    print("Typical estimated number of triangles (first run): {}".format(
        num_triangles_est_tb[0]))
    print("Mean of estimates: {}, off by {}%".format(
        est_mean_tb, round(est_mean_tb / true_num_tri - 1, 3) * 100))
    print("Standard deviation of estimation over {} runs: {}".format(
        args.num_estimates,  est_std_tb))
    print("theoretical standard deviation of estimation: {}".format(
        np.sqrt(theoretical_variance_tb)))

    print('\n---- TriestImpr ----')
    print("Typical estimated number of triangles (first run): {}".format(
        num_triangles_est_ti[0]))
    print("Mean of estimates: {}, off by {}%".format(
        est_mean_tb, round(est_mean_ti / true_num_tri - 1, 3) * 100))
    print("Standard deviation of estimation over {} runs: {}".format(
        args.num_estimates, est_std_ti))
    print("theoretical UPPER BOUND of standard deviation of estimation: {}".format(
        np.sqrt(theoretical_variance_ub_ti)))
