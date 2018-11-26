"""Runs the three different versions of TRIEST; Base, Improved and Fully
Dynamic (FD), on a synthetic dataset. Their estimation accuracy and variance is
compared to each other and the theoretical limits for the two first versions.

The dataset is generated by creating a fully connected (undirected) graph (all nodes
connected to all others) and then create an edge stream by sample a subset of the
edges.
"""
from __future__ import print_function, division

import random
from itertools import combinations

from tqdm import tqdm
import numpy as np

from data_mining.graph import (
    TriestBase,
    TriestImpr,
    TriestFD
)
from data_mining.script_utils import get_trieste_args_parser


if __name__ == '__main__':
    arg_parser = get_trieste_args_parser(
        reservoir_size=1000,
        limit=5000,
        seed=123,
        description=__doc__
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

    base = TriestBase(size=args.reservoir_size, seed=args.seed)
    impr = TriestImpr(size=args.reservoir_size, seed=args.seed)
    fd = TriestFD(size=args.reservoir_size, seed=args.seed)
    num_triangles_est_base = []
    num_triangles_est_impr = []
    num_triangles_est_fd = []
    for _ in tqdm(range(args.num_estimates), desc='Running repeated estimations'):
        base.reset()
        base(edges[:args.limit])
        num_triangles_est_base.append(base.get_estimated_num_triangles())

        impr.reset()
        impr(edges[:args.limit])
        num_triangles_est_impr.append(impr.get_estimated_num_triangles())

        fd.reset()
        fd([(TriestBase.ADD, edge) for edge in edges[:args.limit]])
        num_triangles_est_fd.append(fd.get_estimated_num_triangles())

    print('Computing theoretical variance for TriesteBase the given dataset...')
    theoretical_variance_base = TriestBase.get_variance(
        t=base.t,
        reservoir_size=base.size,
        xi_t=base.xi,
        num_triangles_t=true_num_tri,
        r_t=tb_reference.reservoir.get_r()
    )
    print('Computing theoretical variance for TriesteBase the given dataset...')
    theoretical_variance_ub_impr = TriestImpr.get_variance_upper_bound(
        t=base.t,
        reservoir_size=base.size,
        num_triangles_t=true_num_tri,
        r_t=tb_reference.reservoir.get_r()
    )

    print("True number of triangles: {}".format(true_num_tri))

    def print_results(name, num_triangles_est):
        est_mean = np.mean(num_triangles_est)
        est_std = np.std(num_triangles_est)
        print('\n---- {} ----'.format(name))
        print("Typical estimated number of triangles (first 3 runs): {}".format(
            num_triangles_est[:3]))
        print("Mean of estimates: {}, off by {}%".format(
            est_mean, round(est_mean / true_num_tri - 1, 3) * 100))
        print("Standard deviation of estimation over {} runs: {}".format(
            args.num_estimates,  est_std))

    print_results('TriestBase', num_triangles_est_base)
    print("theoretical standard deviation of estimation: {}".format(
        np.sqrt(theoretical_variance_base)))

    print_results('TriestImpr', num_triangles_est_impr)
    print("theoretical UPPER BOUND of standard deviation of estimation: {}".format(
        np.sqrt(theoretical_variance_ub_impr)))

    print_results('TriestFD', num_triangles_est_fd)
