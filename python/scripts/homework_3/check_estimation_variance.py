from __future__ import print_function, division

import random
from itertools import combinations

from tqdm import tqdm
import numpy as np

from data_mining.graph import TriestBase
from data_mining.graph import get_variance


if __name__ == '__main__':
    seed = None
    random.seed(seed)

    nodes = range(200)
    edges = list(combinations(nodes, 2))
    random.shuffle(edges)

    tb_reference = TriestBase(size=len(edges))
    tb_reference(edges[:4000])

    tb = TriestBase(size=1000, seed=seed)
    num_triangles_est = []
    for _ in tqdm(range(50)):
        tb.reset()
        tb(edges[:4000])
        num_triangles_est.append(tb.get_estimated_num_triangles())

    ref_num = tb_reference.get_estimated_num_triangles()
    mean = np.mean(num_triangles_est)
    std = np.std(num_triangles_est)

    theoretical_variance = get_variance(
        t=tb.t,
        M=tb.size,
        xi_t=tb.xi,
        num_triangles_t=ref_num,
        r_t=tb_reference.reservoir.get_r()
    )
