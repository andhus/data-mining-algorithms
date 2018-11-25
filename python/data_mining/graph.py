from __future__ import print_function, division

import math
import random
from collections import defaultdict
from itertools import combinations

import numpy as np
from data_mining.stream import ReservoirSampling
from tqdm import tqdm


def binomial(n, k):
    if k == n:
        return 1
    if k == 1:
        return n
    if k > n:
        return 0
    return math.factorial(n) // (
        math.factorial(k) * math.factorial(n - k)
    )


class UndirectedGraph(object):

    def __init__(self):
        self.node_neighbors = defaultdict(lambda: set())
        self.edges = set([])

    def put_edge(self, edge):
        u, v = edge
        if v in self.node_neighbors[u]:
            raise ValueError('edge {} already exists'.format(edge))
        self.node_neighbors[u].add(v)
        self.node_neighbors[v].add(u)
        self.edges.add(edge)

    def pop_edge(self, edge):
        if edge not in self.edges:
            raise ValueError('')
        u, v = edge
        self.node_neighbors[u].remove(v)
        self.node_neighbors[v].remove(u)
        self.edges.remove(edge)

    def put_node(self, node):
        if node in self.node_neighbors:
            raise ValueError('node {} exists'.format(node))
        _ = self.node_neighbors[node]

    def pop_node(self, node):
        raise NotImplementedError('')

    @property
    def nodes(self):
        return sorted(self.node_neighbors.keys())

    @property
    def num_edges(self):
        return len(self.edges)

    @property
    def num_nodes(self):
        return len(self.node_neighbors)

    def get_neighbors(self, node):
        return self.node_neighbors.get(node, None)


class EdgeReservoir(UndirectedGraph):

    def pop_replace_random_edge(self, edge):
        replace_idx = np.random.randint(0, self.num_edges)
        removed_edge = self.edges[replace_idx]
        self.edges[replace_idx] = edge
        return removed_edge

    def __len__(self):
        return self.num_edges

    def get_r(self, node=None):
        """The number of unordered pairs of distinct triangles (including node)
        sharing an edge

             3
           / | \
          4--0--1
             | /
             2
        """
        if node is not None:
            shared = 0
            for neig in self.get_neighbors(node):
                num_tri = 0
                for neig_neig in self.get_neighbors(neig) - {neig}:
                    if node in self.get_neighbors(neig_neig):
                        num_tri += 1
                if num_tri > 1:
                    shared += binomial(num_tri, 2)

            return shared

        # iterate over all nodes
        rs = []
        for node in tqdm(self.nodes):
            rs.append(self.get_r(node))
        agg = sum(rs)
        assert agg % 2 == 0  # note says 3 in paper!?
        return agg // 2

    def get_r_pairs(self, node):
        pairs = set([])  # TODO no need for set?
        for neig in self.get_neighbors(node):
            tris = []
            for neig_neig in self.get_neighbors(neig) - {neig}:
                if node in self.get_neighbors(neig_neig):
                    tris.append(tuple(sorted((neig, neig_neig))))
            if len(tris) > 1:
                pairs.update(combinations(tris, 2))

        return pairs


def with_probability(p):
    return np.random.random() < p


class TriestBase(ReservoirSampling):

    REMOVE = -1
    ADD = 1

    def __init__(self, **kwargs):
        self.tau = None
        self.tau_node = None
        if 'seed' in kwargs:
            random.seed(kwargs['seed'])
        super(TriestBase, self).__init__(**kwargs)

    def put(self, edge):
        self.t += 1
        if self.t <= self.size:
            self.reservoir.put_edge(edge)
            self.update_counters(self.ADD, edge)
        elif with_probability(self.size / self.t):
            remove_edge = random.sample(self.reservoir.edges, 1)[0]
            self.reservoir.pop_edge(remove_edge)
            self.update_counters(self.REMOVE, remove_edge)
            self.reservoir.put_edge(edge)
            self.update_counters(self.ADD, edge)
        else:
            pass

    def update_counters(self, operation, edge):
        v, u = edge
        u_neigh = self.reservoir.get_neighbors(u)
        v_neigh = self.reservoir.get_neighbors(v)
        shared_neighborhood = set(u_neigh.intersection(v_neigh))
        for shared_neighbor in shared_neighborhood:
            self.tau += operation
            self.tau_node[shared_neighbor] += operation
            self.tau_node[v] += operation
            self.tau_node[u] += operation

    def reset(self):
        super(TriestBase, self).reset()
        self.reservoir = EdgeReservoir()
        self.tau = 0
        self.tau_node = defaultdict(lambda: 0)

    @property
    def xi(self):
        t = self.t
        M = self.size
        return max(1, t * (t - 1) * (t - 2) / (M * (M - 1) * (M - 2)))

    def get_estimated_num_triangles(self, node=None):
        if node is not None:
            return self.xi * self.tau_node[node]
        return self.xi * self.tau


def get_variance(t, M, xi_t, num_triangles_t, r_t):
    print('num triangles: {}'.format(num_triangles_t))
    print('r_t: {}'.format(r_t))
    f = xi_t - 1
    print('f: {}'.format(f))
    g = xi_t * (M - 3) * (M - 4) / (
        (t - 3) * (t - 4)
    ) - 1
    print('g: {}'.format(g))
    h = xi_t * (M - 3) * (M - 4) * (M - 5) / (
        (t - 3) * (t - 4) * (t - 5)
    ) - 1
    print('h: {}'.format(h))
    w = binomial(num_triangles_t, 2) - r_t
    print('w: {}'.format(w))

    var = (
        num_triangles_t * f +
        r_t * g +
        w * h
    )

    return var
