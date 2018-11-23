from __future__ import print_function, division

from collections import defaultdict
from contextlib import contextmanager

import numpy as np
from data_mining.stream import ReservoirSampling


class UndirectedGraph(object):

    def __init__(self):
        self.node_neighbors = defaultdict(lambda: set())
        self.edges = []

    def put_edge(self, edge):
        u, v = edge
        if v in self.node_neighbors[u]:
            raise ValueError('edge {} already exists'.format(edge))
        self.node_neighbors[u].add(v)
        self.node_neighbors[v].add(u)
        self.edges.append(edge)

    def pop_edge(self, edge):
        raise NotImplementedError('')

    def put_node(self, node):
        if node in self.node_neighbors:
            raise ValueError('node {} exists'.format(node))
        _ = self.node_neighbors[node]

    def pop_node(self, node):
        raise NotImplementedError('')

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


def with_probability(p):
    return np.random.random() < p


class TriestBase(ReservoirSampling):

    REMOVE = -1
    ADD = 1

    def __init__(self):
        self.tau = None
        self.tau_node = None
        super(TriestBase, self).__init__()

    def put(self, edge):
        self.t += 1
        if self.t < self.size:
            self.reservoir.put_edge(edge)
            self.update_counters(self.ADD, edge)
        elif with_probability(self.size / self.t):
            removed_edge = self.reservoir.pop_replace_random_edge(edge)
            self.update_counters(self.REMOVE, removed_edge)
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
        super(ReservoirSampling, self).reset()
        self.reservoir = EdgeReservoir()
        self.tau = 0
        self.tau_node = defaultdict(lambda: 0)

    @property
    def _xi(self):
        t = self.t
        M = self.size
        return max(1, t * (t - 1) * (t - 2) / (M * (M - 1) * (M - 2)))

    def get_expected_num_triangles(self, node=None):
        if node is not None:
            return self._xi * self.tau_node[node]
        return self._xi * self.tau
