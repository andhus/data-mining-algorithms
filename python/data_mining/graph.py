from __future__ import print_function, division

import math
import random

from itertools import combinations
from collections import defaultdict

import numpy as np

from tqdm import tqdm

from data_mining.stream import ReservoirSampling


def binomial(n, k):
    """Computes the binomial coefficient "n over k".
    """
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
    """Represents an undirected graph as a set of edges and mapping from each node
    connected nodes.
    """
    def __init__(self):
        self._node_neighbors = defaultdict(lambda: set())
        self.edges = set([])

    def put_edge(self, edge):
        """Adds the edge to the graph.

        Args:
            edge ((int, int)): edge between nodes edge[0] and edge[1].
        """
        u, v = edge
        if v in self._node_neighbors[u]:
            raise ValueError('edge {} already exists'.format(edge))
        self._node_neighbors[u].add(v)
        self._node_neighbors[v].add(u)
        self.edges.add(edge)

    def pop_edge(self, edge, remove_disconnected_nodes=False):
        """Removes the edge from the graph.

        Args:
            edge ((int, int)): edge between nodes edge[0] and edge[1].
        """

        if edge not in self.edges:
            raise ValueError('edge {} does not exist'.format(edge))
        u, v = edge
        self._node_neighbors[u].remove(v)
        self._node_neighbors[v].remove(u)
        self.edges.remove(edge)
        if remove_disconnected_nodes:
            if len(self._node_neighbors[u]) == 0:
                del self._node_neighbors[u]
            if len(self._node_neighbors[v]) == 0:
                del self._node_neighbors[v]

    def put_node(self, node):
        """Adds a node to the graph.

        Args:
            node (int): the node (index) to add.
        """
        if node in self._node_neighbors:
            raise ValueError('node {} exists'.format(node))
        _ = self._node_neighbors[node]

    def pop_node(self, node):
        """Removes the node from the graph.

        Args:
            node (int): the node (index) to remove.
        """
        if node not in self._node_neighbors:
            raise ValueError('node: {} does not exist'.format(node))
        for neigh in self._node_neighbors[node]:
            self._node_neighbors[neigh].remove(node)
        del self._node_neighbors[node]

    def get_neighbors(self, node, **kwargs):
        """Gets the neighbours of the given node

        Args:
            node (int): the node (index) for which neighbors should be returned.

        Returns
            {int}
        """
        if node not in self._node_neighbors:
            if 'default' in kwargs:
                return kwargs['default']
            raise ValueError('node {} not in graph'.format(node))
        return self._node_neighbors[node]

    def __contains__(self, item):
        if isinstance(item, int):
            return item in self._node_neighbors
        if isinstance(item, tuple):
            return item in self.edges
        raise ValueError(
            'item must be either a node (int) or an edge '
            '(int, int), got {}'.format(item)
        )

    @property
    def nodes(self):
        """List of all nodes ordered by index"""
        return sorted(self._node_neighbors.keys())

    @property
    def num_edges(self):
        """Number of edges in the graph"""
        return len(self.edges)

    @property
    def num_nodes(self):
        """number of nodes in the graph"""
        return len(self._node_neighbors)


class EdgeReservoir(UndirectedGraph):
    """Represents the undirected "edge-reservoir" used in the TRIEST algorith in [1].

    Extends the UndirectedGraph by implementing __len__ so that it can be used as an
    reservoir in data_mining.stream.ReservoirSampling.

    References:
        [1] L. De Stefani, A. Epasto, M. Riondato, and E. Upfal, TRIEST: Counting
            Local and Global Triangles in Fully-Dynamic Streams with Fixed Memory
            Size, KDD'16
            https://www.kdd.org/kdd2016/papers/files/rfp0465-de-stefaniA.pdf
    """

    def __len__(self):
        return self.num_edges

    def get_r(self, node=None):
        """Computes the number of unordered pairs of distinct triangles sharing an
        edge (including node if specified, otherwise in the entire graph).

        This corresponds to r_{u}^(t) with u = `node`, or r^(t) with `node = None`
        in [1].

        In graph below there is e.g. 2 distinct unordered triangle-pairs sharing an
        edge and containing 0: (0-1-2, 0-1-3) and (0-1-3, 0-3-4) which means
        r_{0}^(t) = 2

        The total number of distinct unordered triangle-pairs sharing an
        edge is the same in this case, so r^(t) = 2.

             3
           / | \
          4--0--1
             | /
             2

        Args:
            node (int): if specified


        NOTE that there is an error in [1] regarding this computation; it says that:

            r^(t) = 1/3 * sum_{u \in V^(t)} r_{u}^(t)    [1], p.3

        However, since each distinct triangle-pair sharing an edge will only be
        counted _twice_ (not three times) the sum should be divided by 2.
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
        pairs = set([])  # (no need for set)
        for neig in self.get_neighbors(node):
            tris = []
            for neig_neig in self.get_neighbors(neig) - {neig}:
                if node in self.get_neighbors(neig_neig):
                    tris.append(tuple(sorted((neig, neig_neig))))
            if len(tris) > 1:
                pairs.update(combinations(tris, 2))

        return pairs


def with_probability(p):
    """Flip biased coin.

    Example:
    ```python

    if with_probability(0.75):
        # do the thing that should be done with 75% probability

    ```
    """
    return np.random.random() < p


class TriestBase(ReservoirSampling):
    """Implementation of the TRIEST-BASE algorith in [1].

    References:
        [1] L. De Stefani, A. Epasto, M. Riondato, and E. Upfal, TRIEST: Counting
            Local and Global Triangles in Fully-Dynamic Streams with Fixed Memory
            Size, KDD'16
            https://www.kdd.org/kdd2016/papers/files/rfp0465-de-stefaniA.pdf

    Args:
        size (int): the size of the reservoir (M in [1]).
        seed: seed for random number generation.
    """

    # edge operations
    REMOVE = -1
    ADD = 1

    def __init__(self, **kwargs):
        self.tau = None
        self.tau_node = None
        if 'seed' in kwargs:
            random.seed(kwargs['seed'])  # base class init only seeds numpy.random
        super(TriestBase, self).__init__(**kwargs)

    def put(self, edge):
        """Process next item (graph edge) in the stream.

        Args:
            edge ((int, int)): an _added edge_ connecting nodes edge[0] and edge[1].

        Combination of main loop and `SampleEdge` function in [1]: Algorithm 1.
        """
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
        """`UpdateCounters` function in [1]: Algorithm 1."""
        v, u = edge
        u_neigh = self.reservoir.get_neighbors(u)
        v_neigh = self.reservoir.get_neighbors(v)
        shared_neighborhood = set(u_neigh.intersection(v_neigh))
        for shared_neighbor in shared_neighborhood:
            self.tau += operation
            self.tau_node[shared_neighbor] += operation
            self.tau_node[v] += operation
            self.tau_node[u] += operation

    def get_estimated_num_triangles(self, node=None):
        """Computes the estimated number of triangles.

        Args:
            node (int | None): the estimated number of triangles in the sub-graph
                containing this node is returned if specified, if None (default) for
                the entire graph.

        Returns:
            (float) Estimated number of triangles.
        """
        if node is not None:
            return self.xi * self.tau_node[node]
        return self.xi * self.tau

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

    @staticmethod
    def get_variance(t, reservoir_size, xi_t, num_triangles_t, r_t):
        """Computes the theoretical variance of the TRIEST BASE estimation.

        Args: According notation in [1].

        Returns:
            (float) Variance of estimation for given parameters.
        """
        M = reservoir_size
        f = xi_t - 1
        g = xi_t * (M - 3) * (M - 4) / (
            (t - 3) * (t - 4)
        ) - 1
        h = xi_t * (M - 3) * (M - 4) * (M - 5) / (
            (t - 3) * (t - 4) * (t - 5)
        ) - 1
        w = binomial(num_triangles_t, 2) - r_t
        var = (
            num_triangles_t * f +
            r_t * g +
            w * h
        )

        return var


class TriestImpr(TriestBase):
    """Implementation of the TRIEST-IMPR algorith in [1].

    References:
        [1] L. De Stefani, A. Epasto, M. Riondato, and E. Upfal, TRIEST: Counting
            Local and Global Triangles in Fully-Dynamic Streams with Fixed Memory
            Size, KDD'16
            https://www.kdd.org/kdd2016/papers/files/rfp0465-de-stefaniA.pdf

    Args:
        size (int): the size of the reservoir (M in [1]).
        seed: seed for random number generation.
    """
    def put(self, edge):
        """Process next item (graph edge) in the stream.

        Args:
            edge ((int, int)): an _added edge_ connecting nodes edge[0] and edge[1].

        Combination of main loop and `SampleEdge` function in [1]: Algorithm 1.
        """
        self.t += 1
        self.update_counters(self.ADD, edge)
        if self.t <= self.size:
            self.reservoir.put_edge(edge)
        elif with_probability(self.size / self.t):
            remove_edge = random.sample(self.reservoir.edges, 1)[0]
            self.reservoir.pop_edge(remove_edge)
            self.reservoir.put_edge(edge)
        else:
            pass

    @staticmethod
    def get_eta(t, reservoir_size):
        M = reservoir_size
        eta = max(1, (t - 1) * (t - 2) / (M * (M - 1)))
        return eta

    @property
    def eta(self):
        """The modification weight in UpdateCounter in TRIEST IMPR"""
        return self.get_eta(self.t, self.size)

    def update_counters(self, operation, edge):
        """Modified `UpdateCounters` function in [1]: Algorithm 1. according
        "4.2 Improved insertion algorithm"
        """
        v, u = edge
        u_neigh = self.reservoir.get_neighbors(u, default=set([]))
        v_neigh = self.reservoir.get_neighbors(v, default=set([]))
        shared_neighborhood = set(u_neigh.intersection(v_neigh))
        assert operation == self.ADD  # only add used in TriestImpr
        update = operation * self.eta
        for shared_neighbor in shared_neighborhood:
            self.tau += update
            self.tau_node[shared_neighbor] += update
            self.tau_node[v] += update
            self.tau_node[u] += update

    @property
    def xi(self):
        raise NotImplementedError('not relevant for TriestImpr')

    def get_estimated_num_triangles(self, node=None):
        """Computes the estimated number of triangles.

        "When queried for an estimation, triest-impr returns the value of the
        corresponding counter, unmodified." [1] p.3

        Args:
            node (int | None): the estimated number of triangles in the sub-graph
                containing this node is returned if specified, if None (default) for
                the entire graph.

        Returns:
            (float) Estimated number of triangles.
        """
        if node is not None:
            return self.tau_node[node]
        return self.tau

    @staticmethod
    def get_variance_upper_bound(t, reservoir_size, num_triangles_t, r_t):
        return (
            num_triangles_t * (TriestImpr.get_eta(t, reservoir_size) - 1) +
            r_t * (t - 1 - reservoir_size) / reservoir_size
        )

    @staticmethod
    def get_variance(t, reservoir_size, xi_t, num_triangles_t, r_t):
        raise NotImplementedError(
            'The exact variance of TriestImpr cannot be computed, '
            'see `get_variance_upper_bound` instead'
        )


class TriestFD(TriestBase):
    """Implementation of the TRIEST-FD algorith in [1].

    References:
        [1] L. De Stefani, A. Epasto, M. Riondato, and E. Upfal, TRIEST: Counting
            Local and Global Triangles in Fully-Dynamic Streams with Fixed Memory
            Size, KDD'16
            https://www.kdd.org/kdd2016/papers/files/rfp0465-de-stefaniA.pdf

    Args:
        size (int): the size of the reservoir (M in [1]).
        seed: seed for random number generation.
    """

    def put(self, (operation, edge)):
        """Process next item (graph edge) in the stream.

        Args:
            operation (int): Integer in {1, -1} for ADD, REMOVE respectively.
            edge ((int, int)): an _added edge_ connecting nodes edge[0] and edge[1].

        Combination of main loop and `SampleEdge` function in [1]: Algorithm 1.
        """
        self.t += 1
        self.s += operation
        if operation == self.ADD:
            if self.sample_edge(edge):
                self.update_counters(self.ADD, edge)
        elif edge in self.reservoir:
            self.update_counters(self.REMOVE, edge)
            self.reservoir.pop_edge(edge)
            self.d_i += 1
        else:
            self.d_o += 1

    def sample_edge(self, edge):
        if self.d_o + self.d_i == 0:
            if len(self.reservoir) < self.size:
                self.reservoir.put_edge(edge)
                return True
            if with_probability(self.size / self.t):
                remove_edge = random.sample(self.reservoir.edges, 1)[0]
                self.update_counters(self.REMOVE, remove_edge)
                self.reservoir.pop_edge(remove_edge)
                self.reservoir.put_edge(edge)
                return True
            # return False, or if below !?
        elif with_probability(self.d_i / (self.d_i + self.d_o)):
            self.reservoir.put_edge(edge)
            self.d_i -= 1
            return True
        else:
            self.d_o -= 1
            return False

    def get_estimated_num_triangles(self, node=None):
        if self.size < 3:
            return 0  # special case for completeness
        if node is not None:
            if node not in self.tau_node:
                return 0.
            tau = self.tau_node[node]
        else:
            tau = self.tau

        s = self.s
        M = self.size
        return tau / self.kappa * (
            s * (s - 1) * (s - 2) / (M * (M - 1) * (M - 2))
        )

    @property
    def kappa(self):
        d_o = self.d_o
        d_i = self.d_i
        s = self.s
        M = self.size
        omega = min(M, s + d_i + d_o)
        return 1 - sum([
            binomial(s, j) * binomial(d_i + d_o, omega - j) / (s + d_o + d_i)
            for j in range(3)
        ])

    def reset(self):
        super(TriestFD, self).reset()
        self.d_o = 0
        self.d_i = 0
        self.s = 0
