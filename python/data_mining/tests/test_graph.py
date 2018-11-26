from __future__ import print_function, division

import random

from itertools import combinations

import numpy as np

from nose.tools import (
    assert_equal,
    assert_raises,
    assert_less,
    assert_raises
)
from numpy.testing import assert_allclose


class GraphTests(object):
    test_class = None

    def test_put_edge(self):
        g = self.test_class()
        g.put_edge((0, 1))
        assert_equal(g.get_neighbors(0), {1})
        assert_equal(g.get_neighbors(1), {0})
        assert_equal(g.edges, {(0, 1)})
        assert_equal(g.num_edges, 1)
        assert_equal(g.num_nodes, 2)
        assert (0, 1) in g
        assert 0 in g
        assert 1 in g

        g.put_edge((0, 2))
        assert_equal(g.get_neighbors(0), {1, 2})
        assert_equal(g.get_neighbors(2), {0})
        assert_equal(g.get_neighbors(1), {0})
        assert_equal(g.edges, {(0, 1), (0, 2)})
        assert_equal(g.num_edges, 2)
        assert_equal(g.num_nodes, 3)

        g.put_edge((2, 1))
        assert_equal(g.get_neighbors(0), {1, 2})
        assert_equal(g.get_neighbors(2), {0, 1})
        assert_equal(g.get_neighbors(1), {0, 2})
        assert_equal(g.edges, {(0, 1), (0, 2), (2, 1)})
        assert_equal(g.num_edges, 3)
        assert_equal(g.num_nodes, 3)

    def test_put_existing_edge(self):
        g = self.test_class()
        g.put_edge((0, 1))
        assert_equal(g.get_neighbors(0), {1})
        assert_equal(g.get_neighbors(1), {0})
        with assert_raises(ValueError):
            g.put_edge((0, 1))
        assert_equal(g.get_neighbors(0), {1})
        assert_equal(g.get_neighbors(1), {0})
        with assert_raises(ValueError):
            g.put_edge((1, 0))
        assert_equal(g.get_neighbors(0), {1})
        assert_equal(g.get_neighbors(1), {0})

    def test_pop_edge(self):
        g = self.test_class()
        g.put_edge((0, 1))
        g.put_edge((0, 2))
        g.put_edge((1, 2))
        assert_equal(g.get_neighbors(0), {1, 2})
        assert_equal(g.get_neighbors(1), {0, 2})
        assert_equal(g.get_neighbors(2), {0, 1})

        g.pop_edge((1, 2))
        assert_equal(g.get_neighbors(0), {1, 2})
        assert_equal(g.get_neighbors(1), {0})
        assert_equal(g.get_neighbors(2), {0})

        g.pop_edge((0, 2))
        assert_equal(g.get_neighbors(0), {1})
        assert_equal(g.get_neighbors(1), {0})
        assert_equal(g.get_neighbors(2), set([]))

    def test_pop_edge_remove_disconnected_nodes(self):
        g = self.test_class()
        g.put_edge((0, 1))
        g.put_edge((0, 2))
        g.put_edge((1, 2))
        assert_equal(g.get_neighbors(0), {1, 2})
        assert_equal(g.get_neighbors(1), {0, 2})
        assert_equal(g.get_neighbors(2), {0, 1})

        g.pop_edge((1, 2), remove_disconnected_nodes=True)
        assert_equal(g.get_neighbors(0), {1, 2})
        assert_equal(g.get_neighbors(1), {0})
        assert_equal(g.get_neighbors(2), {0})

        g.pop_edge((0, 2), remove_disconnected_nodes=True)
        assert_equal(g.get_neighbors(0), {1})
        assert_equal(g.get_neighbors(1), {0})
        assert 2 not in g
        with assert_raises(ValueError):
            g.get_neighbors(2)

    def test_put_pop_node(self):
        g = self.test_class()
        g.put_edge((0, 1))
        g.put_edge((0, 2))
        g.put_edge((1, 2))
        assert_equal(g.get_neighbors(0), {1, 2})
        assert_equal(g.get_neighbors(1), {0, 2})
        assert_equal(g.get_neighbors(2), {0, 1})
        assert_equal(g.num_nodes, 3)

        g.put_node(4)
        assert_equal(g.get_neighbors(0), {1, 2})
        assert_equal(g.get_neighbors(1), {0, 2})
        assert_equal(g.get_neighbors(2), {0, 1})
        assert_equal(g.get_neighbors(4), set([]))
        assert_equal(g.num_nodes, 4)

        g.put_edge((0, 4))
        assert_equal(g.get_neighbors(0), {1, 2, 4})
        assert_equal(g.get_neighbors(4), {0})
        assert_equal(g.num_nodes, 4)

        g.pop_node(2)
        assert_equal(g.get_neighbors(0), {1, 4})
        assert 2 not in g
        with assert_raises(ValueError):
            g.get_neighbors(2)


class TestUndirectedGraph(GraphTests):
    from data_mining.graph import UndirectedGraph as test_class


class TestEdgeReservoir(GraphTests):
    from data_mining.graph import EdgeReservoir as test_class

    def test_get_r(self):
        g = self.test_class()
        for edge in [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (3, 4)]:
            g.put_edge(edge)
        #    3
        #  / | \
        # 4--0--1
        #    | /
        #    2
        assert_equal(g.get_r(0), 2)
        assert_equal(g.get_r(1), 1)
        assert_equal(g.get_r(2), 0)
        assert_equal(g.get_r(3), 1)
        assert_equal(g.get_r(4), 0)
        assert_equal(g.get_r(), 2)

        g.put_edge((2, 4))
        #    3
        #  / | \
        # 4--0--1
        #  \ | /
        #    2
        assert_equal(g.get_r(0), 4)
        assert_equal(g.get_r(1), 1)
        assert_equal(g.get_r(2), 1)
        assert_equal(g.get_r(3), 1)
        assert_equal(g.get_r(4), 1)
        assert_equal(g.get_r(), 4)


        g.put_edge((0, 5))
        #    3
        #  / | \
        # 4--0--1
        #  \ | /
        #    2    5 --{0}
        assert_equal(g.get_r(0), 4)
        assert_equal(g.get_r(), 4)

        g.put_edge((2, 5))
        #    3
        #  / | \
        # 4--0--1
        #  \ | /
        #    2 -- 5 --{0}
        assert_equal(g.get_r(0), 6)
        assert_equal(g.get_r(1), 1)
        assert_equal(g.get_r(2), 3)
        assert_equal(g.get_r(3), 1)
        assert_equal(g.get_r(4), 1)
        assert_equal(g.get_r(5), 0)
        assert_equal(g.get_r(0), 6)

        g.put_edge((1, 5))
        #    3
        #  / | \
        # 4--0--1
        #  \ | / \
        #    2 -- 5 --{0}
        assert_equal(g.get_r(0), 9)


class TriestTests():
    test_class = None

    def test_exact_at_t_less_than_size(self):
        tb = self.test_class(size=10)

        def assert_num_triangles(node, expected):
            assert_equal(tb.get_estimated_num_triangles(node=node), expected)

        tb([(0, 1), (0, 2)])
        #  0--1---4
        #  | / \ /
        #  2----3
        assert_num_triangles(None, 0)
        assert_num_triangles(0, 0)
        assert_num_triangles(1, 0)
        assert_num_triangles(2, 0)

        tb.put((1, 2))
        #  0--1---4
        #  | / \ /
        #  2----3
        assert_num_triangles(None, 1)
        assert_num_triangles(0, 1)
        assert_num_triangles(1, 1)
        assert_num_triangles(2, 1)

        tb.put((1, 3))
        #  0--1
        #  | / \
        #  2    3
        assert_num_triangles(None, 1)
        assert_num_triangles(0, 1)
        assert_num_triangles(1, 1)
        assert_num_triangles(2, 1)
        assert_num_triangles(3, 0)

        tb.put((2, 3))
        #  0--1
        #  | / \
        #  2----3
        assert_num_triangles(None, 2)
        assert_num_triangles(0, 1)
        assert_num_triangles(1, 2)
        assert_num_triangles(2, 2)
        assert_num_triangles(3, 1)

        tb.put((3, 4))
        #  0--1   4
        #  | / \ /
        #  2----3
        assert_num_triangles(None, 2)
        assert_num_triangles(0, 1)
        assert_num_triangles(1, 2)
        assert_num_triangles(2, 2)
        assert_num_triangles(3, 1)
        assert_num_triangles(4, 0)

        tb.put((1, 4))
        #  0--1---4
        #  | / \ /
        #  2----3
        assert_num_triangles(None, 3)
        assert_num_triangles(0, 1)
        assert_num_triangles(1, 3)
        assert_num_triangles(2, 2)
        assert_num_triangles(3, 2)
        assert_num_triangles(4, 1)

        tb.put((0, 4))
        #   _____
        #  /     \
        #  0--1---4
        #  | / \ /
        #  2----3
        assert_num_triangles(None, 4)
        assert_num_triangles(0, 2)
        assert_num_triangles(1, 4)
        assert_num_triangles(2, 2)
        assert_num_triangles(3, 2)
        assert_num_triangles(4, 2)

        tb([(4, 5), ])
        #   _____
        #  /     \
        #  0--1---4---5
        #  | / \ /   /
        #  2----3   /
        #   \______/
        assert_num_triangles(None, 4)
        assert_num_triangles(0, 2)
        assert_num_triangles(1, 4)
        assert_num_triangles(2, 2)
        assert_num_triangles(3, 2)
        assert_num_triangles(4, 2)
        assert_num_triangles(5, 0)

    def test_approximation(self):
        seed = 1234
        random.seed(seed)

        nodes = range(100)
        edges = list(combinations(nodes, 2))
        random.shuffle(edges)

        tb_reference = self.test_class(size=len(edges))
        tb = self.test_class(size=1000, seed=seed)

        tb_reference(edges[:1000])
        tb(edges[:1000])
        assert_equal(
            tb.get_estimated_num_triangles(),
            tb_reference.get_estimated_num_triangles()
        )
        sample_nodes = random.sample(tb.reservoir.nodes, 100)
        for node in sample_nodes:
            assert_equal(
                tb.get_estimated_num_triangles(node),
                tb_reference.get_estimated_num_triangles(node)
            )
        tb_reference(edges[1000:])

        glob_num_triangles_est = []
        node_to_num_triangles_est = {node: [] for node in sample_nodes}
        for _ in range(10):
            tb.reset()
            tb(edges)
            glob_num_triangles_est.append(tb.get_estimated_num_triangles())
            for node in sample_nodes:
                node_to_num_triangles_est[node].append(
                    tb.get_estimated_num_triangles(node)
                )

        ref_num = tb_reference.get_estimated_num_triangles()
        assert_allclose(np.mean(glob_num_triangles_est), ref_num, rtol=0.10)
        assert_less(np.std(glob_num_triangles_est), 0.1 * ref_num)

        rel_diffs = []
        for node in sample_nodes:
            ref_num = tb_reference.get_estimated_num_triangles(node)
            est_num = np.mean(node_to_num_triangles_est[node])
            rel_diff = (est_num - ref_num) / ref_num
            rel_diffs.append(rel_diff)
        assert_allclose(1 + np.mean(rel_diffs), 1., rtol=0.10)


class TestTriestBase(TriestTests):
    from data_mining.graph import TriestBase as test_class


class TestTriestImpr(TriestTests):
    from data_mining.graph import TriestImpr as test_class
