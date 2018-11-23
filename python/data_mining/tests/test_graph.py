from __future__ import print_function, division

from nose.tools import assert_equal, assert_raises


class GraphTests(object):
    test_class = None

    def test_put_edge(self):
        g = self.test_class()
        g.put_edge((0, 1))
        assert_equal(g.get_neighbors(0), {1})
        assert_equal(g.get_neighbors(1), {0})
        assert_equal(g.edges, [(0, 1)])
        assert_equal(g.num_edges, 1)

        g.put_edge((0, 2))
        assert_equal(g.get_neighbors(0), {1, 2})
        assert_equal(g.get_neighbors(2), {0})
        assert_equal(g.get_neighbors(1), {0})
        assert_equal(g.edges, [(0, 1), (0, 2)])
        assert_equal(g.num_edges, 2)

        g.put_edge((2, 1))
        assert_equal(g.get_neighbors(0), {1, 2})
        assert_equal(g.get_neighbors(2), {0, 1})
        assert_equal(g.get_neighbors(1), {0, 2})
        assert_equal(g.edges, [(0, 1), (0, 2), (2, 1)])
        assert_equal(g.num_edges, 3)

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


class TestUndirectedGraph(GraphTests):
    from data_mining.graph import UndirectedGraph as test_class


class TestEdgeReservoir(GraphTests):
    from data_mining.graph import EdgeReservoir as test_class

    def test_pop_replace_random_edge(self):
        g = self.test_class()
        initial_edges = [(0, 1), (0, 2), (1, 2)]
        for edge in initial_edges:
            g.put_edge(edge)
        edge = g.pop_replace_random_edge((0, 3))
        assert edge in initial_edges
        assert_equal(set(g.edges).union({edge}), set(initial_edges).union({(0, 3)}))
