from __future__ import print_function, division

import numpy as np

from numpy.testing import assert_allclose
from nose.tools import assert_equal, assert_not_equal


class TestReservoirSampling(object):
    from data_mining.stream import ReservoirSampling as test_class

    def test_basics(self):
        stream = range(100)
        rs = self.test_class(size=100, seed=0)
        rs(stream)
        assert_equal(rs.reservoir, stream)
        assert_equal(rs.t, 100)
        rs(stream)
        assert_equal(len(rs.reservoir), 100)
        assert_not_equal(rs.reservoir, stream)
        assert_equal(rs.t, 200)
        rs.reset()
        assert_equal(rs.reservoir, [])

    def test_uniformity(self):
        stream = np.random.randn(10000) + 1
        rs = self.test_class(size=1000, seed=0)
        rs(stream)
        for stat in [np.mean, np.std]:
            assert_allclose(stat(rs.reservoir), stat(stream), rtol=0.05)

    def test_uniformity_non_stationary(self):
        stream = [
            mean + np.random.randn() * std for mean, std in
            zip(np.linspace(0, 10, 10000), np.linspace(0, 3, 10000))
        ]
        rs = self.test_class(size=1000, seed=0)
        rs(stream)
        for stat in [np.mean, np.std]:
            assert_allclose(stat(rs.reservoir), stat(stream), rtol=0.05)
