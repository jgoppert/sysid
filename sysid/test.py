"""
Unit testing.
"""
import unittest
import pylab as pl
from . import ss, subspace

#pylint: disable=invalid-name, no-self-use

class TestSS(unittest.TestCase):
    """
    Unit testing.
    """

    def test_state_space(self):
        """
        Check state space manipulations.
        """
        data = ss.StateSpaceDataList([], [], [], [])
        for i in range(10):
            data.append(t=i, x=1, y=2, u=3)
        data = data.to_StateSpaceDataArray()

    def test_state_space_discrete_linear(self):
        """
        State space discrete linear.
        """
        sys1 = ss.StateSpaceDiscreteLinear(
            A=0.9, B=0.01, C=1, D=0, dt=0.1)
        x0 = 1
        u0 = 1
        y0 = sys1.measurement(x0, u0)
        x1 = sys1.dynamics(x0, u0)
        data = sys1.simulate(f_u=lambda t, x: pl.sin(t), x0=x0, tf=10)
        data.plot()
        pl.show()

class TestSubspace(unittest.TestCase):
    """
    Unit testing.
    """

    def test_block_hankel(self):
        """
        Test block hankel function.
        """
        y = pl.matrix(pl.rand(3, 100))
        Y = subspace.block_hankel(y, 5)
        self.assertEqual(Y.shape, (15, 95))

    def test_subspace_ident(self):
        """
        Test subspace identification.
        """
        y = pl.matrix(pl.randn(3, 100))
        u = pl.matrix(pl.randn(2, 100))
        subspace.subspace_ident(y, u, 5)

    def test_subspace_det_algo1(self):
        """
        Test subspace det algorithm.
        """
        sys1 = ss.StateSpaceDiscreteLinear(
            A=0.9, B=0.01, C=1, D=0, dt=0.1)
        x0 = 1
        tf = 20
        f = 5
        s_tol = 0.1
        p = f
        f_u = lambda t, x: pl.sin(t)
        data = sys1.simulate(f_u=f_u, x0=x0, tf=tf)
        sys1_id = subspace.subspace_det_algo1(
            y=data.y, u=data.u, f=f, p=p,
            s_tol=s_tol, dt=sys1.dt)
        T_x0 = sys1_id.C.I*x0
        data_id = sys1_id.simulate(f_u=f_u, x0=T_x0, tf=tf)

        print sys1
        print sys1_id

        pl.plot(data.t.T, data.y.T, label='true')
        pl.plot(data_id.t.T, data_id.y.T, label='fit')
        pl.legend()
        pl.show()
