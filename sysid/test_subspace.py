"""
Unit testing.
"""
import unittest
import pylab as pl
from sysid import ss, subspace

pl.ion()

#pylint: disable=invalid-name, no-self-use

ENABLE_PLOTTING = True

class TestSubspace(unittest.TestCase):
    """
    Unit testing.
    """

    def test_block_hankel(self):
        """
        Block hankel function.
        """
        y = pl.rand(3, 100)
        Y = subspace.block_hankel(y, 5)
        self.assertEqual(Y.shape, (15, 95))

    def test_subspace_det_algo1(self):
        """
        Subspace deterministic algorithm.
        """
        x0 = 0
        dt = 0.1
        tf = 20
        f = 4
        s_tol = 1e-1
        p = f
        sys1 = ss.StateSpaceDiscreteLinear(
            A=0.9, B=0.1, C=1, D=0, Q=0.001, R=0.001, dt=dt)
        prbs1 = subspace.prbs(tf/dt)

        def f_u_prbs(t, x, i):
            """
            Pseudo random binary sequence input function.
            """
            #pylint: disable=unused-argument
            return 2*prbs1[i] - 1

        def f_u_square(t, x, i):
            """
            Square wave input function.
            """
            #pylint: disable=unused-argument, unused-variable
            if pl.sin(t) > 0:
                return -1
            else:
                return 1

        data = sys1.simulate(f_u=f_u_prbs, x0=x0, tf=tf)
        sys1_id = subspace.subspace_det_algo1(
            y=data.y, u=data.u, f=f, p=p,
            s_tol=s_tol, dt=sys1.dt)
        data_id = sys1_id.simulate(
            f_u=f_u_prbs,
            x0=pl.zeros((sys1_id.A.shape[0], 1)), tf=tf)

        #print sys1
        #print sys1_id
        nrms = subspace.nrms(data_id.y, data.y)
        #print "nrms", nrms

        if ENABLE_PLOTTING:
            pl.figure()
            pl.plot(data.t.T, data.y.T, label='true')
            pl.plot(data_id.t.T, data_id.y.T, label='fit')
            pl.legend()
            pl.show()

        self.assertGreater(nrms, 0.9)


if __name__ == "__main__":
    unittest.main()
