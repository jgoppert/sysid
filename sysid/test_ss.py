"""
Unit testing.
"""
import unittest
import numpy as np
import matplotlib.pyplot as plt

from sysid import ss

# pylint: disable=invalid-name, no-self-use

ENABLE_PLOTTING = True


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
            A=0.9, B=0.01, C=1, D=0, Q=0, R=0, dt=0.1)
        x0 = 1
        u0 = 1
        v0 = 0
        w0 = 0
        #pylint: disable=unused-variable
        y0 = sys1.measurement(x0, u0, v0)
        x1 = sys1.dynamics(x0, u0, w0)
        data = sys1.simulate(f_u=lambda t, x, i: np.sin(t), x0=x0, tf=10)

        if ENABLE_PLOTTING:
            data.plot()
            plt.show()


if __name__ == "__main__":
    unittest.main()
