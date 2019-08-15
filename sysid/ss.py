"""
This module performs system identification.
"""
import numpy as np

import scipy.linalg
import matplotlib.pyplot as plt

# pylint: disable=invalid-name, too-few-public-methods, no-self-use


__all__ = ['StateSpaceDiscreteLinear',
           'StateSpaceDataList', 'StateSpaceDataArray']


class StateSpaceDiscreteLinear(object):
    """
    State space for discrete linear systems.
    """

    def __init__(self, A, B, C, D, Q, R, dt):
        #pylint: disable=too-many-arguments
        self.A = np.matrix(A)
        self.B = np.matrix(B)
        self.C = np.matrix(C)
        self.D = np.matrix(D)
        self.Q = np.matrix(Q)
        self.R = np.matrix(R)
        self.dt = dt

        n_x = self.A.shape[0]
        n_u = self.B.shape[1]
        n_y = self.C.shape[0]

        assert self.A.shape[1] == n_x
        assert self.B.shape[0] == n_x
        assert self.C.shape[1] == n_x
        assert self.D.shape[0] == n_y
        assert self.D.shape[1] == n_u
        assert self.Q.shape[0] == n_x
        assert self.Q.shape[1] == n_x
        assert self.R.shape[0] == n_u
        assert self.R.shape[1] == n_u
        assert np.matrix(dt).shape == (1, 1)

    def dynamics(self, x, u, w):
        """
        Dynamics
        x(k+1) = A x(k) + B u(k) + w(k)

        E(ww^T) = Q

        Parameters
        ----------
        x : The current state.
        u : The current input.
        w : The current process noise.

        Return
        ------
        x(k+1) : The next state.

        """
        x = np.matrix(x)
        u = np.matrix(u)
        w = np.matrix(w)
        assert x.shape[1] == 1
        assert u.shape[1] == 1
        assert w.shape[1] == 1
        return self.A*x + self.B*u + w

    def measurement(self, x, u, v):
        """
        Measurement.
        y(k) = C x(k) + D u(k) + v(k)

        E(vv^T) = R

        Parameters
        ----------
        x : The current state.
        u : The current input.
        v : The current measurement noise.

        Return
        ------
        y(k) : The current measurement
        """
        x = np.matrix(x)
        u = np.matrix(u)
        v = np.matrix(v)
        assert x.shape[1] == 1
        assert u.shape[1] == 1
        assert v.shape[1] == 1
        return self.C*x + self.D*u + v

    def simulate(self, f_u, x0, tf):
        """
        Simulate the system.

        Parameters
        ----------
        f_u: The input function  f_u(t, x, i)
        x0: The initial state.
        tf: The final time.

        Return
        ------
        data : A StateSpaceDataArray object.

        """
        # pylint: disable=too-many-locals, no-member
        x0 = np.matrix(x0)
        assert x0.shape[1] == 1
        t = 0
        x = x0
        dt = self.dt
        data = StateSpaceDataList([], [], [], [])
        i = 0
        n_x = self.A.shape[0]
        n_y = self.C.shape[0]
        assert np.matrix(f_u(0, x0, 0)).shape[1] == 1
        assert np.matrix(f_u(0, x0, 0)).shape[0] == n_y

        # take square root of noise cov to prepare for noise sim
        if np.linalg.norm(self.Q) > 0:
            sqrtQ = scipy.linalg.sqrtm(self.Q)
        else:
            sqrtQ = self.Q

        if np.linalg.norm(self.R) > 0:
            sqrtR = scipy.linalg.sqrtm(self.R)
        else:
            sqrtR = self.R

        # main simulation loop
        while t + dt < tf:
            u = f_u(t, x, i)
            v = sqrtR.dot(np.random.randn(n_y, 1))
            y = self.measurement(x, u, v)
            data.append(t, x, y, u)
            w = sqrtQ.dot(np.random.randn(n_x, 1))
            x = self.dynamics(x, u, w)
            t += dt
            i += 1
        return data.to_StateSpaceDataArray()

    def __repr__(self):
        return repr(self.__dict__)

    def __str__(self):
        return str(self.__dict__)


class StateSpaceDataList(object):
    """
    An expandable state space data list.
    """

    def __init__(self, t, x, y, u):

        self.t = t
        self.x = x
        self.y = y
        self.u = u

    def append(self, t, x, y, u):
        """
        Add to list.
        """
        self.t += [t]
        self.x += [x]
        self.y += [y]
        self.u += [u]

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return repr(self.__dict__)

    def to_StateSpaceDataArray(self):
        """
        Converts to an state space data  array object.
        With fixed sizes.
        """
        return StateSpaceDataArray(
            t=np.array(self.t).T,
            x=np.array(self.x).T,
            y=np.array(self.y).T,
            u=np.array(self.u).T)


class StateSpaceDataArray(object):
    """
    A fixed size state space data lit.
    """

    def __init__(self, t, x, y, u):

        self.t = np.matrix(t)
        self.x = np.matrix(x)
        self.y = np.matrix(y)
        self.u = np.matrix(u)

        assert self.t.shape[0] == 1
        assert self.x.shape[0] < self.x.shape[1]
        assert self.y.shape[0] < self.y.shape[1]
        assert self.u.shape[0] < self.u.shape[1]

    def to_StateSpaceDataList(self):
        """
        Convert to StateSpaceDataList that you can append to.
        """
        return StateSpaceDataList(
            t=list(self.t),
            x=list(self.x),
            y=list(self.y),
            u=list(self.u))

    def plot(self, plot_x=False, plot_y=False, plot_u=False):
        """
        Plot data.
        """
        t = self.t.T
        x = self.x.T
        y = self.y.T
        u = self.u.T
        if plot_x:
            plt.plot(t, x)
        if plot_y:
            plt.plot(t, y)
        if plot_u:
            plt.plot(t, u)


# vim: set et fenc=utf-8 ft=python ff=unix sts=0 sw=4 ts=4 :
