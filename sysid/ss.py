"""
This module performs system identification.
"""
import pylab as pl

#pylint: disable=invalid-name, too-few-public-methods, no-self-use


class StateSpaceDiscreteLinear(object):
    """
    State space for discrete linear systems.
    """

    def __init__(self, A, B, C, D, dt):
        #pylint: disable=too-many-arguments
        self.A = pl.matrix(A)
        self.B = pl.matrix(B)
        self.C = pl.matrix(C)
        self.D = pl.matrix(D)
        self.dt = dt

    def dynamics(self, x, u):
        """
        Dynamics
        x(k+1) = A x(k) + B u(k)

        Parameters
        ----------
        x : The current state.
        u : The current input.

        Return
        ------
        x(k+1) : The next state.

        """
        x = pl.matrix(x)
        u = pl.matrix(u)
        A = self.A
        B = self.B
        return A*x + B*u

    def measurement(self, x, u):
        """
        Measurement.
        y(k) = C x(k) + D u(k)

        Parameters
        ----------
        x : The current state.
        u : The current input.

        Return
        ------
        y(k) : The current measurement
        """
        x = pl.matrix(x)
        u = pl.matrix(u)
        C = self.C
        D = self.D
        return C*x + D*u

    def simulate(self, f_u, x0, tf):
        """
        Simulate the system.

        Parameters
        ----------
        f_u: The input function of time.
        x0: The initial state.
        tf: The final time.

        Return
        ------
        data : A StateSpaceDataArray object.

        """
        t = 0
        x = x0
        dt = self.dt
        data = StateSpaceDataList([], [], [], [])
        while t + dt < tf:
            u = f_u(t, x)
            x = self.dynamics(x, u)
            y = self.measurement(x, u)
            data.append(t, x, y, u)
            t += dt
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
            t=pl.array(self.t),
            x=pl.array(self.x),
            y=pl.array(self.y),
            u=pl.array(self.u))


class StateSpaceDataArray(object):
    """
    A fixed size state space data lit.
    """

    def __init__(self, t, x, y, u):

        self.t = pl.matrix(t)
        self.x = pl.matrix(x)
        self.y = pl.matrix(y)
        self.u = pl.matrix(u)

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
            pl.plot(t, x)
        if plot_y:
            pl.plot(t, y)
        if plot_u:
            pl.plot(t, u)


# vim: set et fenc=utf-8 ft=python ff=unix sts=0 sw=4 ts=4 :
