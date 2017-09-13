from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

pinv = np.linalg.pinv
rank = np.linalg.matrix_rank

def block(B):
    """
    create a block matrix and return an array, numpy.bmat
    returns a np.matrix which is problematic
    """
    return np.array(np.bmat(B))

def project(B):
    """
    projection onto the rowspace of B
    """
    return B.T.dot(pinv(B.dot(B.T))).dot(B)

def project_compliment(B):
    """
    projection onto the compliment of the rowspace of B
    """
    P = project(B)
    return np.eye(P.shape[0]) - P

def project_oblique(B, C):
    """
    oblique projection along the row space of B on the
    row space of C
    """
    r = C.shape[0]
    F = block([[C.dot(C.T), C.dot(B.T)], [B.dot(C.T), B.dot(B.T)]])
    return block([C.T, B.T]).dot(pinv(F)[:,:r]).dot(C)

def test_projections():
    A = np.random.randn(3, 3)
    B = np.random.randn(1, 3)
    C = np.vstack([B, np.random.randn(1, 3)])
    assert np.allclose(A, A.dot(project(B)) + A.dot(project_compliment(B)))
    assert np.allclose(A, A.dot(project_oblique(C, B)) + A.dot(project_oblique(B, C)) +         A.dot(project_compliment(np.vstack([B, C]))))

test_projections()


def block_hankel(data, i):
    """
    Create a block hankel matrix.
    i : number of rows in future/past block
    """
    assert len(data.shape) == 2
    s = data.shape[1]
    n_u = data.shape[0]
    j = s - 2*i + 1
    U = np.vstack([
        np.hstack([np.array([data[:, ii+jj]]).T for jj in range(j)])
        for ii in range(2*i)])
    return {
        'full': U,
        'i': U[i*n_u:(i + 1)*n_u, :],
        'p': U[0:i*n_u, :],
        'f': U[i*n_u:(2*i)*n_u, :],
        'pp': U[0:(i + 1)*n_u, :],
        'fm': U[(i + 1)*n_u:(2*i)*n_u, :],
        'pm': U[0:(i - 1)*n_u, :],
        'fp': U[(i - 1)*n_u:(2*i)*n_u, :],
    }


class StochasticStateSpaceDiscrete(object):
    
    def __init__(self, A, B, C, D, Q, R, x0, dt):
        self._A = np.array(A)
        self._B = np.array(B)
        self._C = np.array(C)
        self._D = np.array(D)
        self._Q = np.array(Q)
        self._R = np.array(R)
        self._x0 = np.array(x0)
        self._dt = dt
        for m in ['A', 'B', 'C', 'D', 'Q', 'R']:
            val = getattr(self, '_' + m)
            if len(val.shape) != 2:
                raise ValueError(m, 'must be a 2D array, got shape', val.shape)
        for m in ['x0']:
            val = getattr(self, '_' + m)
            if len(val.shape) != 1:
                raise ValueError(m, 'must be a 1D array, got shape', val.shape)

    @property
    def n_x(self):
        return self._A.shape[0]

    @property
    def n_y(self):
        return self._C.shape[0]

    @property
    def n_u(self):
        return self._B.shape[1]

    @property
    def dt(self):
        return self._dt

    @classmethod
    def rand(cls, n_x, n_y, n_u, dt):
        A = np.random.randn(n_x, n_x)
        A /= A.max()
        B = np.random.randn(n_x, n_u)
        C = np.random.randn(n_y, n_x)
        D = np.random.randn(n_y, n_u)
        Q = np.diag(0.01*np.random.rand(n_x))
        R = np.diag(0.01*np.random.rand(n_y))
        x0 = np.random.randn(n_x)
        return cls(A, B, C, D, Q, R, x0, dt)
    
    def sim(self, t, u, plot=False):
        u = np.array(u)
        if u.shape[1] != self.n_u:
            raise ValueError('u shape must be (, {:d}), got {:s}'.format(self.n_u, str(u.shape)))
        x0 = self._x0
        A = self._A
        B = self._B
        C = self._C
        D = self._D
        Q = self._Q
        R = self._R
        n_y = C.shape[0]
        n_x = A.shape[0]
        n_u = B.shape[1]
        xi = x0
        x = []
        y = []
        for i in range(u.shape[0]):
            ui = u[i, :]
            x.append(xi)
            v = np.random.multivariate_normal(np.zeros(self.n_y), self._R)
            yi = C.dot(xi) + D.dot(ui) + v
            y.append(yi)
            w = np.random.multivariate_normal(np.zeros(self.n_x), self._Q)
            xi = A.dot(xi) + B.dot(ui) + w

        x = np.array(x)
        y = np.array(y)

        if plot:
            plt.title('simulation')
            y_lines = plt.plot(t, y, '.-')
            y_labels = ['y_{:d}'.format(i) for i in range(n_y)]
            x_lines = plt.plot(t, x, '.-', label=['x_{:d}'.format(i) for i in range(n_x)])
            x_labels = ['x_{:d}'.format(i) for i in range(n_x)]
            plt.legend(x_lines + y_lines, x_labels + y_labels)
            plt.grid()
            plt.xlabel('t')
            
        return y, x

    def __repr__(self):
        return repr(self.__dict__)

def compute_fitness(y, y_fit):
    return 1 - np.var(y_fit - y)/np.var(y)

def normalized_error(y, y_fit):
    return (y_fit - y)/np.std(y)

def plot_normalized_error(t, y, y_fit):
    e = normalized_error(y, y_fit)
    plt.plot(t, e)
    plt.xlabel('t, sec')
    plt.ylabel('$e/\sigma(y)$')
    plt.title('normalized error')
    plt.grid()

def plot_output_comparison(t, y, y_fit):
    e = normalized_error(y, y_fit)
    plt.plot(t, y)
    plt.plot(t, y_fit)
    plt.xlabel('t, sec')
    plt.title('output comaprison error')
    plt.grid()

def combined_algo_2(y, u, n_x_max, dt):
    i = 1 + n_x_max**2
    
    # transpose to match definitions in book
    y = y.T
    u = u.T

    n_u = u.shape[0]
    n_y = y.shape[0]

    U = block_hankel(u, i)
    Y = block_hankel(y, i)

    u_rank = rank(np.cov(U['full']))
    if u_rank < 2*n_u*i:
        print('WARNING: input not persistently exciting'
            ' order {:d} < {:d}'.format(
            u_rank, 2*n_u*i))

    W_p = np.vstack([U['p'], Y['p']])
    W_pp = np.vstack([U['pp'], Y['pp']])

    O_i = Y['f'].dot(project_oblique(U['f'], W_p))
    O_im = Y['fm'].dot(project_oblique(U['fm'], W_pp))

    W1 = np.eye(n_y, O_i.shape[0])
    W2 = project_compliment(U['f'])

    if rank(W_p) == rank(W_p.dot(W2)):
        print('WARNING: rank(W_p) != rank(W_p*W2)')

    U_, s, VT = np.linalg.svd(W1.dot(O_i).dot(W2), full_matrices=0)
    assert np.allclose(W1.dot(O_i).dot(W2), U_.dot(np.diag(s).dot(VT)))

    s_tol = 1e-3
    U1 = np.zeros_like(U_)
    n_x = np.count_nonzero(s/s.max() > s_tol)
    U1 = np.array(U_[:,:n_x])
    S1_sqrt = np.diag(np.sqrt(s[:n_x]))
    Gamma_i = pinv(W1).dot(U1).dot(S1_sqrt)
    Gamma_im = Gamma_i[:-n_y, :]

    X_i_d = pinv(Gamma_i).dot(O_i)
    X_ip_d = pinv(Gamma_im).dot(O_im)

    LHS = np.vstack([X_ip_d, Y['i']])
    RHS = np.vstack([X_i_d, U['i']])
    Coeff = LHS.dot(pinv(RHS))

    A = Coeff[:n_x ,:n_x]
    B = Coeff[:n_x ,n_x:]
    C = Coeff[n_x: ,:n_x]
    D = Coeff[n_x: ,n_x:]

    residuals = LHS - Coeff.dot(RHS)
    QR = np.cov(residuals)
    Q = QR[:n_x, :n_x]
    R = QR[n_x:, n_x:]
    S = QR[:n_x, n_x:]

    x0 = np.zeros(n_x)
    return StochasticStateSpaceDiscrete(A, B, C, D, Q, R, x0, dt)

def robust_combined_stochastic(y, u, n_x_max, dt):
    i = 1 + n_x_max**2
    
    # transpose to match definitions in book
    y = y.T
    u = u.T

    n_u = u.shape[0]
    n_y = y.shape[0]

    U = block_hankel(u, i)
    Y = block_hankel(y, i)

    u_rank = rank(np.cov(U['full']))
    if u_rank < 2*n_u*i:
        raise ValueError('input not persistently exciting'
            ' order {:d} < {:d}'.format(
            u_rank, 2*n_u*i))

    W_p = np.vstack([U['p'], Y['p']])
    W_pp = np.vstack([U['pp'], Y['pp']])

    O_i = Y['f'].dot(project_oblique(U['f'], W_p))
    Z_i = Y['f'].dot(project(np.vstack([W_p, U['f']])))
    Z_ip = Y['fm'].dot(project(np.vstack([W_pp, U['fm']])))

    U_S_VT = O_i.dot(project_compliment(U['f']))
    U_, s, VT = np.linalg.svd(U_S_VT, full_matrices=0)
    assert np.allclose(U_S_VT, U_.dot(np.diag(s).dot(VT)))

    s_tol = 1e-3
    U1 = np.zeros_like(U_)
    n_x = np.count_nonzero(s/s.max() > s_tol)
    U1 = np.array(U_[:,:n_x])
    S1_sqrt = np.diag(np.sqrt(s[:n_x]))
    Gamma_i = U1.dot(S1_sqrt)
    Gamma_im = Gamma_i[:-n_y, :]

    LHS = np.vstack([pinv(Gamma_im).dot(Z_ip), Y['i']])
    RHS = np.vstack([pinv(Gamma_i).dot(Z_i), U['f']])
    print('LHS shape', LHS.shape)
    print('RHS shape', RHS.shape)
    Coeff = LHS.dot(pinv(RHS))
    print('Coeff', Coeff)

    residuals = LHS - Coeff.dot(RHS)
    QR = np.cov(residuals)
    
    A = Coeff[:n_x ,:n_x]
    C = Coeff[n_x: ,:n_x]
    K = Coeff[:, n_x:]
    print('K\n', K.shape)

    # TODO
    B = np.eye(n_x, n_u)
    D = np.eye(n_y, n_u)

    Q = QR[:n_x, :n_x]
    R = QR[n_x:, n_x:]
    S = QR[:n_x, n_x:]

    print('A\n', A)
    print('C\n', C)
    print('Q\n', Q)
    print('R\n', R)
    print('S\n', S)

    x0 = np.zeros(n_x)
    return StochasticStateSpaceDiscrete(A, B, C, D, Q, R, x0, dt)

def prbs(m, n):
    """
    Pseudo random binary sequence.
    """
    return np.array(np.random.rand(m, n) > 0.5, dtype=np.int) - 0.5

def sinusoid(m, f, t):
    u = []
    for i in range(m):
        A = np.random.rand()
        phase = 2*np.pi*np.random.rand()
        fi = f + np.random.randn()
        u.append(A*np.sin(phase + 2*np.pi*fi*t))
    return np.vstack(u).T

if __name__ == "__main__":

    n_x = 2
    n_y = 3
    n_u = 4
    dt = 0.01
    tf = 10
    np.random.seed(1235)

    ss1 = StochasticStateSpaceDiscrete.rand(n_x, n_y, n_u, dt)
    t = np.arange(0, tf, dt)
    u = sinusoid(n_u, 1, t)
    # u = prbs(len(t), n_u)

    plt.figure()
    plt.subplot(311)
    y, x = ss1.sim(t, u, plot=True)

    ss_fit = combined_algo_2(y, u, n_x_max=n_x, dt=dt)
    # ss_fit = robust_combined_stochastic(y, u, n_x_max=n_x, dt=dt)
    print(ss_fit)
    y_fit, x_fit = ss_fit.sim(t, u)

    fit = compute_fitness(y, y_fit)
    print('fitness: {:f}%'.format(100*fit))

    plt.subplot(312)
    plot_normalized_error(t, y, y_fit)

    plt.subplot(313)
    plot_output_comparison(t, y, y_fit)

    plt.show()
