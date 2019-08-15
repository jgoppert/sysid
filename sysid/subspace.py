"""
This module performs subspace system identification.

It enforces that matrices are used instead of arrays
to avoid dimension conflicts.
"""
import numpy as np

from . import ss

__all__ = ['subspace_det_algo1', 'prbs', 'nrms']

#pylint: disable=invalid-name


def block_hankel(data, f):
    """
    Create a block hankel matrix.
    f : number of rows
    """
    data = np.array(data)
    assert len(data.shape) == 2
    n = data.shape[1] - f
    return np.vstack([
        np.hstack([data[:, i+j] for i in range(f)])
        for j in range(n)]).T


def project(A):
    """
    Creates a projection matrix onto the rowspace of A.
    """
    return A.T.dot(np.linalg.inv(A.dot(A.T))).dot(A)


def project_perp(A):
    """
    Creates a projection matrix onto the space perpendicular to the
    rowspace of A.
    """
    I = np.eye(A.shape[1])
    P = project(A)
    return I - P


def project_oblique(B, C):
    """
    Projects along rowspace of B onto rowspace of C.
    """
    proj_B_perp = project_perp(B)
    return proj_B_perp.dot(np.linalg.pinv(C.dot(proj_B_perp))).dot(C)


def subspace_det_algo1(y, u, f, p, s_tol, dt):
    """
    Subspace Identification for deterministic systems
    algorithm 1 from (1)

    assuming a system of the form:

    x(k+1) = A x(k) + B u(k)
    y(k)   = C x(k) + D u(k)

    and given y and u.

    Find A, B, C, D

    See page 52. of (1)

    (1) Subspace Identification for Linear
    Systems, by Van Overschee and Moor. 1996
    """
    # pylint: disable=too-many-arguments, too-many-locals
    # for this algorithm, we need future and past
    # to be more than 1
    assert f > 1
    assert p > 1

    # setup matrices1
    y = np.array(np.atleast_2d(y))
    n_y = y.shape[0]
    u = np.array(np.atleast_2d(u))
    n_u = u.shape[0]
    w = np.vstack([y, u])
    n_w = w.shape[0]

    # make sure the input is column vectors
    assert y.shape[0] < y.shape[1]
    assert u.shape[0] < u.shape[1]

    W = block_hankel(w, f + p)
    U = block_hankel(u, f + p)
    Y = block_hankel(y, f + p)

    W_p = W[:n_w*p, :]
    W_pp = W[:n_w*(p+1), :]

    Y_f = Y[n_y*f:, :]
    U_f = U[n_y*f:, :]

    Y_fm = Y[n_y*(f+1):, :]
    U_fm = U[n_u*(f+1):, :]

    # see (Subspace Identification for Linear Systems)
    # Van Overschee pg 56

    # step 1, calculate the oblique projections
    # ------------------------------------------
    # Y_p = G_i Xd_p + Hd_i U_p
    # After the oblique projection, U_p component is eliminated,
    # without changing the Xd_p component:
    # Proj_perp_(U_p) Y_p = W1 O_i W2 = G_i Xd_p
    O_i = Y_f.dot(project_oblique(U_f, W_p))
    O_im = Y_fm.dot(project_oblique(U_fm, W_pp))

    # step 2, calculate the SVD of the weighted oblique projection
    # ------------------------------------------
    # given: W1 O_i W2 = G_i Xd_p
    # want to solve for G_i, but know product, and not Xd_p
    # so can only find Xd_p up to a similarity transformation
    W1 = np.eye(O_i.shape[0])
    W2 = np.eye(O_i.shape[1])
    U0, s0, VT0 = np.linalg.svd(W1.dot(O_i).dot(W2))  # pylint: disable=unused-variable

    # step 3, determine the order by inspecting the singular
    # ------------------------------------------
    # values in S and partition the SVD accordingly to obtain U1, S1
    # print s0
    n_x = np.where(s0/s0.max() > s_tol)[0][-1] + 1
    U1 = U0[:, :n_x]
    # S1 = np.array(np.diag(s0[:n_x]))
    # VT1 = VT0[:n_x, :n_x]

    # step 4, determine Gi and Gim
    # ------------------------------------------
    G_i = np.linalg.pinv(W1).dot(U1).dot(np.diag(np.sqrt(s0[:n_x])))
    G_im = G_i[:-n_y, :]  # check

    # step 5, determine Xd_ip and Xd_p
    # ------------------------------------------
    # only know Xd up to a similarity transformation
    Xd_i = np.linalg.pinv(G_i).dot(O_i)
    Xd_ip = np.linalg.pinv(G_im).dot(O_im)

    # step 6, solve the set of linear eqs
    # for A, B, C, D
    # ------------------------------------------
    Y_ii = Y[n_y*p:n_y*(p+1), :]
    U_ii = U[n_u*p:n_u*(p+1), :]

    a_mat = np.vstack([Xd_ip, Y_ii])
    b_mat = np.vstack([Xd_i, U_ii])
    ss_mat = a_mat.dot(np.linalg.pinv(b_mat))
    A_id = ss_mat[:n_x, :n_x]
    B_id = ss_mat[:n_x, n_x:]
    assert B_id.shape[0] == n_x
    assert B_id.shape[1] == n_u
    C_id = ss_mat[n_x:, :n_x]
    assert C_id.shape[0] == n_y
    assert C_id.shape[1] == n_x
    D_id = ss_mat[n_x:, n_x:]
    assert D_id.shape[0] == n_y
    assert D_id.shape[1] == n_u

    if np.linalg.matrix_rank(C_id) == n_x:
        T = np.linalg.inv(C_id)  # try to make C identity, want it to look like state feedback
    else:
        T = np.eye(n_x)

    Q_id = np.zeros((n_x, n_x))
    R_id = np.zeros((n_y, n_y))
    sys = ss.StateSpaceDiscreteLinear(
        A=np.linalg.inv(T).dot(A_id).dot(T), B=np.linalg.inv(T).dot(B_id), C=C_id.dot(T), D=D_id,
        Q=Q_id, R=R_id, dt=dt)
    return sys


def nrms(data_fit, data_true):
    """
    Normalized root mean square error.
    """
    # root mean square error
    rms = np.mean(np.linalg.norm(data_fit - data_true, axis=0))

    # normalization factor is the max - min magnitude, or 2 times max dist from mean
    norm_factor = 2 * \
        np.linalg.norm(data_true - np.mean(data_true, axis=1), axis=0).max()
    return (norm_factor - rms)/norm_factor


def prbs(n):
    """
    Pseudo random binary sequence.
    """
    return np.where(np.random.rand(n) > 0.5, 0, 1)


# vim: set et fenc=utf-8 ft=python  ff=unix sts=4 sw=4 ts=4 :
