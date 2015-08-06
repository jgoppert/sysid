"""
This module performs subspace system identification.

It enforces that matrices are used instead of arrays
to avoid dimension conflicts.
"""
import pylab as pl
from . import ss

#pylint: disable=invalid-name

def block_hankel(data, f):
    """
    Create a block hankel matrix.
    f : number of rows
    """
    data = pl.matrix(data)
    assert len(data.shape) == 2
    n = data.shape[1] - f
    return pl.matrix(pl.hstack([
        pl.vstack([data[:, i+j] for i in range(f)])
        for j in range(n)]))

def project(A):
    """
    Creates a projection matrix onto the rowspace of A.
    """
    A = pl.matrix(A)
    return  A.T*(A*A.T).I*A

def project_perp(A):
    """
    Creates a projection matrix onto the space perpendicular to the
    rowspace of A.
    """
    A = pl.matrix(A)
    I = pl.matrix(pl.eye(A.shape[1]))
    P = project(A)
    return  I - P

def project_oblique(B, C):
    """
    Projects along rowspace of B onto rowspace of C.
    """
    proj_B_perp = project_perp(B)
    return proj_B_perp*(C*proj_B_perp).I*C

def subspace_ident(y, u, f):
    """
    Do subspace identification.
    """
    y = pl.matrix(y)
    u = pl.matrix(u)
    f = 5
    Y = block_hankel(y, f)
    U = block_hankel(u, f)
    proj_perp_U = project_perp(U)
    Y_proj_U_perp = Y*proj_perp_U
    #pl.plot(Y[0,:].T)
    #pl.plot(Y_proj_row_U_perp[0,:].T)
    #pl.show()

def subspace_det_algo1(y, u, f, p, s_tol, dt):
    """
    Subspace id for deterministic systems, algorithm 1.
    """
    assert f > 1
    assert p > 1
    y = pl.matrix(y)
    n_y = y.shape[0]
    u = pl.matrix(u)
    n_u = u.shape[0]
    w = pl.vstack([y, u])
    n_w = w.shape[0]

    W = block_hankel(w, f + p)
    U = block_hankel(u, f + p)
    Y = block_hankel(y, f + p)

    W_p = W[:n_w*p, :]
    W_pp = W[:n_w*(p+1), :]

    Y_f = Y[n_y*f:, :]
    U_f = U[n_y*f:, :]

    Y_fm = Y[n_y*(f+1):, :]
    U_fm = U[n_u*(f+1):, :]

    # step 1, calculate the oblique projections
    #------------------------------------------
    O_i = Y_f*project_oblique(U_f, W_p)
    O_im = Y_fm*project_oblique(U_fm, W_pp)

    # step 2, calculate the SVD of the weighted oblique projection
    #------------------------------------------
    W1 = pl.matrix(pl.eye(O_i.shape[0]))
    W2 = pl.matrix(pl.eye(O_i.shape[1]))

    # step 3, determine the order by inspecting the singular
    #------------------------------------------
    # values in S and partition the SVD accordingly to obtain U1, S1
    U0, s0, VT0 = pl.svd(W1*O_i*W2)  #pylint: disable=unused-variable
    #print s0
    n_x = pl.find(s0/s0.max() > s_tol)[-1] + 1
    U1 = U0[:, :n_x]
    # S1 = pl.matrix(pl.diag(s0[:n_x]))
    # VT1 = VT0[:n_x, :n_x]

    # step 4, determine Gi and Gim
    #------------------------------------------
    G_i = W1.I*U1*pl.matrix(pl.diag(pl.sqrt(s0[:n_x])))
    G_im = G_i[:-n_y, :] # check

    # step 5, determine Xd_ip and Xd_p
    #------------------------------------------
    Xd_i = G_i.I*O_i
    Xd_ip = G_im.I*O_im

    # step 6, solve the set of linear eqs
    # for A, B, C, D
    #------------------------------------------
    Y_ii = Y[n_y*p, :]
    U_ii = U[n_u*p, :]

    a_mat = pl.matrix(pl.vstack([Xd_ip, Y_ii]))
    b_mat = pl.matrix(pl.vstack([Xd_i, U_ii]))
    ss_mat = a_mat*b_mat.I
    A_id = ss_mat[:n_x, :n_x]
    B_id = ss_mat[:n_x, n_x:]
    C_id = ss_mat[n_x:, :n_x]
    D_id = ss_mat[n_x:, n_x:]

    if n_x == n_y:
        T = C_id.I # try to make C identity, want it to look like state feedback
    else:
        T = pl.matrix(pl.eye(n_x))

    Q_id = pl.zeros((n_x, n_x))
    R_id = pl.zeros((n_y, n_y))
    sys = ss.StateSpaceDiscreteLinear(
        A=T.I*A_id*T, B=T.I*B_id, C=C_id*T, D=D_id,
        Q=Q_id, R=R_id, dt=dt)
    return sys


def nrms(data_fit, data_true):
    """
    Normalized root mean square error.
    """
    # root mean square error
    rms = pl.mean(pl.norm(data_fit - data_true, axis=0))

    # normalization factor is the max - min magnitude, or 2 times max dist from mean
    norm_factor = 2*pl.norm(data_true - pl.mean(data_true, axis=1), axis=0).max()
    return (norm_factor - rms)/norm_factor

def prbs(n):
    """
    Pseudo random binary sequence.
    """
    return pl.where(pl.rand(n) > 0.5, 0, 1)


# vim: set et fenc=utf-8 ft=python  ff=unix sts=4 sw=4 ts=4 :
