# Several functions for transformations
import numpy as np
# import matplotlib as mpl
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


def linear_point(X, Y, transform, show_energy=True):
    """
    Calculate the optimal transform from landmarks X to Y

    Parameters
    ----------
    X : nparray
        (d x :math:`n_x`) array containing :math:`n` :math:`d`-dimensional points
    Y : nparray
        (d x :math:`n_y`) array containing :math:`n` :math:`d`-dimensional points 
    transform : str
        Either 'T', 's', 'R', or 'aff'
    show_energy : bool
        Display initial and final L2 norm

    Returns
    -------
    A : nparray
        (4 x 4) homogenous transformation matrix
    AX : nparray
        (d x :math:`n_x`) array containing transformed X
    """
    d = X.shape[0]
    A = np.eye(d + 1)
    AX = np.zeros((X.shape))
    if transform == 'T':
        t = np.mean(Y, axis=1) - np.mean(X, axis=1)
        A[:d, d] = np.squeeze(t)
        AX = X + t[:, None]
        print("calculated optimal translation t : ", t)
    if transform == 's':
        s = np.mean(X * Y) / np.mean(X * X)
        A *= s
        AX = X * s
        print("calculated optimal scaling : ", s)
    if transform == 'aff':
        Xh = np.concatenate([X, np.ones((1, X.shape[1]))], axis=0)
        Yh = np.concatenate([Y, np.ones((1, Y.shape[1]))], axis=0)

        A = np.dot(np.dot(Yh, Xh.T), np.linalg.inv(np.dot(Xh, Xh.T)))
        AXh = np.dot(A, Xh)
        AX = AXh[:d, :]
    if transform == 'R':
        U, s, Vt = np.linalg.svd(np.dot(Y, X.T))

        A[:d, :d] = np.dot(np.dot(U, np.sign(np.diag(s))), Vt)
        AX = np.dot(A[:d, :d], X)

    if show_energy:
        print("Original energy : ", np.sum(0.5 * (X - Y)**2))
        print("Final energy . : ", np.sum(0.5 * (AX - Y)**2))
    return A, AX


def load_rigid_matrix(fname):
    """
    Loads a .dat file into a rigid transformation matrix

    Parameters
    ----------
    fname : str
       Filename with extension .dat

    Returns
    -------
    matrix : nparray
        (4 x 4) matrix in homogeneous coordinates
    """
    matrix = np.empty((4, 4))
    with open(fname) as f:
        for i, line in enumerate(f):
            matrix[i, :] = line.split()

    matrix[3, :] = [0.0, 0.0, 0.0, 1.0]
    return matrix


def perform_rigid_transform(vertices, matrix):
    """
    Performs a rigid transform on vertices

    Parameters
    ----------
    vertices : nparray
       (d x nv) vertices to perform transform on
    matrix : nparray
        (d+1 x d+1) rigid transform array in homogenous coords

    Returns
    -------
    vertices_trans : nparray
        (3 x nv) transformed vertices
    """
    d = vertices.shape[0]
    Vh = np.concatenate([vertices, np.ones((1, vertices.shape[1]))])

    Avh = np.dot(matrix, Vh)
    return(Avh[:d, :])
