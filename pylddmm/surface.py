# Several functions for working with surface data (.byu etc)
import numpy as np
# import matplotlib as mpl
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


def load_landmarks(fname):
    """
    Load landmarks from text file, where first line has size info

    Parameters
    ----------
    fname : str
        Filename of file containing landmarks as :math:`d`-rows of :math:`n` vertices

    Returns
    -------
    X : nparray
        (d x n) array containing :math:`d` :math:`n`-dimensional points

    Notes
    -----
    Courtesy Dr. Daniel Tward
    """
    with open(fname) as f:
        for i, line in enumerate(f):
            if i == 0:
                # the first line says the size of the file
                d, n = [int(n) for n in line.split()]
                X = np.empty((d, n))
                continue
            X[:, i - 1] = [float(n) for n in line.split()]
    return X


def load_rigid_matrix(fname):
    """
    Loads a .dat or .txt file into a rigid transformation matrix

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
       (3 x nv) vertices to perform transform on
    matrix : nparray
        (4 x 4) rigid transform array in homogenous coords

    Returns
    -------
    vertices_trans : nparray
        (3 x nv) transformed vertices
    """
    Vh = np.concatenate([vertices, np.ones((1, vertices.shape[1]))])

    Avh = np.dot(matrix, Vh)
    return(Avh[:3, :])


def reflect_surface(vertices, faces, axis=0):
    """
    Reflects a surface around a specified axis

    Parameters
    ----------
    vertices : nparray
        (3 x nv) array of vertex coordinates
    faces : nparray
        (3 x nf) array of triplets of vertices making triangular face

    Returns
    -------
    """
    reflect_matrix = np.eye(4)
    reflect_matrix[axis, axis] = -1
    return perform_rigid_transform(vertices, reflect_matrix), faces


def load_surface(fname):
    """
    Loads a .byu or .vtk file into vertices and faces, overloads load_byu and
    load_vtk

    Parameters
    ----------
    fname : str
       Filename with extension .byu or .vtk

    Returns
    -------
    vertices : nparray
        (3 x nv) array of vertex coordinates
    faces : nparray
        (3 x nf) array of triplets of vertices making triangular face
    """
    if fname[-4:] == '.byu':
        return load_byu(fname, arbitrary=False)
    elif fname[-4:] == 'vtk':
        print(".vtk functionality coming soon")


def load_byu(fname, arbitrary=False):
    """
    Loads a .byu file into vertices and faces

    Parameters
    ----------
    fname : str
       Filename with extension .byu
    arbitrary : bool
        Whether file format is arbitrary polygon

    Returns
    -------
    vertices : nparray
        (3 x nv) array of vertex coordinates
    faces : nparray
        (3 x nf) array of triplets of vertices making triangular face


    Notes
    -----
    Courtesy Dr. Daniel Tward
    """
    if not arbitrary:
        try:
            with open(fname) as f:
                for i, line in enumerate(f):
                    if i == 0:
                        # the first gives info about file
                        _, nv, nf, _ = [int(n) for n in line.split()]
                        vertices = np.empty((3, nv), dtype=float)
                        faces = np.empty((nf, 3), dtype=int)
                        continue
                    elif i == 1:
                        continue
                    if i <= nv + 1:
                        vertices[:, i - 2] = [float(n) for n in line.split()]
                    else:
                        faces[i - (nv + 2), :] = [np.abs(int(n)) -
                                                  1 for n in line.split()]
            return vertices, faces
        except:
            print("File is in original .byu format")
            return load_byu(fname, arbitrary=True)
    else:
        face_list = []
        with open(fname) as f:
            for i, line in enumerate(f):
                if i == 0:
                    # the first gives info about file
                    vals = [int(n) for n in line.split()]
                    ns, nv, nf, ne = vals[0], vals[1], vals[2], vals[3]
                    vertices = np.empty((3, nv), dtype=float)
                    faces = np.empty((nf, 3), dtype=int)

                    nv_count, nf_count = 0, 0
                    continue
                elif i >= 1 and i <= ns:
                    # the next ns lines define surfaces
                    continue
                elif i > ns and i <= ns + -(-nv // 2):
                    vals = line.split()
                    vertices[:, 2 * (i - ns - 1)] = [float(n)
                                                     for n in vals[:3]]
                    if len(vals) > 3:
                        vertices[:, 2 * (i - ns - 1) + 1] = [float(n)
                                                             for n in vals[3:]]
                else:
                    vals = [abs(int(n)) - 1 for n in line.split()]
                    face_list.extend(vals)

            face_list = np.asarray(face_list)
            faces = face_list.reshape((nf, 3))

            return vertices, faces


def save_surface(vertices, faces, fname, filetype='byu'):
    """
    Overloads `save_byu` and `save_vtk`

    Parameters
    ----------
    vertices : nparray
        (3 x nv) array of vertex coordinates
    faces : nparray
        (3 x nf) array of triplets of vertices making triangular face
    fname : str
       Output filename with extension .byu
    filetype : str
        Either '.byu' or '.vtk'
    """
    if filetype == 'byu':
        save_byu(vertices, faces, fname)
    elif filetype == 'vtk':
        print("vtk functionality coming soon")


def save_byu(vertices, faces, fname):
    """
    Saves vertices and faces as a .byu file

    Parameters
    ----------
    vertices : nparray
        (3 x nv) array of vertex coordinates
    faces : nparray
        (3 x nf) array of triplets of vertices making triangular face
    fname : str
       Output filename with extension .byu
    """
    nv = vertices.shape[1]
    nf = faces.shape[0]
    with open(fname, 'w+') as f:
        f.write('{} {} {} {}\n'.format(1, nv, nf, nf * 3))
        f.write('{} {}\n'.format(1, nf))
        for i in range(nv):
            f.write('{} {} {}\n'.format(vertices[0, i],
                                        vertices[1, i],
                                        vertices[2, i]))

        for i in range(nf):
            f.write('{} {} {}\n'.format(faces[i, 0] + 1,
                                        faces[i, 1] + 1,
                                        -1 * (faces[i, 2] + 1)))


def flip_surface_normals(fname, out_fname):
    """
    Loads a .byu file, and flips the surface normal directions.

    Parameters
    ----------
    fname : str
        Filename of .byu file
    out_fname : str
        Filename of output flipped .byu file
    """
    V, F = load_surface(fname)
    F[:, 0], F[:, 1] = F[:, 1], F[:, 0].copy()
    save_surface(V, F, out_fname)


def vol_from_surface(*args, **kwargs):
    """
    Calculates the volume of a surface either as .byu, .vtk, or (V, F) tuple

    Parameters
    ----------
    *args
        Specify either the filename or a pair of vertex, face arrays
    **kwargs
        Specify either `filename` or a pair of `V`, `F`

    Returns
    -------
    volume : float
        The volume of the surface
    """
    if len(args) == 2:
        V, F = args[0], args[1]
    elif len(args) == 1:
        V, F = load_surface(args[0])
    elif "filename" in kwargs:
        V, F = load_surface(kwargs['filename'])
    elif "V" in kwargs and "F" in kwargs:
        V, F = kwargs['V'], kwargs['F']
    else:
        print("Please input either a filename or vertices and faces")
    volume = np.sum(
        np.sum(np.cross(V[:, F[:, 0]].T,
                        V[:, F[:, 1]].T) * V[:, F[:, 2]].T)) / 6.0
    return volume


def vertex_area(*args, **kwargs):
    """
    Calculates the vertex areas of a surface either as .byu, .vtk, or (V, F) 
    tuple

    Parameters
    ----------
    *args
        Specify either the filename or a pair of vertex, face arrays
    **kwargs
        Specify either `filename` or a pair of `V`, `F`

    Returns
    -------
    v_areas : nparray
        (nv x 1) array containing vertex areas
    f_areas : nparray
        (nf x 1) array containing face areas
    """
    if len(args) == 2:
        V, F = args[0], args[1]
    elif len(args) == 1:
        V, F = load_surface(args[0])
    elif "filename" in kwargs:
        V, F = load_surface(kwargs['filename'])
    elif "V" in kwargs and "F" in kwargs:
        V, F = kwargs['V'], kwargs['F']
    else:
        print("Please input either a filename or vertices and faces")

    nV, nF = V.shape[1], F.shape[0]
    v_areas, f_areas = np.zeros((nV, 1)), np.zeros((nF, 1))

    for i in range(nF):
        a = np.sqrt(np.sum((V[:, F[i, 0]] - V[:, F[i, 1]])**2))
        b = np.sqrt(np.sum((V[:, F[i, 0]] - V[:, F[i, 2]])**2))
        c = np.sqrt(np.sum((V[:, F[i, 2]] - V[:, F[i, 1]])**2))
        s = (a+b+c)/2.0
        f_areas[i] = np.sqrt(s * (s-a) * (s-b) * (s-c))

    for i in range(nV):
        a, b = np.where(F == i)
        v_areas[i] = np.sum([f_areas[x] for x in a])/3.0

    return v_areas, f_areas


def axis_equal(ax=None):
    '''Set x,y,z axes to constant aspect ratio
    ax is a matplotlib 3d axex object'''
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    cx = np.mean(xlim)
    cy = np.mean(ylim)
    cz = np.mean(zlim)
    rx = np.diff(xlim)
    ry = np.diff(ylim)
    rz = np.diff(zlim)
    r = np.max([rx, ry, rz])
    ax.set_xlim(cx + np.array([-1, 1]) * 0.5 * r)
    ax.set_ylim(cy + np.array([-1, 1]) * 0.5 * r)
    ax.set_zlim(cz + np.array([-1, 1]) * 0.5 * r)


def plot_surface(vertices, faces, ax=None, **kwargs):
    """
    Plots a 3D surface given vertices and faces

    Parameters
    ----------
    vertices : nparray
        (3 x nv) array of vertex coordinates
    faces : nparray
        (3 x nf) array of triplets of vertices making triangular face

    Returns
    -------
    plot_trisurf : matplotlib plot_trisurf object


    Notes
    -----
    Courtesy Dr. Daniel Tward
    """
    ax = ax or plt.gca()

    return ax.plot_trisurf(
        # the x,y,z components of all the vertices
        vertices[0, :], vertices[1, :], vertices[2, :],
        triangles=faces,  # how are the vertices connected up into tri faces
        edgecolor='none',  # don't draw edges, this will look too busy
        **kwargs
    )


def byu_to_vtk(byu_filename, vtk_filename):
    """
    Converts .byu file to .vtk file

    Parameters
    ----------
    byu_filename: str
        Specify filename of .byu file
    vtk_filename: str
        Specify output filename of .vtk file
    """
    V, F = load_surface(byu_filename)
    nv, nf = V.shape[1], F.shape[0]
    with open(vtk_filename, 'w+') as f:
        f.write('# vtk DataFile Version 3.0\n')
        f.write('Surface Data\n')
        f.write('ASCII\n')
        f.write('DATASET POLYDATA\n\n')
        f.write('POINTS {} float\n'.format(nv))
        for i in range(nv):
            f.write('{} {} {}\n'.format(V[0, i], V[1, i], V[2, i]))

        f.write('POLYGONS {} {}\n'.format(nf, nf*4))
        for i in range(nf):
            f.write('3 {} {} {}\n'.format(F[i, 0]-1, F[i, 1]-1, F[i, 2]-1))


def load_R(filename):
    """
    Loads a 3x3 R matrix from file

    Parameters
    ----------
    filename: str
        Specify filename containing R data
    """
    R = np.zeros((3, 3))
    with open(filename) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            else:
                R[i-1, :] = [float(n) for n in line.split(' ')]

    return R


def load_T(filename):
    """
    Loads a 1x3 T vector from file

    Parameters
    ----------
    filename: str
        Specify filename containing T data
    """
    T = np.zeros((1, 3))
    with open(filename) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            else:
                T[0, :] = [float(n) for n in line.split(' ')]

    return T


def make_affine_from_RT(R_file, T_file):
    """
    Loads a 4x4 affine matrix from R and T files, overloads load_R and load_T

    Parameters
    ----------
    R_file: str
        Specify filename containing R data
    T_file: str
        Specify filename containing T data
    """
    R, T = load_R(R_file), load_T(T_file)
    aff_matrix = np.zeros((4, 4))
    aff_matrix[0:3, 0:3] = R
    aff_matrix[0:3, 3] = np.squeeze(T.T)
    aff_matrix[3, :] = [0.0, 0.0, 0.0, 1.0]
    return aff_matrix
