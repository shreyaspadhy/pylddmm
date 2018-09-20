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


def load_surface(fname, arbitrary=False):
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
            return load_surface(fname, arbitrary=True)
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
                    vertices[:, 2 * (i - ns - 1)] = [float(n) for n in vals[:3]]
                    if len(vals) > 3:
                        vertices[:, 2 * (i - ns - 1) + 1] = [float(n) for n in vals[3:]]      
                else:
                    vals = [abs(int(n))-1 for n in line.split()]
                    face_list.extend(vals)
            
            face_list = np.asarray(face_list)
            faces = face_list.reshape((nf,3))
            
            return vertices, faces


def save_surface(vertices, faces, fname):
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
    F[:,0], F[:,1] = F[:,1], F[:,0].copy()
    save_surface(V, F, out_fname)
    
            
def vol_from_byu(*args,**kwargs):
    """
    Calculates the volume of a surface defined as a .byu file
    
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
    volume = np.sum(np.sum(np.cross(V[:,F[:,0]].T, V[:,F[:,1]].T) * V[:,F[:,2]].T)) / 6.0
    return volume


def extract_from_txt(filename):
    """
        Extracts left and right volumes with names from MRICloud Output text files

        Parameters
        ----------
        filename: str
            Name of .txt file output from MRICloud

        Returns
        -------
        vol_L: Pandas Dataframe
            Columns [vol_name, vol (mm^3)] for left volumes
        vol_R: Pandas Dataframe 
            Columns [vol_name, vol (mm^3)] for right volumes
    """
    lines = open(filename, 'r')
    table = []
    for i, line in enumerate(lines):
        if i >= 247 and i < 523:
            table.append(line.strip().split('\t'))

    table = pd.DataFrame(table)

    vol_list = table.iloc[:, 1:3]

    # print(vol_list)
    vol_L = vol_list[vol_list[1].str.contains('_L')]
    vol_R = vol_list[vol_list[1].str.contains('_R')]

    vol_L = vol_L[vol_L[1] != 'Chroid_LVetc_R']

    vol_L = vol_L.sort_values(by=[1])
    vol_R = vol_R.sort_values(by=[1])

    # Swap SFG and SFG_PFC in right
    a, b = vol_R.iloc[104].copy(), vol_R.iloc[105].copy()
    vol_R.iloc[104], vol_R.iloc[105] = b, a

    # Swap SFWM and SFWM_PFC in right
    a, b = vol_R.iloc[108].copy(), vol_R.iloc[109].copy()
    vol_R.iloc[108], vol_R.iloc[109] = b, a

    return vol_L, vol_R


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
    """
    ax = ax or plt.gca()

    return ax.plot_trisurf(
            vertices[0,:], vertices[1,:], vertices[2,:], # the x,y,z components of all the vertices
            triangles=faces, # how are the vertices connected up into triangular faces
            edgecolor='none', # don't draw edges, this will look too busy
            **kwargs
        )


def plot_grid(X0, X1, rstride=1, cstride=1, ax=None, **kwargs):
    '''
    plot a grid defined by X0 and X1,
    which are 2d arrays that should be the same size
    for example as output from meshgrid

    rstride defines how many rows to skip
    cstride defines how many columns to skip
    '''
    if ax is None:
        ax = plt.gca()
    # some defaults
    args = {
        'color': 'k'
    }
    args.update(kwargs)
    # plot rows
    for i in range(0, X0.shape[0], rstride):
        ax.plot(X0[i, :], X1[i, :], **args)
    if i < X0.shape[0] - 1:
        ax.plot(X0[-1, :], X1[-1, :], **args)
    # plot columns
    for j in range(0, X0.shape[1], cstride):
        ax.plot(X0[:, j], X1[:, j], **args)
    if j < X0.shape[1] - 1:
        ax.plot(X0[:, -1], X1[:, -1], **args)


# def reflect_surface(V, axis=0, )


def downsample_image(I, down):
    ''' downsample an image by averaging
    down should either be a triple, or a single number
    '''
    try:
        # check if its an iterable with 3 elements
        d0 = down[2]
    except TypeError:
        down = [down, down, down]
    down = np.array(down)
    nx = I.shape
    nxd = nx // down
    Id = np.zeros(nxd)
    for i in range(down[0]):
        for j in range(down[1]):
            for k in range(down[2]):
                Id += I[i:nxd[0] * down[0]:down[0], j:nxd[1] *
                        down[1]:down[1], k:nxd[2] * down[2]:down[2]]
    Id = Id / down[0] / down[1] / down[2]
    return Id


def sample_points_from_affine(X0, X1, X2, A):
    ''' From affine matrix A,
    and meshgrid domain X0, X1, X2,
    construct sample points to interpolate an image at to apply deformation
    '''
    B = np.linalg.inv(A)
    # get the sample points by matrix multiplication
    X0s = B[0, 0] * X0 + B[0, 1] * X1 + B[0, 2] * X2 + B[0, 3]
    X1s = B[1, 0] * X0 + B[1, 1] * X1 + B[1, 2] * X2 + B[1, 3]
    X2s = B[2, 0] * X0 + B[2, 1] * X1 + B[2, 2] * X2 + B[2, 3]
    return X0s, X1s, X2s

# simple function for drawing 3 slices


def imshow_slices(x0, x1, x2, I, axlist=None, **kwargs):
    ''' Draw three slices through the middle of an image'''
    if axlist is None:
        f, axlist = plt.subplots(1, 3)
    args = {'cmap': 'gray',
            'aspect': 'equal',
            'interpolation': 'none'}
    args.update(kwargs)

    h0 = axlist[0].imshow(np.squeeze(I[:, :, I.shape[2] // 2]),
                          extent=(x1[0], x1[-1], x0[0], x0[-1]), **args)
    axlist[0].set_xlabel('x1')
    axlist[0].set_ylabel('x0')
    h1 = axlist[1].imshow(np.squeeze(I[:, I.shape[1] // 2, :]),
                          extent=(x2[0], x2[-1], x0[0], x0[-1]), **args)
    axlist[1].set_xlabel('x2')
    axlist[1].set_ylabel('x0')
    h2 = axlist[2].imshow(np.squeeze(I[I.shape[0] // 2, :, :]),
                          extent=(x2[0], x2[-1], x1[0], x1[-1]), **args)
    axlist[2].set_xlabel('x2')
    axlist[2].set_ylabel('x1')
    return [h0, h1, h2]
