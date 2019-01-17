.. Packaging Scientific Python documentation master file, created by
   sphinx-quickstart on Thu Jun 28 12:35:56 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyLDDMM Documentation
=====================

.. toctree::
   :maxdepth: 2

   installation
   usage
   release-history

=====
Usage
=====

Start by importing PyLDDMM.

.. code-block:: python

    import pylddmm

=======
Surface
=======

.. autofunction:: pylddmm.surface.load_landmarks

.. autofunction:: pylddmm.surface.load_rigid_matrix

.. autofunction:: pylddmm.surface.load_surface

.. autofunction:: pylddmm.surface.load_byu

.. autofunction:: pylddmm.surface.load_R

.. autofunction:: pylddmm.surface.load_T

.. autofunction:: pylddmm.surface.perform_rigid_transform

.. autofunction:: pylddmm.surface.flip_surface_normals

.. autofunction:: pylddmm.surface.reflect_surface

.. autofunction:: pylddmm.surface.make_affine_from_RT

.. autofunction:: pylddmm.surface.save_surface

.. autofunction:: pylddmm.surface.save_byu

.. autofunction:: pylddmm.surface.byu_to_vtk

.. autofunction:: pylddmm.surface.vol_from_surface

.. autofunction:: pylddmm.surface.vertex_area

.. autofunction:: pylddmm.surface.plot_surface

======
Volume
======

.. autofunction:: pylddmm.volume.extract_from_txt

=========
Utilities
=========

.. autofunction:: pylddmm.utils.run_bash_command
=======
