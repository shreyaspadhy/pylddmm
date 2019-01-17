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

.. autofunction:: pylddmm.surface.load_landmarks

.. autofunction:: pylddmm.surface.reflect_surface

.. autofunction:: pylddmm.surface.load_surface

.. autofunction:: pylddmm.surface.save_surface

.. autofunction:: pylddmm.surface.flip_surface_normals

.. autofunction:: pylddmm.surface.vol_from_byu

.. autofunction:: pylddmm.surface.extract_from_txt

.. autofunction:: pylddmm.surface.plot_surface

.. autofunction:: pylddmm.transform.load_rigid_matrix

.. autofunction:: pylddmm.transform.perform_rigid_transform

.. autofunction:: pylddmm.transform.linear_point
