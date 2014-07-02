Introduction
============

PyFLOTRAN is a set of classes and methods that the enable use of the massively parallel subsurface flow and reactive transport code PFLOTRAN  within the Python scripting environment. This allows the use of a wide range of libraries available in Python for pre- and post- processing. The main advantages of using PyFLOTRAN include

1. PFLOTRAN input file construction from Python environment. Reduces the need to learn all the keywords in PFLOTRAN.    (see Chapter :ref:`3 <pdata-chapter>`).

2. Post-processing of output using matplotlib as well as including Python-based commandline calls to the open-source visualization toolkits such as VisIT and Paraview.

3. Scripting tools that supports Python's built-in multi-processing capabilities for batch simulations. This reduces the time spent in creating separate input files either for multiple realizations or multiple simulations for comparison.

4. Streamlined workflow from pre-processing to post-processing. Typical workflow with PFLOTRAN involves - pre-processing in Python or MATLAB, writing input files, executing the input files and then post-processing using matplotlib, VisIT or Paraview. PyFLOTRAN allows to perform all these steps using one Python script.


Introductory :ref:`tutorials <ftutorial-chapter>` to PyFLOTRAN are also provided.

Installation
------------

Python 
^^^^^^

PyFLOTRAN is supported on Python 2.6 and 2.7, but NOT 3.x. Instructions for downloading and installing Python can be
found at www.python.org

PyFLOTRAN requires the following Python modules to be installed: NumPy, SciPy, Matplotlib. For windows users, 32- and 64-bit installers (for several
Python versions) of these modules can be obtained from http://www.lfd.uci.edu/~gohlke/pythonlibs/

PyFLOTRAN
^^^^^^

A download link for the latest release version of PyFLOTRAN_ can be found at `pyflotran.lanl.gov`__

.. _PyFLOTRAN: pyflotran.lanl.gov

__ PyFLOTRAN_

To install, download and extract the zip file, and run the setup script at the command line: 

``python setup.py install``

PFLOTRAN
^^^^^^^^
PFLOTRAN is a massively parallel subsurface flow and reactive transport code. PFLOTRAN solves a system of partial differential equations for multiphase, multicomponent and multiscale reactive flow and transport in porous media. The code is designed to run on leadership-class supercomputers as well as workstations and laptops.

For successfully using PyFLOTRAN, one needs to install PFLOTRAN. For details to install PFLOTRAN please see the wikipage: https://bitbucket.org/pflotran/pflotran-dev/wiki/Home 


.. _PFLOTRAN: https://www.pflotran.org/

__ PFLOTRAN_
