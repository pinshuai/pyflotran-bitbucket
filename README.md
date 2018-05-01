## About PyFLOTRAN ##
PyFLOTRAN is a set of classes and methods that the enable use of the massively parallel subsurface flow and reactive transport code PFLOTRAN (http://www.pflotran.org) within the Python scripting environment. This allows the use of a wide range of libraries available in Python for pre- and post- processing. The main advantages of using PyFLOTRAN include:

* PFLOTRAN input file construction from Python environment. Reduces the need to learn all the keywords in PFLOTRAN.
* Post-processing of output using matplotlib as well as including Python-based commandline calls to the open-source visualization toolkits such as VisIt and Paraview.
Scripting tools that supports Python's built-in multi-processing capabilities for batch simulations. This reduces the time spent in creating separate input files either for multiple realizations or multiple simulations for comparison.
* Streamlined workflow from pre-processing to post-processing. Typical workflow with PFLOTRAN involves -- pre-processing in Python or MATLAB, writing input files, executing the input files and then post-processing using matplotlib, VisIt or Paraview. PyFLOTRAN allows users to perform all these steps using one Python script.


## Installation ##

One can install PyFLOTRAN using ``pip`` as follows:

``pip install hg+https://satkarra@bitbucket.org/satkarra/pyflotran``

To remove PyFLOTRAN, type:

``pip uninstall pyflotran``