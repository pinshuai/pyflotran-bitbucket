.. _pdata-chapter:

pdata: PFLOTRAN input file generation
=====================================

The pdata  module is the main module in PyFLOTRAN which contains classes
and methods to read, manipulate, write and execute PFLOTRAN input files.

pdata class
-----------

The :class:`.pdata` is the main class that does the reading, writing, manipulation and execution of the PFLOTRAN input files.

.. autoclass:: pdata.pdata

Attributes
^^^^^^^^^^

Methods
^^^^^^^
.. automethod:: pdata.pdata.run


Grid
----
.. autoclass:: pdata.pgrid

Attributes
^^^^^^^^^^

Material 
--------
.. autoclass:: pdata.pmaterial

Attributes
^^^^^^^^^^

Time
----
.. autoclass:: pdata.ptime

Attributes
^^^^^^^^^^

Uniform Velocity
----------------
.. autoclass:: pdata.puniform_velocity

Attributes
^^^^^^^^^^

Mode
----
.. autoclass:: pdata.pmode

Attributes
^^^^^^^^^^

Timestepper
-----------
.. autoclass:: pdata.ptimestepper

Attributes
^^^^^^^^^^

Linear Solver
-------------
.. autoclass:: pdata.plsolver

Attributes
^^^^^^^^^^

Newton Solver
-------------
.. autoclass:: pdata.pnsolver

Attributes
^^^^^^^^^^

Output
------
.. autoclass:: pdata.poutput

Attributes
^^^^^^^^^^

Fluid Properties
----------------
.. autoclass:: pdata.pfluid

Attributes
^^^^^^^^^^

Saturation Function
-------------------
.. autoclass:: pdata.psaturation

Attributes
^^^^^^^^^^

Region
------
.. autoclass:: pdata.pregion

Attributes
^^^^^^^^^^


