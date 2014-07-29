.. _pdata-chapter:

pdata: PFLOTRAN input file generation
=====================================

The pdata  module is the main module in PyFLOTRAN which contains classes
and methods to read, manipulate, write and execute PFLOTRAN input files.

pdata class
-----------

The :class:`.pdata` is the main class that does the reading, writing, manipulation and execution of the PFLOTRAN input files. The other classes discussed in this section are defined to increase modularity and are used to set the attributes of :class:`.pdata`.

.. autoclass:: pdata.pdata

Attributes
^^^^^^^^^^
.. autoattribute:: pdata.pdata.filename
.. autoattribute:: pdata.pdata.work_dir

Methods
^^^^^^^
.. automethod:: pdata.pdata.read
.. automethod:: pdata.pdata.write
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

Observation
------
.. autoclass:: pdata.pobservation

Attributes
^^^^^^^^^^

Flow
------
.. autoclass:: pdata.pflow

Attributes
^^^^^^^^^^

Flow Variable
------
.. autoclass:: pdata.pflow_variable

Attributes
^^^^^^^^^^

Flow Variable List
------
.. autoclass:: pdata.pflow_variable_list

Attributes
^^^^^^^^^^

Initial Condition
------
.. autoclass:: pdata.pinitial_condition

Attributes
^^^^^^^^^^

Boundary Condition
------
.. autoclass:: pdata.pboundary_condition

Attributes
^^^^^^^^^^

Source Sink
------
.. autoclass:: pdata.psource_sink

Attributes
^^^^^^^^^^

Stratigraphy Coupler
------
.. autoclass:: pdata.pstrata

Attributes
^^^^^^^^^^

Checkpoint
------
.. autoclass:: pdata.pcheckpoint

Attributes
^^^^^^^^^^

Restart
------
.. autoclass:: pdata.prestart

Attributes
^^^^^^^^^^

Chemistry
------
.. autoclass:: pdata.pchemistry

Attributes
^^^^^^^^^^

Chemistry Mineral Kinetic
------
.. autoclass:: pdata.pchemistry_m_kinetic

Attributes
^^^^^^^^^^

Transport Condition
------
.. autoclass:: pdata.ptransport

Attributes
^^^^^^^^^^

Constraint Condition
------
.. autoclass:: pdata.pconstraint

Attributes
^^^^^^^^^^

Constraint Condition Concentration
------
.. autoclass:: pdata.pconstraint_concentration

Attributes
^^^^^^^^^^

Constraint Condition Mineral
------
.. autoclass:: pdata.pconstraint_mineral

Attributes
^^^^^^^^^^