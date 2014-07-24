.. _pdata-chapter:

pdata: PFLOTRAN input file generation
=====================================

The pdata  module is the main module in PyFLOTRAN which contains classes
and methods to read, manipulate, write and execute PFLOTRAN input files.

pdata class
-----------

The :class:`.pdata` is the main class that has 

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


.. autoclass:: pdata.pmode

.. autoclass:: pdata.ptimestepper

.. autoclass:: pdata.plsolver

.. autoclass:: pdata.pnsolver

.. autoclass:: pdata.poutput

.. autoclass:: pdata.pfluid

.. autoclass:: pdata.psaturation

.. autoclass:: pdata.pregion
