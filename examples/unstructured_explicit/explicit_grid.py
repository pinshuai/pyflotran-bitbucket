from pdata import *

grid = pgrid(type='unstructured_explicit', filename='mixed.uge')
print 'cell list:'
print grid.celllist
print 'connectivity:'
print grid.connectivity
print 'num cells:'
print grid.nxyz
print 'lower bound x-direction:'
print grid.xmin
print 'lower bound y-direction:'
print grid.ymin
print 'lower bound z-direction:'
print grid.zmin
print 'upper bound x-direction:'
print grid.xmax
print 'upper bound y-direction:'
print grid.ymax
print 'upper bound z-direction:'
print grid.zmax
