
from pdata import*

dat = pdata('pflotran.in')

print 'nxyz', dat.grid.nxyz
print 'origin', dat.grid.origin
print 'dxyz', dat.grid.dxyz
print 'upper_bounds', dat.grid.upper_bounds
print 'gravity', dat.grid.gravity

print 'time', dat.time
print 'timestepper', dat.timestepper

'''
print 'filename',dat.filename

print 'tf',dat.time.tf
print 'dti',dat.time.dti
print 'dtf',dat.time.dtf

for prop in dat.proplist:
	print 'name',prop.name
	print 'id',prop.id
	print 'porosity',prop.porosity
	print 'tortuosity',prop.tortuosity
	print 'density',prop.density
	print 'specific_heat',prop.specific_heat
	print 'cond_dry',prop.cond_dry
	print 'cond_wet',prop.cond_wet
	print 'permeability',prop.permeability
'''
	
dat.write('pflotran2.in')
