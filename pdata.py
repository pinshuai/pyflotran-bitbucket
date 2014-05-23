"""Class for pflotran data"""

import numpy as np
from copy import deepcopy
from copy import copy
import os,time
import platform

from ptool import*

WINDOWS = platform.system()=='Windows'
if WINDOWS: copyStr = 'copy'; delStr = 'del'; slash = '\\'
else: copyStr = 'cp'; delStr = 'rm'; slash = '/'

cards = ['mode','grid','timestepper','material_property','time']
headers = ['mode','grid','time stepping','material properties',
		   'time']
headers = dict(zip(cards,headers))

class pmaterial(object):
	""" Class for material property card

	"""

	def __init__(self,id,name,porosity=0.1,tortuosity=0.1,density=2.5e3,specific_heat=1.e3,cond_dry=0.5, cond_wet = 0.5,
				 saturation='sf2',permeability = [1.e-15,1.e-15,1.e-15]):
		self._id = id
		self._name = name
		self._porosity = porosity
		self._tortuosity = tortuosity
		self._density = density
		self._specific_heat = specific_heat
		self._cond_dry = cond_dry
		self._cond_wet = cond_wet
		self._saturation = saturation
		self._permeability = permeability

	def _get_id(self): return self._id
	def _set_id(self,value): self._id = value
	id = property(_get_id, _set_id) #: (**)
	def _get_name(self): return self._name
	def _set_name(self,value): self._name = value
	name = property(_get_name, _set_name) #: (**)
	def _get_porosity(self): return self._porosity
	def _set_porosity(self,value): self._porosity = value
	porosity = property(_get_porosity, _set_porosity) #: (**)
	def _get_tortuosity(self): return self._tortuosity
	def _set_tortuosity(self,value): self._tortuosity = value
	tortuosity = property(_get_tortuosity, _set_tortuosity) #: (**)
	def _get_density(self): return self._density
	def _set_density(self,value): self._density = value
	density = property(_get_density, _set_density) #: (**)
	def _get_specific_heat(self): return self._specific_heat
	def _set_specific_heat(self,value): self._specific_heat = value
	specific_heat = property(_get_specific_heat, _set_specific_heat) #: (**)
	def _get_cond_dry(self): return self._cond_dry
	def _set_cond_dry(self,value): self._cond_dry = value
	cond_dry = property(_get_cond_dry, _set_cond_dry) #: (**)
	def _get_cond_wet(self): return self._cond_wet
	def _set_cond_wet(self,value): self._cond_wet = value
	cond_wet = property(_get_cond_wet, _set_cond_wet) #: (**)
	def _get_saturation(self): return self._saturation
	def _set_saturation(self,value): self._saturation = value
	saturation = property(_get_saturation, _set_saturation) #: (**)
	def _get_permeability(self): return self._permeability
	def _set_permeability(self,value): self._permeability = value
	permeability = property(_get_permeability, _set_permeability) #: (**)
class ptime(object):
	""" Class for time property

	"""

	def __init__(self,tf=10.,dti=1.e-2,dtf=50.):
		self._tf = tf
		self._dti = dti
		self._dtf = dtf

	def _get_tf(self): return self._tf
	def _set_tf(self,value): self._tf = value
	tf = property(_get_tf, _set_tf) #: (**)
	def _get_dti(self): return self._dti
	def _set_dti(self,value): self._dti = value
	dti = property(_get_dti, _set_dti) #: (**)
	def _get_dtf(self): return self._dtf
	def _set_dtf(self,value): self._dtf = value
	dtf = property(_get_dtf, _set_dtf) #: (**)
class pgrid(object):
	""" Class for grid property

	"""

	def __init__(self,type='structured',lower_bounds=[0.0,0.0,0.0],upper_bounds=[50.0,50.0,50.0],
				 bounds_bool=True,origin=[0.0,0.0,0.0],nxyz=[10,10,10],dxyz=[5,5,5],gravity_bool=False,
				 gravity=[0.0,0.0,-9.8068],filename=''):
		self._type = type
		self._lower_bounds = lower_bounds
		self._upper_bounds = upper_bounds
		self._origin = origin
		self._nxyz = nxyz
		self._dxyz = dxyz
		self._gravity = gravity
		self._filename = filename
		self._bounds_bool = bounds_bool
		self._gravity_bool = gravity_bool

	def _get_type(self): return self._type
	def _set_type(self,value): self._type = value
	type = property(_get_type, _set_type) #: (**)
	def _get_lower_bounds(self): return self._lower_bounds
	def _set_lower_bounds(self,value): self._lower_bounds = value
	lower_bounds = property(_get_lower_bounds, _set_lower_bounds) #: (**)
	def _get_upper_bounds(self): return self._upper_bounds
	def _set_upper_bounds(self,value): self._upper_bounds = value
	upper_bounds = property(_get_upper_bounds, _set_upper_bounds) #: (**)
	def _get_origin(self): return self._origin
	def _set_origin(self,value): self._origin = value
	origin = property(_get_origin, _set_origin) #: (**)
	def _get_nxyz(self): return self._nxyz
	def _set_nxyz(self,value): self._nxyz = value
	nxyz = property(_get_nxyz, _set_nxyz) #: (**)
	def _get_dxyz(self): return self._dxyz
	def _set_dxyz(self,value): self._dxyz = value
	dxyz = property(_get_dxyz, _set_dxyz) #: (**)
	def _get_gravity(self): return self._gravity
	def _set_gravity(self,value): self._gravity = value
	gravity = property(_get_gravity, _set_gravity) #: (**)
	def _get_filename(self): return self._filename
	def _set_filename(self,value): self._filename = value
	filename = property(_get_filename, _set_filename) #: (**)
	def _get_bounds_bool(self): return self._bounds_bool
	def _set_bounds_bool(self,value): self._bounds_bool = value
	bounds_bool = property(_get_bounds_bool, _set_bounds_bool) #: (**)
	def _get_gravity_bool(self): return self._gravity_bool
	def _set_gravity_bool(self,value): self._gravity_bool = value
	gravity_bool = property(_get_gravity_bool, _set_gravity_bool) #: (**)
class pmode(object):
	""" Class for mode

	"""

	def __init__(self,name='richards'):
		self._name = name

	def _get_name(self): return self._name
	def _set_name(self,value): self._name = value
	name = property(_get_name, _set_name)
class ptimestepper(object):
	""" Class for time stepper

	"""

	def __init__(self,ts_acceleration=None,num_steps_after_cut=None,max_steps=None,max_ts_cuts=None,
                          cfl_limiter=None,initialize_to_steady_state=False,run_as_steady_state=False,
                          max_pressure_change=None,max_temperature_change=None,max_concentration_change=None,
                          max_saturation_change=None):
		self._ts_acceleration = ts_acceleration
		self._num_steps_after_cut = num_steps_after_cut 
		self._max_steps = max_steps
		self._max_ts_cuts = max_ts_cuts
		self._cfl_limiter = cfl_limiter
		self._initialize_to_steady_state = initialize_to_steady_state
		self._run_as_steady_state = run_as_steady_state
		self._max_pressure_change = max_pressure_change
		self._max_temperature_change = max_temperature_change
		self._max_concentration_change = max_concentration_change
		self._max_saturation_change = max_saturation_change

	def _get_ts_acceleration(self): return self._ts_acceleration
	def _set_ts_acceleration(self,value): self._ts_acceleration = value
	ts_acceleration = property(_get_ts_acceleration, _set_ts_acceleration)
	def _get_num_steps_after_cut(self): return self._num_steps_after_cut
	def _set_num_steps_after_cut(self,value): self._num_steps_after_cut = value
	num_steps_after_cut = property(_get_num_steps_after_cut, _set_num_steps_after_cut)
	def _get_max_steps(self): return self._max_steps
	def _set_max_steps(self,value): self._max_steps = value
	max_steps = property(_get_max_steps, _set_max_steps)
	def _get_max_ts_cuts(self): return self._max_ts_cuts
	def _set_max_ts_cuts(self,value): self._max_ts_cuts = value
	max_ts_cuts = property(_get_max_ts_cuts, _set_max_ts_cuts)
	def _get_cfl_limiter(self): return self._cfl_limiter
	def _set_cfl_limiter(self,value): self._cfl_limiter = value
	cfl_limiter = property(_get_cfl_limiter, _set_cfl_limiter)
	def _get_initialize_to_steady_state(self): return self._initialize_to_steady_state
	def _set_initialize_to_steady_state(self,value): self._initialize_to_steady_state = value
	initialize_to_steady_state = property(_get_initialize_to_steady_state, _set_initialize_to_steady_state)
	def _get_run_as_steady_state(self): return self._run_as_steady_state
	def _set_run_as_steady_state(self,value): self._run_as_steady_state = value
	run_as_steady_state = property(_get_run_as_steady_state, _set_run_as_steady_state)
	def _get_max_pressure_change(self): return self._max_pressure_change
	def _set_max_pressure_change(self,value): self._max_pressure_change = value
	max_pressure_change = property(_get_max_pressure_change, _set_max_pressure_change)
	def _get_max_temperature_change(self): return self._max_temperature_change
	def _set_max_temperature_change(self,value): self._max_temperature_change = value
	max_temperature_change = property(_get_max_temperature_change, _set_max_temperature_change)
	def _get_max_concentration_change(self): return self._max_concentration_change
	def _set_max_concentration_change(self,value): self._max_concentration_change = value
	max_concentration_change = property(_get_max_concentration_change, _set_max_concentration_change)
	def _get_max_saturation_change(self): return self._max_saturation_change
	def _set_max_saturation_change(self,value): self._max_saturation_change = value
	max_saturation_change = property(_get_max_saturation_change, _set_max_saturation_change)
class pdata(object):
	"""Class for pflotran data file

	"""

	def __init__(self, filename=None):
		from copy import copy
		self._filename = filename
		self._proplist = []
		self._time = ptime()
		self._mode = ''
		self._grid = pgrid()
		self._timestepper = ptimestepper()

		if filename: self.read(filename) 		# read in file if requested upon initialisation
	def __repr__(self): return self.filename 	# print to screen when called
	def read(self, filename):
		if not os.path.isfile(filename): print filename + ' not found...'
		self._filename = filename 				# assign filename attribute
		read_fn = dict(zip(cards, 				# associate each card name with a read function, defined further below
				[self._read_mode,
				 self._read_grid,
				 self._read_timestepper,
				 self._read_prop,
				 self._read_time]
				 ))
		with open(self._filename,'r') as infile:
			keepReading = True
			while keepReading:
				line = infile.readline()
#				print line.strip()
				if not line: keepReading = False
				if len(line.strip())==0: continue
				card = line.split()[0].lower() 		# make card lower case
				if card in cards: 					# check if a valid cardname
					if card in ['material_property','mode','grid','timestepper']:
						read_fn[card](infile,line)
					else:
						read_fn[card](infile)
	def write(self, filename=None):
		"""Write pdata object to pflotran input file.

		:param filename: Name of pflotran input file.
		:type filename: str
		"""
		if filename: self._filename = filename
		outfile = open(self.filename,'w')
		if self.mode: self._write_mode(outfile)
		if self.grid: self._write_grid(outfile)
		if self.timestepper : self._write_timestepper(outfile)
		if self.time: self._write_time(outfile)
		if self.proplist: self._write_prop(outfile)
		outfile.close()
	def _read_mode(self,infile,line):
		mode_name = line.split()[-1]
		new_mode = pmode(mode_name)
		self._mode = new_mode
	def _write_mode(self,outfile):
		self._header(outfile,headers['mode'])
		outfile.write('MODE\t')
		outfile.write(self.mode.name+'\n\n')
	def _read_grid(self,infile,line):
		g = pgrid()				# assign defaults before reading in values
		ng_type = g.type
		ng_lower_bounds = g.lower_bounds
		ng_upper_bounds = g.upper_bounds
		ng_origin = g.origin
		ng_nxyz = g.nxyz
		ng_gravity = g.gravity
		ng_filename = g.filename
		ng_dxyz = g.dxyz

		keepReading = True
		bounds_key = False
		gravity_key = False
		while keepReading:
			line = infile.readline() 			# get next line
			key = line.strip().split()[0].lower() 		# take first keyword
			if key in ['#']:
				pass
			if key == 'type':
				ng_type = line.split()[-1]
			elif key == 'bounds':
				bounds_key = True
				keepReading2 = True
				while keepReading2:
					line1 = infile.readline()
					ng_lower_bounds[0] = floatD(line1.split()[0])
					ng_lower_bounds[1] = floatD(line1.split()[1])
					ng_lower_bounds[2] = floatD(line1.split()[2])
					line2 = infile.readline()
					ng_upper_bounds[0] = floatD(line2.split()[0])
					ng_upper_bounds[1] = floatD(line2.split()[1])
					ng_upper_bounds[2] = floatD(line2.split()[2])
					line3 = infile.readline()
					if line3.strip().split()[0].lower() in ['/','end']: keepReading2 = False
			elif key == 'origin':
				ng_origin[0] = floatD(line.strip().split()[1])
				ng_origin[1] = floatD(line.strip().split()[2])
				ng_origin[2] = floatD(line.strip().split()[3])
			elif key == 'nxyz':
				ng_nxyz[0] = floatD(line.split()[1])
				ng_nxyz[1] = floatD(line.split()[2])
				ng_nxyz[2] = floatD(line.split()[3])
			elif key == 'gravity':
				gravity_key = True
				ng_gravity[0] = floatD(line.split()[1])
				ng_gravity[1] = floatD(line.split()[2])
				ng_gravity[2] = floatD(line.split()[3])
			elif key == 'filename':
				if ng_type != 'unstructured': print 'Error - filename not need with structure grid'; return
				ng_filename = line.split()[-1]
			elif key == 'dxyz':
				if bounds_key: print 'Error - specify either bounds of dxyz'; return
				keepReading2 = True
				count = 0
				while keepReading2:
					line = infile.readline()
					if line.strip().split()[0].lower() in ['/','end']:
						keepReading2 = False
					else:
						ng_dxyz[count] = floatD(line.strip().split()[0])
						count = count + 1
			elif key in ['/','end']: keepReading = False
		new_grid = pgrid(ng_type,ng_lower_bounds,ng_upper_bounds,bounds_key,ng_origin,ng_nxyz,
						 ng_dxyz,gravity_key,ng_gravity,ng_filename) 		# create an empty grid
		self._grid = new_grid
	def _write_grid(self,outfile):
		self._header(outfile,headers['grid'])
		grid = self.grid
		outfile.write('GRID\n')
		outfile.write('\tTYPE\t' + grid.type + '\n')
		if grid.bounds_bool:
			outfile.write('\tBOUNDS\n')
			outfile.write('\t\t')
			for i in range(3):
				outfile.write(str(grid.lower_bounds[i]) + ' ')
			outfile.write('\n')
			outfile.write('\t\t')
			for i in range(3):
				outfile.write(str(grid.upper_bounds[i]) + ' ')
			outfile.write('\n')
			outfile.write('\tEND\n')
		else:
			outfile.write('\tDXYZ\n')
			for i in range(3):
				outfile.write('\t\t' + str(grid.dxyz[i]) + '\n')
			outfile.write('\tEND\n')
		outfile.write('\tORIGIN' + ' ')
		for i in range(3):
			outfile.write(str(grid.origin[i]) + ' ')
		outfile.write('\n')
		outfile.write('\tNXYZ' + ' ')
		for i in range(3):
			outfile.write(str(grid.nxyz[i]) + ' ')
		outfile.write('\n')
		if grid.gravity_bool:
			outfile.write('\tGRAVITY' + ' ')
			for i in range(3):
				outfile.write(str(grid.gravity[i]) + ' ')
			outfile.write('\n')
		if grid.type == 'unstructured':
			outfile.write('\tFILENAME' + grid.filename + '\n')
		outfile.write('END\n\n')
	def _read_timestepper(self,infile,line):
		p = ptimestepper()
		np_ts_acceleration = p.ts_acceleration
		np_num_steps_after_cut = p.num_steps_after_cut 
		np_max_steps = p.max_steps
		np_max_ts_cuts = p.max_ts_cuts
		np_cfl_limiter = p.cfl_limiter
		np_initialize_to_steady_state = p.initialize_to_steady_state
		np_run_as_steady_state = p.run_as_steady_state
		np_max_pressure_change = p.max_pressure_change
		np_max_temperature_change = p.max_temperature_change
		np_max_concentration_change = p.max_concentration_change
		np_max_saturation_change = p.max_saturation_change

		keepReading = True

		while keepReading: 			# read through all cards
			line = infile.readline() 			# get next line
			key = line.strip().split()[0].lower() 		# take first keyword
			if key == 'ts_acceleration':
				np_ts_acceleration = int(line.split()[-1])
			elif key == 'num_steps_after_cut':
				np_num_steps_after_cut = int(line.split()[-1])
			elif key == 'max_steps':
				np_max_steps = int(line.split()[-1])
			elif key == 'max_ts_cuts':
				np_max_ts_cuts = int(line.split()[-1])
			elif key == 'cfl_limiter':
				np_clf_limiter = float(line.split()[-1])
			elif key == 'initialize_to_steady_state':
				np_initialize_to_steady_state = True 
			elif key == 'run_as_steady_state':
				np_run_as_steady_state = True 
			elif key == 'max_pressure_change':
				np_max_pressure_change = float(line.split()[-1])
			elif key == 'max_temperature_change':
				np_max_temperature_change = float(line.split()[-1]) 
			elif key == 'max_concentration_change':
				np_max_concentration_change = float(line.split()[-1])
 			elif key == 'max_saturation_change':
				np_max_saturation_change = float(line.split()[-1])
			elif key in ['/','end']: keepReading = False

		new_timestep = ptimestepper(np_ts_acceleration,np_num_steps_after_cut,np_max_steps,
					    np_max_ts_cuts,np_cfl_limiter,np_initialize_to_steady_state,
					    np_run_as_steady_state,np_max_pressure_change,
					    np_max_temperature_change,np_max_concentration_change,
		                            np_max_saturation_change)

		self._timestepper = new_timestep
	def _write_timestepper(self,outfile):
		self._header(outfile,headers['timestepper'])
		outfile.write('TIMESTEPPER\n')
		if self.timestepper.ts_acceleration:
                  outfile.write('\t' + 'TS_ACCELERATION ' + str(self.timestepper.ts_acceleration) + '\n')
		if self.timestepper.num_steps_after_cut:
                  outfile.write('\t' + 'NUM_STEPS_AFTER_CUT ' + str(self.timestepper.num_steps_after_cut) + '\n')
		outfile.write('END\n')
	def _read_prop(self,infile,line):
		np_name = line.split()[-1] 		# property name
		np_id = None
		p = pmaterial(0,'')				# assign defaults before reading in values
		np_porosity=p.porosity
		np_tortuosity=p.tortuosity
		np_density=p.density
		np_specific_heat=p.specific_heat
		np_cond_dry=p.cond_dry
		np_cond_wet=p.cond_wet
		np_saturation=p.saturation
		np_permeability=p.permeability

		keepReading = True

		while keepReading: 			# read through all cards
			line = infile.readline() 			# get next line
			key = line.strip().split()[0].lower() 		# take first keyword
			if key == 'id':
				np_id = int(line.split()[-1])
			elif key == 'porosity':
				np_porosity = floatD(line.split()[-1])
			elif key == 'tortuosity':
				np_tortuosity = floatD(line.split()[-1])
			elif key == 'rock_density':
				np_density = floatD(line.split()[-1])
			elif key == 'specific_heat':
				np_specific_heat = floatD(line.split()[-1])
			elif key == 'thermal_conductivity_dry':
				np_cond_dry = floatD(line.split()[-1])
			elif key == 'thermal_conductivity_wet':
				np_cond_wet = floatD(line.split()[-1])
			elif key == 'saturation_function':
				np_saturation = line.split()[-1]
			elif key == 'permeability':
				keepReading2 = True
				while keepReading2:
					line = infile.readline() 			# get next line
					key = line.split()[0].lower() 		# take first keyword
					if key == 'perm_x':
						np_permeability[0] = floatD(line.split()[-1])
					elif key == 'perm_y':
						np_permeability[1] = floatD(line.split()[-1])
					elif key == 'perm_z':
						np_permeability[2] = floatD(line.split()[-1])
					elif key in ['/','end']: keepReading2 = False
			elif key in ['/','end']: keepReading = False
		new_prop = pmaterial(np_id,np_name,np_porosity,np_tortuosity,np_density,
		                     np_specific_heat,np_cond_dry,np_cond_wet,
				             np_saturation,np_permeability) 		# create an empty material property

		self._proplist.append(new_prop)
	def _write_prop(self,outfile):
		self._header(outfile,headers['material_property'])
		for prop in self.proplist:
			outfile.write('MATERIAL_PROPERTY\t')
			outfile.write(prop.name+'\n')
			outfile.write('\tID\t'+str(prop.id)+'\n')
			outfile.write('\tPOROSITY\t'+str(prop.porosity)+'\n')
			outfile.write('\tTORTUOSITY\t'+str(prop.tortuosity)+'\n')
			outfile.write('\tROCK_DENSITY\t'+str(prop.density)+'\n')
			outfile.write('\tSPECIFIC_HEAT\t'+str(prop.specific_heat)+'\n')
			outfile.write('\tTHERMAL_CONDUCTIVITY_DRY\t'+str(prop.cond_dry)+'\n')
			outfile.write('\tTHERMAL_CONDUCTIVITY_WET\t'+str(prop.cond_wet)+'\n')
			outfile.write('\tSATURATION_FUNCTION\t'+prop.saturation+'\n')
			outfile.write('\tPERMEABILITY\n')
			outfile.write('\t\tPERM_X\t'+strD(prop.permeability[0])+'\n')
			outfile.write('\t\tPERM_Y\t'+strD(prop.permeability[1])+'\n')
			outfile.write('\t\tPERM_Z\t'+strD(prop.permeability[2])+'\n')
			outfile.write('\tEND\n')
			outfile.write('END\n\n')
	def _read_time(self,infile):
		keepReading = True
		while keepReading:
			line = infile.readline() 			# get next line
			key = line.split()[0].lower() 		# take first keyword
			if key == 'final_time':
				tstring = line.split()[1:]
				if len(tstring) == 2:
					if tstring[-1] == 'y':
						self._tf = floatD(tstring[0])*365.25*24*3600
				else:
					self._tf = floatD(tstring[0])
			elif key == 'initial_timestep_size':
				tstring = line.split()[1:]
				if len(tstring) == 2:
					if tstring[-1] == 'y':
						self._dti = floatD(tstring[0])*365.25*24*3600
				else:
					self._dti = floatD(tstring[0])
			elif key == 'maximum_timestep_size':
				tstring = line.split()[1:]
				if len(tstring) == 2:
					if tstring[-1] == 'y':
						self._dtf = floatD(tstring[0])*365.25*24*3600
				else:
					self._dtf = floatD(tstring[0])
			elif key in ['/','end']: keepReading = False
	def _write_time(self,outfile):
		self._header(outfile,headers['time'])
		outfile.write('TIME\n')
		outfile.write('\tFINAL_TIME\t')
		if self.time.tf>365.25*3600*24*0.1:
			outfile.write(str(self.time.tf/(365.25*24*3600))+' y\n')
		else:
			outfile.write(str(self.time.tf)+'\n')
		outfile.write('\tINITIAL_TIMESTEP_SIZE\t')
		if self.time.dti>365.25*3600*24*0.1:
			outfile.write(str(self.time.dti/(365.25*24*3600))+' y\n')
		else:
			outfile.write(str(self.time.dti)+'\n')
		outfile.write('\tMAXIMUM_TIMESTEP_SIZE\t')
		if self.time.dtf>365.25*3600*24*0.1:
			outfile.write(str(self.time.dtf/(365.25*24*3600))+' y\n')
		else:
			outfile.write(str(self.time.dtf)+'\n')
		outfile.write('END\n\n')
	def _header(self,outfile,header):
		if not header: return
		ws = '# '
		pad = int(np.floor((80 - len(header) - 4)/2))
		for i in range(pad): ws+='='
		ws+=' '+header+' '
		for i in range(pad): ws+='='
		ws+='\n'
		outfile.write(ws)
	def _get_grid(self): return self._grid
	grid = property(_get_grid) #: (**)
	def _get_time(self): return self._time
	time = property(_get_time) #: (**)
	def _get_proplist(self): return self._proplist
	proplist = property(_get_proplist) #: (**) list of material properties
	def _get_prop(self): return dict([(p.id,p) for p in self.proplist]+[(p.name,p) for p in self.proplist])
	prop = property(_get_prop) #: (**) dictionary of material properties, indexable by ID or name
	def _get_filename(self): return self._filename
	def _set_filename(self,value): self._filename = value
	filename = property(_get_filename, _set_filename) #: (**)
	def _get_mode(self): return self._mode
	mode = property(_get_mode) #: (**)
	def _get_timestepper(self): return self._timestepper
	timestepper = property(_get_timestepper) #: (**)
