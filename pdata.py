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

cards = ['material_property','time']
headers = ['material properties',
		   'time stepping']
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
class pdata(object):
	"""Class for pflotran data file
	
	"""
	
	def __init__(self, filename=None):
		from copy import copy
		self._filename = filename 		
		self._proplist = []
		self._time = ptime()
		
		if filename: self.read(filename) 		# read in file if requested upon initialisation
	def __repr__(self): return self.filename 	# print to screen when called
	def read(self, filename):
		if not os.path.isfile(filename): print filename + ' not found...'
		self._filename = filename 				# assign filename attribute
		read_fn = dict(zip(cards, 				# associate each card name with a read function, defined further below
				[self._read_prop,
				 self._read_time]
				 ))
		with open(self._filename,'r') as infile:
			keepReading = True
			while keepReading:
				line = infile.readline()
				print line.strip()
				if not line: keepReading = False
				if len(line.strip())==0: continue
				card = line.split()[0].lower() 		# make card lower case
				if card in cards: 					# check if a valid cardname
					if card == 'material_property':
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
		if self.time: self._write_time(outfile)
		if self.proplist: self._write_prop(outfile)
		outfile.close()
	def _read_prop(self,infile,line):
		np_name = line.split()[-1] 		# property name
		np_id=None
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
		new_prop = pmaterial(np_id,np_name,np_porosity,np_tortuosity,np_density,np_specific_heat,np_cond_dry,np_cond_wet,
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
		ws = ': '
		pad = int(np.floor((80 - len(header) - 4)/2))
		for i in range(pad): ws+='='
		ws+=' '+header+' '
		for i in range(pad): ws+='='
		ws+='\n'
		outfile.write(ws)
	def _get_time(self): return self._time
	time = property(_get_time) #: (**)
	def _get_proplist(self): return self._proplist
	proplist = property(_get_proplist) #: (**) list of material properties
	def _get_prop(self): return dict([(p.id,p) for p in self.proplist]+[(p.name,p) for p in self.proplist])
	prop = property(_get_prop) #: (**) dictionary of material properties, indexable by ID or name
	def _get_filename(self): return self._filename
	def _set_filename(self,value): self._filename = value
	filename = property(_get_filename, _set_filename) #: (**)	
		
		
		