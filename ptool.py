"""Tools for use with pflotran python"""

import numpy as np
import os,platform,string
WINDOWS = platform.system()=='Windows'
if WINDOWS: slash = '\\'
else: slash = '/'

from pdflt import*
dflt = pdflt()

# Suggested for determining file path when reading an input file in-case it is relative
#def produce_valid_file_path(path):
#	# Return the string without modifications if it's an absolute file path
#	if path[0] == '/':  
#		return path
#	# Determine PLOFTRAN installation directory and made appropriate changes to string
#	else:
#		try:
#			pflotran_dir = os.environ['PFLOTRAN_DIR']
#		except KeyError:
#			print('PFLOTRAN_DIR must point to PFLOTRAN installation directory and be defined in system environment variables.')

# Simple Function to determine if a '/' already exists somewhere in a relative file path name
# Don't use with absolute file path name
# NOT TESTED WITH WINDOWS
def del_extra_slash(path): return path.replace(slash+slash, slash)
	

def floatD(string):
	"""Converts input number string to float, replacing 'd' with 'e'
	
	:param string: Number in string format.
	:type string: str
	"""
	if not isinstance(string, str): string = strD(string) # Convert to string if not string.
	return float(string.lower().replace('d','e'))
	
def strD(number):
	"""Converts input float to string, replacing 'e' with 'd'
	
	:param number: Number to be converted.
	:type number: str
	"""
	tstring = str(number)
	if 'e' in tstring:
		return ('%8.3e'%number).replace('e','d')
	else:
		return tstring
	
def pyflotran_print(s):
	if not pyflotran_print.silent: print s
#-----------------------------------------------------------------------------------------------------
#------------------------------ FUNCTIONS AND CLASSES FOR INTERNAL USE -------------------------------
#-----------------------------------------------------------------------------------------------------
	
class ppath(object):
	def __init__(self, filename='', work_dir='', parent=None):
		self._filename = filename
		self.absolute_to_file = ''	# location where originally read DOES NOT CHANGE
		self.absolute_to_workdir = ''	# working directory CAN CHANGE
		self.parent = parent
	def update(self, wd):
		'''called when work_dir is updated'''		
		if wd == '': 
			self.absolute_to_workdir = ''
			return
		
		if WINDOWS: wd = wd.replace('/','\\')
		else: wd = wd.replace('\\','/')
		
		absolute = False
		if WINDOWS and wd[1]==':': absolute = True
		if not WINDOWS and wd[0]=='/': absolute = True
		if absolute:
			self.absolute_to_workdir = wd
		else:
			self.absolute_to_workdir = os.getcwd()+slash+wd	
	def _get_filename(self): return self._filename
	def _set_filename(self,value): 
		# ensure path specification consistent with OS
		if WINDOWS: value = value.replace('/','\\')
		else: value = value.replace('\\','/')
		# check if any slashes exist
		if slash in value:
			self._filename = value.split(slash)[-1]
		else: 
			self._filename = value
			self.absolute_to_file = os.getcwd()
			return
		# check if absoulte or relative specification
		path = value.split(slash)[:-1]
		path = string.join(path,slash)
		
		absolute = False
		if WINDOWS and path[1]==':': absolute = True
		if not WINDOWS and path[0]=='/': absolute = True
		
		if absolute:
			self.absolute_to_file = path
		else:
			self.absolute_to_file = os.getcwd()+slash+path
	filename = property(_get_filename, _set_filename) #: (**)
	def _get_full_path(self): 
		return self.absolute_to_file+slash+self.filename
	full_path = property(_get_full_path) #: (**)
		
def os_path(path):
	if WINDOWS: path = path.replace('/','\\')
	else: path = path.replace('\\','/')
	return path
