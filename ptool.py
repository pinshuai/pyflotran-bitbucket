"""Tools for use with pflotran python"""

import numpy as np

def floatD(string):
	"""Converts input number string to float, replacing 'd' with 'e'
	
	:param string: Number in string format.
	:type string: str
	"""
	return float(string.lower().replace('d','e'))
	
def strD(number):
	"""Converts input float to string, replacing 'e' with 'd'
	
	:param number: Number to be converted.
	:type number: str
	"""
	tstring = str(number)
	if 'e' in tstring:
		return ('%8.7e'%number).replace('e','d')
	else:
		return tstring