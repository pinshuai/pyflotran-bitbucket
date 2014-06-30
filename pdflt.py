


class pdflt(object):
	def __init__(self):
		# set this to the pflotran executable to be used if no default assigned
		self.pflotran_path = '/usr/bin/pflotran'
		
	def _get_pflotran_path(self): return self._pflotran_path
	def _set_pflotran_path(self, object): self._pflotran_path = object
	pflotran_path = property(_get_pflotran_path, _set_pflotran_path) #: (**)	