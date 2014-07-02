# PyFLOTRAN setup script
from distutils.core import setup

setup(name='PyFLOTRAN',
	version='1.0.0',
	description='Python scripting library for PFLOTRAN simulations',
	author='Satish Karra, David Dempsey, Cory Kitay',
	author_email='satkarra@lanl.gov, d.dempsey@lanl.gov',
#	url='pyfehm.lanl.gov',
#	license='LGPL',
	py_modules=['ptool','pdata','pdflt'],
	packages=['pyvtk'],
#	scripts = ['scripts/diagnose.py','scripts/fehm_paraview.py']
	)
