# PyFLOTRAN setup script
from setuptools import setup

setup(name='pyflotran',
      version='1.0',
      description='Python scripting library for PFLOTRAN simulations',
      author='Satish Karra',
      author_email='satkarra@lanl.gov',
      py_modules=['ptool', 'pdata', 'pdflt'],
      install_requires=['numpy','matplotlib','pyvtk','pickle','h5py','json']
      )
