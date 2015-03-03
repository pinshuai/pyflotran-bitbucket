import sys
import os
try:
  pflotran_dir = os.environ['PFLOTRAN_DIR']
except KeyError:
  print('PFLOTRAN_DIR must point to PFLOTRAN installation directory and be defined in system environment variables.')
  sys.exit(1)
sys.path.append(pflotran_dir + '/src/python')
import pflotran as pft

try:
  pyflotran_dir = os.environ['PYFLOTRAN_DIR']
except KeyError:
  print('PYFLOTRAN_DIR must point to PYFLOTRAN installation directory and be defined in system environment variables.')
  sys.exit(1)
sys.path.append(pyflotran_dir)
from pdata import*

test_dir = '/tests/mphase_multiple/'

# Read and execute input file
work_dir = pyflotran_dir + test_dir + 'mphase-run1/'
dat = pdata(filename=pyflotran_dir + test_dir + 'mphase_multi.in', work_dir=work_dir)
dat.co2_database = pflotran_dir + '/database/co2data0.dat'
dat.write(work_dir + 'mphase2.in')

# Write to a different workdir
work_dir = pyflotran_dir + test_dir + 'mphase-run2/'
dat = pdata(filename=pyflotran_dir + test_dir + 'mphase_multi.in', work_dir=work_dir)
dat.co2_database = pflotran_dir + '/database/co2data0.dat'
dat.write(work_dir + 'mphase2.in')
