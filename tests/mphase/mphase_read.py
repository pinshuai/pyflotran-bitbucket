import sys
import os

try:
  pflotran_dir = os.environ['PFLOTRAN_DIR']
except KeyError:
  print('PFLOTRAN_DIR must point to PFLOTRAN installation directory and be defined in system environment variables.')
  sys.exit(1)

try:
  pyflotran_dir = os.environ['PYFLOTRAN_DIR']
except KeyError:
  print('PYFLOTRAN_DIR must point to PYFLOTRAN installation directory and be defined in system environment variables.')
  sys.exit(1)
sys.path.append(pyflotran_dir)

from pdata import*

test_dir = '/tests/mphase/'

# Read input file
dat = pdata(pyflotran_dir + test_dir + 'mphase.in')

# Test write
#dat.write('mphase2.in')

dat.co2_database = pflotran_dir + '/database/co2data0.dat'
pflotran_exe = del_extra_slash(pflotran_dir + '/src/pflotran/pflotran')
# Write to file and execute that input file
dat.run(input=pyflotran_dir + test_dir + 'mphase2.in',exe=pflotran_exe)
