import sys
sys.path.append('../../.')
from pdata import*
import os
try:
  pflotran_dir = os.environ['PFLOTRAN_DIR']
except KeyError:
  print('PFLOTRAN_DIR must point to PFLOTRAN installation directory and be defined in system environment variables.')
  sys.exit(1)
sys.path.append(del_extra_slash(pflotran_dir + '/src/python'))
import pflotran as pft

# Read input file
dat = pdata('mphase.in')

# Test write
#dat.write('mphase2.in')

dat.co2_database = pflotran_dir + '/database/co2data0.dat'
pflotran_exe = del_extra_slash(pflotran_dir + '/src/pflotran/pflotran')
# Write to file and execute that input file
dat.run(input='mphase2.in',exe=pflotran_exe)
