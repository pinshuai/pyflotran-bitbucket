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

# Read and execute input file
dat = pdata(filename='mphase.in', work_dir='./mphase-run1')
dat.co2_database = pflotran_dir + '/database/co2data0.dat'
dat.run(input='mphase.in',exe=pflotran_dir + '/src/pflotran/pflotran')

# Write to a different workdir
dat = pdata(filename='mphase.in', work_dir='./mphase-run2')
dat.co2_database = pflotran_dir + '/database/co2data0.dat'
dat.run(input='mphase.in',exe=pflotran_dir + '/src/pflotran/pflotran')

