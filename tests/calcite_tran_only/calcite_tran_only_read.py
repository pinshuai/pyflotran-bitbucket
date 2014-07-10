import sys
sys.path.append('../../.')
from pdata import*
import os
try:
  pflotran_dir = os.environ['PFLOTRAN_DIR']
except KeyError:
  print('PFLOTRAN_DIR must point to PFLOTRAN installation directory and be defined in system environment variables.')
  sys.exit(1)
sys.path.append(pflotran_dir + '/src/python')
import pflotran as pft

pflotran_exe = pflotran_dir + '/src/pflotran/pflotran'

# Read input file
dat = pdata('calcite_tran_only.in')

# Testing write
#dat.write('calcite_tran_only_2.in')

# Write to file and execute that input file
dat.run(input='calcite_tran_only_2.in',exe=pflotran_exe)
