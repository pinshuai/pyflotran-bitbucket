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

pflotran_exe = pflotran_dir + '/src/pflotran/pflotran'

# Read input file
dat = pdata('calcite_tran_only.in')

# Testing write
#dat.write('calcite_tran_only_2.in')

# Write to file and execute that input file
dat.chemistry.database = pflotran_dir + '/database/hanford.dat'
dat.run(input='calcite_tran_only_2.in',exe=pflotran_exe)
