import sys
import os

try:
    pflotran_dir = os.environ['PFLOTRAN_DIR']
except KeyError:
    print('PFLOTRAN_DIR must point to PFLOTRAN installation directory and be defined in system environment variables.')
    sys.exit(1)
sys.path.append(pflotran_dir + '/src/python')

try:
    pyflotran_dir = os.environ['PYFLOTRAN_DIR']
except KeyError:
    print(
        'PYFLOTRAN_DIR must point to PYFLOTRAN installation directory and be defined in system environment variables.')
    sys.exit(1)
sys.path.append(pyflotran_dir)
from pdata import *

test_dir = '/tests/vsat_2layer/'
###############################################################

# Read Test Data
dat = pdata(pyflotran_dir + test_dir + 'vsat_flow_pulse_2layer.in')

# Test write
dat.write(pyflotran_dir + test_dir + 'vsat_flow2.in')

# Writing to a different file and executing that file
# dat.run(input='vsat_flow2.in',exe=pflotran_dir + '/src/pflotran/pflotran')
