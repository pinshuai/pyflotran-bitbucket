import sys
import os
from pdata import *

try:
    pflotran_dir = os.environ['PFLOTRAN_DIR']
except KeyError:
    print('PFLOTRAN_DIR must point to PFLOTRAN installation directory \
           and be defined in system environment variables.')
    sys.exit(1)
sys.path.append(pflotran_dir + '/src/python')

try:
    pyflotran_dir = os.environ['PYFLOTRAN_DIR']
except KeyError:
    print('PYFLOTRAN_DIR must point to PYFLOTRAN installation directory \
           and be defined in system environment variables.')
    sys.exit(1)
sys.path.append(pyflotran_dir)

test_dir = '/tests/tracer_1D/'

# Read Test Data
dat = pdata(pyflotran_dir + test_dir + 'tracer_1D_SC.in')

# Test write
dat.write(pyflotran_dir + test_dir + 'tracer_1D_SC_2.in')
