import sys
sys.path.append('../../.')
from pdata import*

# Read input file
dat = pdata('calcite_tran_only.in')

# Testing write
#dat.write('calcite_tran_only_2.in')

# Write to file and execute that input file
dat.run(input='calcite_tran_only_2.in',exe='/home/satkarra/src/pflotran-dev-PM-RHEL-6.5-nodebug/src/pflotran/pflotran')