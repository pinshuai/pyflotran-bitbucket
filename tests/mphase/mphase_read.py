import sys
sys.path.append('../../.')
from pdata import*

# Read input file
dat = pdata('mphase.in')

# Test write
#dat.write('mphase2.in')

# Write to file and execute that input file
dat.run(input='mphase2.in',exe='/home/satkarra/src/pflotran-dev-PM-RHEL-6.5-nodebug/src/pflotran/pflotran')
