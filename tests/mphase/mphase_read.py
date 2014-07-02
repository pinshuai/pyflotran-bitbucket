import sys
sys.path.append('../../.')
from pdata import*

# Read input file
dat = pdata('mphase.in')


# Write to a different input file 
dat.write('mphase2.in')
