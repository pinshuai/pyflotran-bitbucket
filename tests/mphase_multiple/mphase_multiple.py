import sys
sys.path.append('../../.')
from pdata import*

# Read and execute input file
dat = pdata(filename='mphase.in', work_dir='./mphase-run1')
dat.run(input='mphase2.in',exe='/home/satkarra/src/pflotran-dev-PM-RHEL-6.5-nodebug/src/pflotran/pflotran')

# Write to a different workdir
dat = pdata(filename='mphase.in', work_dir='./mphase-run2')
dat.run(input='mphase2.in',exe='/home/satkarra/src/pflotran-dev-PM-RHEL-6.5-nodebug/src/pflotran/pflotran')

