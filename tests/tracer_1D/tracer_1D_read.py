import sys
from pdata import*

sys.path.append('../../.')

print '\nTEST EXECUTED\n'	# console header

###############################################################

# Read Test Data
dat = pdata('tracer_1D_SC.in')
	
# Writing to a different file and executing that file

dat.run(input='tracer_1D_SC_2.in',exe='/home/satkarra/src/pflotran-dev-PM-RHEL-6.5-nodebug/src/pflotran/pflotran')
