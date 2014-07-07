import sys
sys.path.append('../../.')
from pdata import*


print '\nTEST EXECUTED\n'	# console header

###############################################################

# Read Test Data
dat = pdata('tracer_1D_SC.in')

# Test write
#dat.write('tracer_1D_SC_2.in')
	
# Writing to a different file and executing that file
dat.run(input='tracer_1D_SC_2.in',exe='/home/satkarra/src/pflotran-dev-PM-RHEL-6.5-nodebug/src/pflotran/pflotran')