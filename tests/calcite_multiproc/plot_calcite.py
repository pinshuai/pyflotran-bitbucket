# ------------------------------------------------------------------------------
# Plot the 1D horizontal data for multiple 1D calcite problem with PyFLOTRAN.
# Satish Karra
# Sept. 15, 2014
# ------------------------------------------------------------------------------

import sys
import os

try:
    pyflotran_dir = os.environ['PYFLOTRAN_DIR']
except KeyError:
    print('PFLOTRAN_DIR must point to PFLOTRAN installation directory and be defined in system environment variables.')
    sys.exit(1)
sys.path.append(pyflotran_dir)
from pdata import *

try:
    pflotran_dir = os.environ['PFLOTRAN_DIR']
except KeyError:
    print('PFLOTRAN_DIR must point to PFLOTRAN installation directory and be defined in system environment variables.')
    sys.exit(1)
sys.path.append(pflotran_dir + '/src/python')

start_file = 5
end_file = 6
num_realizations = 4

files = []
legend_list = []
for i in range(1, num_realizations + 1):
    work_dir = 'multi_' + str(i)
    legend = 'case ' + str(i)
    for j in range(start_file, end_file):
        files.append(work_dir + '/' + work_dir + '_calcite-00' + str(j) + '.tec')
        legend_list.append(legend)

dat = pdata('')
dat.plot_data_from_tec(variable_list=['Calcite VF', 'Calcite Rate', 'pH'], tec_filenames=files,
                       plot_filename='combined_calcite_vf.pdf', legend_list=legend_list, xlabel='x [m]',
                       ylabel_list=['Calcite VF', 'Calcite Rate', 'pH'])
