from pdata import *
import subprocess
import shlex
import os
import filecmp
import pickle

try:
    pflotran_dir = os.environ['PFLOTRAN_DIR']
except KeyError:
    print('PFLOTRAN_DIR must point to PFLOTRAN installation' +
          'directory and be defined in system environment variables.')
    sys.exit(1)

regression_tests_filename = 'regression_input_filenames.pickle'

with open(regression_tests_filename, 'rb') as f:
    tests_list = pickle.load(f)

success_files = []
read_fail_files = []
write_fail_files = []
regression_pass_files = []
regression_fail_files = []


def check_file(cmd):
    process = subprocess.Popen(
        shlex.split(cmd), stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    out, err = process.communicate()
    files = out.split('\n')
    return files


def run_popen(cmd):
    process = subprocess.Popen(
        shlex.split(cmd), stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    out, err = process.communicate()
    return out, err


def read(file):
    try:
        dat = pdata(file)
    except:
        print('Error in reading ' + file)


def read_with_error(file):
    dat = pdata(file)


def read_and_write_with_error(file):
    dat = pdata(file)
    new_file = file.replace('/', ' ').split()[-1]
    dat.write(new_file)


def compare_regression_tests(file1, file2):
    current_dir = os.getcwd()
    os.chdir(os.path.dirname(file1))
    out, err = run_popen(
        pflotran_dir + '/src/pflotran/pflotran -pflotranin ' + file1)
    # print out, err
    if 'ERROR' in out:
        print 'Did not run: ' + file1
    os.chdir(os.path.dirname(file2))
    out, err = run_popen(
        pflotran_dir + '/src/pflotran/pflotran -pflotranin ' + file2)
    # print out, err
    if 'ERROR' in out:
        print 'Did not run: ' + file2
    else:
        reg_file1 = file1.replace('.', ' ').split()[0] + '.regression'
        reg_file2 = file2.replace('.', ' ').split()[0] + '.regression'
        #out, err = run_popen('diff ' + reg_file1 + ' ' + reg_file2)
        # print out, err
        if not filecmp.cmp(reg_file1, reg_file2):
            print 'Difference seen in regression files' + reg_file1 + ' and ' + reg_file2
            regression_fail_files.append(file1)
        else:
            regression_pass_files.append(file1)
    os.chdir(current_dir)


def read_and_write(file):
    try:
        dat = pdata(file)
        dir_name = os.path.dirname(file)
        print ('Successfully read file: ' + file)
        try:
            new_file = dir_name + '/' + file.replace('/', ' ').split()[-1]
            new_file = new_file.replace('.', ' ').split()[0] + '_pyflotran.in'
            print new_file
            dat.write(new_file)
            print ('  Successfully wrote file: ' + file)
            success_files.append(file)
        except:
            print ('  Error in writing: ' + file)
            write_fail_files.append(file)
    except:
        print('Error in reading: ' + file)
        read_fail_files.append(file)


def cleanup():
    files = check_file('find ' + pflotran_dir +
                       '/regression_tests/ -type f -name "*pyflotran.in"')
    for file in files:
        print file
        run_popen('rm -f ' + file)


cleanup()

files = check_file('find ' + pflotran_dir +
                   '/regression_tests/ -type f -name "*.in"')


for file in files:
    if file != '':
        # print file
        if file.replace('/', ' ').split()[-1] in tests_list:
            read_and_write(file)

# print results
print 'Successful files:'
print success_files
print 'Read fail files:'
print read_fail_files
print 'Write fail files:'
print write_fail_files

print 'Number of successful files: ' + str(len(success_files))
print 'Number of read fail files: ' + str(len(read_fail_files))
print 'Number of write fail files: ' + str(len(write_fail_files))

# read_with_error('/Users/satkarra/src/pflotran-dev-git/regression_tests/default//543/543_flow.in')

# Compare regression outputs

# file = '/Users/satkarra/src/pflotran-dev-git/regression_tests//ascem/1d/1d-calcite/1d-calcite.in'
# dir_name = os.path.dirname(file)
# read_and_write_with_error(file)
# new_file = dir_name + '/' + file.replace('/', ' ').split()[-1]
# new_file = new_file.replace('.', ' ').split()[0] + '_pyflotran.in'
# compare_regression_tests(file, new_file)


# for file in success_files:
#     dir_name = os.path.dirname(file)
#     new_file = dir_name + '/' + file.replace('/', ' ').split()[-1]
#     new_file = new_file.replace('.', ' ').split()[0] + '_pyflotran.in'
#     compare_regression_tests(file, new_file)
