from pdata import *
import subprocess
import shlex

try:
    pflotran_dir = os.environ['PFLOTRAN_DIR']
except KeyError:
    print('PFLOTRAN_DIR must point to PFLOTRAN installation' +
          'directory and be defined in system environment variables.')
    sys.exit(1)


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
    run_popen(pflotran_dir + '/src/pflotran/pflotran -pflotranin ' + file1)
    run_popen(pflotran_dir + '/src/pflotran/pflotran -pflotranin ' + file2)
    reg_file1 = file1.strip('.in') + '.regression'
    reg_file2 = file2.strip('.in') + '.regression'
    out, err = run_popen('diff ' + reg_file1 + ' ' + reg_file2)
    print out, err

success_files = []
read_fail_files = []
write_fail_files = []


def read_and_write(file):
    try:
        dat = pdata(file)
        print ('Successfully read file: ' + file)
        try:
            new_file = file.replace('/', ' ').split()[-1]
            dat.write(new_file)
            print ('  Successfully wrote file: ' + file)
            success_files.append(file)
        except:
            print ('  Error in writing: ' + file)
            write_fail_files.append(file)
    except:
        print('Error in reading: ' + file)
        read_fail_files.append(file)


files = check_file('find ' + pflotran_dir +
                  '/regression_tests/ -type f -name "*.in"')


for file in files:
    if file != '':
        # print file
        read_and_write(file)

print 'Successful files:'
print success_files
print 'Read fail files:'
print read_fail_files
print 'Write fail files:'
print write_fail_files

# read_with_error('/Users/satkarra/src/pflotran-dev-git/regression_tests/default//543/543_flow.in')

file = '/Users/satkarra/src/pflotran-dev-git/regression_tests//ascem/1d/1d-calcite/1d-calcite-np2.in'
read_and_write_with_error(file)
#file_new = file.replace('/', ' ').split()[-1]
#compare_regression_tests(file, file_new)



