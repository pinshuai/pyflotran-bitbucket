from pdata import *
import subprocess
import shlex
import os
import filecmp
import pickle
import json

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

def regression_diff(file1, file2, verbose=False, json_diff_dir=None):
    '''
    Performs regression tests with native input file vs. PyFLOTRAN input file.
    Constructs a dictionary from the regression output file, and uses
    json-diff to compare the two.

    This function requires the json-diff library.

    To install, do:

       npm install json-diff

    :param file1: first file to test against
    :type file1: str
    :param file2: second file to test against
    :param file2: str
    :param verbose: prints status and output
    :type verbose: bool
    :param json_diff_dir: filepath to json-diff (default: pyflotran/)
    :type json_diff_dir: str

    Returns:

    :param success: true/false value indicating validation
    :type success: bool
    '''

    # Autocreate path if None.
    if json_diff_dir is None:
        json_diff_dir = os.path.join(os.getcwd(),'node_modules/json-diff/bin/json-diff.js')
    
    current_dir = os.getcwd()
    capture = '--> write regression output file:'
    regression_out = []

    # First, we are going to capture the regression output
    for file in [file1,file2]:
        # Change directory to first file
        os.chdir(os.path.dirname(file))

        out, err = run_popen(
            pflotran_dir + '/src/pflotran/pflotran -pflotranin ' + file)

        if 'ERROR' in out:
            print('PFLOTRAN ERROR: Fail on read ' + file)
            os.chdir(current_dir)
            return False

        # Capture the regression file written out
        capture_idx = out.find(capture)
        if capture_idx == -1:
            os.chdir(current_dir)
            return False

        reg_file = out[capture_idx+len(capture)+1:out.find('\n',capture_idx)]
        regression_out.append(open(reg_file).read())

    # Now, we want to store the regression output as a dictionary
    a = dict()

    for (i,regression_file) in enumerate(regression_out):
        b = dict()
        for line in regression_file.split('\n'):
            # Skip empty lines
            if line.strip() == '': continue

            # Capture headers as new dicts
            if line[:2] == '--':
                key = line.strip('--').strip()
                b[key] = dict()
            else:
                # Split lines in headers as [key,value]
                parsed = list(map(str.strip,line.strip().split(':')))
                if parsed[0] != 'Time (seconds)':
                    try:
                        b[key][parsed[0]] = round(float(parsed[1]),3)
                    except ValueError:
                        b[key][parsed[0]] = [round(float(p),3) for p in parsed[1].split()]
        a[i] = b

    os.chdir(current_dir)

    # Write out JSON files so they can be compared
    with open('tmp_dump1.json','w') as f:
        f.write(json.dumps(a[0], sort_keys=True, indent=2))

    with open('tmp_dump2.json','w') as f:
        f.write(json.dumps(a[1], sort_keys=True, indent=2))

    # Compare JSON files
    out, err = run_popen(json_diff_dir+' tmp_dump1.json tmp_dump2.json')

    if verbose:
        print(' | '.join(list(map(os.path.basename,[file1,file2]))) + \
            '\n'.join([out,err,'-'*50]))

    if out == '' and err == '':
        return True
    else:
        return False

def regression_validation(file_list,tmp_out="temp.in",verbose=False,json_diff_dir=None):
    '''
    Performs batch regression tests with native input files vs.
    PyFLOTRAN-created input files.

    This function requires the json-diff library.

    To install, do:

       npm install json-diff

    :param file_list: list of PFLOTRAN input files to test
    :type file_list: list(str)
    :param tmp_out: file handle to write PyFLOTRAN output to
    :param tmp_out: str
    :param verbose: prints status and output
    :type verbose: bool
    :param json_diff_dir: filepath to json-diff (default: cwd()/)
    :type json_diff_dir: str
    '''

    current_dir = os.getcwd()

    # Autocreate path if None.
    if json_diff_dir is None:
        json_diff_dir = os.path.join(current_dir,'node_modules/json-diff/bin/json-diff.js')

    # Check that json-diff exists.
    if not os.path.isfile(json_diff_dir):
        print('ERROR: json-diff.js not found.\n'+
            'To install, run\n\n    npm install json-diff\n')
        sys.exit()

    fail_list = []
    success_list = []

    # Iterate over each file...
    for file in file_list:
        # Try reading
        try:
            dat = pdata(file)
            tmp_file = os.path.join(os.path.dirname(file),tmp_out)
            dat.write(tmp_file)
        except:
            print('PyFLOTRAN ERROR: Could not parse \'%s\'' % file)
            fail_list.append(file)
            continue

        os.chdir(current_dir)
        status = regression_diff(file,tmp_file,verbose=verbose,json_diff_dir=json_diff_dir)

        if status:
            success_list.append(file)
        else:
            fail_list.append(file)

        os.remove(tmp_file)

    print("Number of successful regressions:   %d" % len(success_list))
    print("Number of failed regressions:       %d" % len(fail_list))

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

# Run regression validation
regression_validation(files,verbose=True)

# Compare regression outputs

file = '/Users/satkarra/src/pflotran-dev-git/regression_tests//default/543/543_flow_dbase.in'
dir_name = os.path.dirname(file)
read_and_write_with_error(file)
new_file = dir_name + '/' + file.replace('/', ' ').split()[-1]
new_file = new_file.replace('.', ' ').split()[0] + '_pyflotran.in'
# compare_regression_tests(file, new_file)

# for file in success_files:
#     dir_name = os.path.dirname(file)
#     new_file = dir_name + '/' + file.replace('/', ' ').split()[-1]
#     new_file = new_file.replace('.', ' ').split()[0] + '_pyflotran.in'
#     compare_regression_tests(file, new_file)
