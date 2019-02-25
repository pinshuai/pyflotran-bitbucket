from pdata import *
from ptool import decode
import subprocess
import shlex
import os
import filecmp
import pickle
import json
import argparse
from glob import glob

PYFLOTRAN_RF_ERROR   = 15 # Read fail error
PYFLOTRAN_WF_ERROR   = 16 # Write fail error
PYFLOTRAN_PFLO_ERROR = 12 # PFLOTRAN runtime error
PYFLOTRAN_COMP_ERROR = 13 # Regression comparison error

try:
    pflotran_dir = os.environ['PFLOTRAN_DIR']
except KeyError:
    print('PFLOTRAN_DIR must point to PFLOTRAN installation' +
          'directory and be defined in system environment variables.')
    sys.exit(1)

# Files to set the EXTERNAL_FLAG to False
table_files = ['tl_omega_1_cc_tables.in',
               'tl4pr1_cc_tables_np2.in',
               'TOWG_RCOL8_cc_table.in',
               'bt4_cc_tables.in',
               '1d_th_water_flood_ad_cc_tables.in']

regression_tests_filename = 'regression_input_filenames.pickle'

try:
    pflotran_testlog = glob(pflotran_dir + '/regression_tests/pflotran-tests-*')[-1]
except IndexError:
    pflotran_testlog = ''

with open(regression_tests_filename, 'rb') as f:
    tests_list = pickle.load(f)

success_files = []
read_fail_files = []
write_fail_files = []
regression_pass_files = []
regression_fail_files = []

def check_file(cmd):
    process = subprocess.Popen(shlex.split(cmd),
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)

    out, err = process.communicate()
    return decode(out).split('\n')

def get_test_command(fname):
    '''
    Gets shell command to use on a test case, parsed from
    regression_tests testlog.
    '''

    with open(fname,'rb') as f:
        file = f.read().decode(errors='replace').split('\n')

    cmds = {}
    current_key = None

    for line in file:
        if '...' in line and not 'passed' in line:
            current_key = line.split('...')[0]
            cmds[current_key] = ''
        else:
            if current_key is not None and '-' in line and 'src' in line:
                line = line.strip()

                # Strip -input_prefix from command
                if '-input_prefix' in line:
                    _pre = line.split()
                    _idx = _pre.index('-input_prefix')
                    line = ' '.join(_pre[:_idx] + _pre[_idx+2:])
                cmds[current_key] = line

    return cmds


def run_popen(cmd):
    process = subprocess.Popen(shlex.split(cmd),
                               stdout=subprocess.PIPE,
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

def check_keywords(f1,f2):
    from ptool import floatD

    with open(f1,'r') as f:
        f1 = f.read().lower().split('\n')

    with open(f2,'r') as f:
        f2 = f.read().lower().split('\n')

    a = []
    ln = []
    for (i,line) in enumerate(f1):
        line = line.strip()
        if len(line) > 1:
            if line[0] != "#" and line[0] != "!":
                a.append(line.split()[0])
                ln.append(i+1)
    f1 = a

    a = []
    for line in f2:
        line = line.strip()
        if len(line) > 1:
            if line[0] != "#" and line[0] != "!":
               a.append(line.split()[0])

    f2 = a

    skip = False
    for (i,line1) in enumerate(f1):
        if skip == True:
            if line1 == 'noskip':
                skip = False
        elif len(line1) <= 1:
            continue
        elif line1[0] == "#":
            continue
        else:

            if line1 == 'skip':
                skip = True
            elif line1 not in f2 and not ('.' in line1 and 'd' in line1):
                print("%d : %s" % (ln[i],line1))

def compare_regression_tests(file1, file2):
    current_dir = os.getcwd()
    os.chdir(os.path.dirname(file1))
    out, err = run_popen(
        pflotran_dir + '/src/pflotran/pflotran -pflotranin ' + file1)

    out = decode(out)
    err = decode(err)

    # print out, err
    if 'ERROR' in out:
        print('Did not run: ' + file1)
    os.chdir(os.path.dirname(file2))
    out, err = run_popen(
        pflotran_dir + '/src/pflotran/pflotran -pflotranin ' + file2)
    # print out, err
    if 'ERROR' in out:
        print('Did not run: ' + file2)
    else:
        reg_file1 = file1.replace('.', ' ').split()[0] + '.regression'
        reg_file2 = file2.replace('.', ' ').split()[0] + '.regression'
        #out, err = run_popen('diff ' + reg_file1 + ' ' + reg_file2)
        # print out, err
        if not filecmp.cmp(reg_file1, reg_file2):
            print('Difference seen in regression files' + reg_file1 + ' and ' + reg_file2)
            regression_fail_files.append(file1)
        else:
            regression_pass_files.append(file1)
    os.chdir(current_dir)


def read_and_write(file):
    try:
        dat = pdata(file)
        dir_name = os.path.dirname(file)
        print('Successfully read file: ' + file)
        try:
            new_file = dir_name + '/' + file.replace('/', ' ').split()[-1]
            new_file = new_file.replace('.', ' ').split()[0] + '_pyflotran.in'
            print(new_file)
            dat.write(new_file)
            print('  Successfully wrote file: ' + file)
            success_files.append(file)
        except:
            print('  Error in writing: ' + file)
            write_fail_files.append(file)
    except:
        print('Error in reading: ' + file)
        read_fail_files.append(file)

def regression_diff(file1, file2, verbose=False, json_diff_dir=None, debug_dict=None, check_integrity=False, cmd=None):
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
    for (i,file) in enumerate([file1,file2]):
        # Change directory to first file
        os.chdir(os.path.dirname(file))

        if cmd is None or cmd.strip() == '':
            out, err = run_popen(pflotran_dir + '/src/pflotran/pflotran -pflotranin ' + file)
        else:
            out, err = run_popen('%s -pflotranin %s' % (cmd,file))

        try:
            out = decode(out)
            err = decode(err)
        except UnicodeDecodeError:
            print('Could not decode')
            return False

        if 'ERROR' in out:

            print('\033[91mPFLOTRAN RUNTIME ERROR: \033[0m'+
                out[out.find('ERROR'):out.find('\n',out.find('ERROR'))]
                +'\n'+'-'*50)

            os.chdir(current_dir)
            if debug_dict is not None:
                debug_dict['failed']['pflotran_runtime'][file] = out[out.find('ERROR'):out.find('\n',out.find('ERROR'))]

            if check_integrity and i == 0:
                print('\033[93mWARNING: Corrupted original regression test file\033[0m\n'+'-'*50)
                return True
            return False

        elif 'ERROR' in err:
            print('\033[91mPFLOTRAN RUNTIME ERROR: \033[0m\n'+err+'\n'+'-'*50)
            os.chdir(current_dir)
            if debug_dict is not None:
                debug_dict['failed']['pflotran_runtime'][file] = err

            if check_integrity and i == 0:
                print('\033[93mWARNING: Corrupted original regression test file\033[0m\n'+'-'*50)
                return True
            return False

        # Capture the regression file written out
        capture_idx = out.find(capture)

        if capture_idx == -1:
            os.chdir(current_dir)
            if debug_dict is not None:
                debug_dict['failed']['pflotran_runtime'][file] = out
            print('\033[93mWARNING: Can\'t find PFLOTRAN regression file output\033[0m\n'+'-'*50)

            if verbose: print(err)

            if check_integrity and i == 0:
                print('\033[93mWARNING: Corrupted original regression test file\033[0m\n'+'-'*50)
                return True
            return False

            #return False
            return True

        reg_file = out[capture_idx+len(capture)+1:out.find('\n',capture_idx)]
        regression_out.append(open(reg_file).read())

    # Now, we want to store the regression output as a dictionary
    a = dict()

    # Regession output to ignore in comparison
    _ignore_keys = ['Time (seconds)','Newton Iterations',
                    'Solver Iterations','Time Steps']

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
                if parsed[0] not in _ignore_keys:
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

    out = decode(out)
    err = decode(err)

    # Remove JSON files
    os.remove('tmp_dump1.json')
    os.remove('tmp_dump2.json')

    if verbose:
        print(' | '.join(list(map(os.path.basename,[file1,file2]))) + \
            '\n'.join([out,err,'-'*50]))


    if out == '' and err == '':
        print('\033[92mPASSED\033[0m\n'+'-'*50)
        if debug_dict is not None:
                debug_dict['passed'].append(file1)
        return True
    else:
        print('\033[91mREGRESSION COMPARISON ERROR:\033[0m FAILED ON COMPARISON\n'+'-'*50)
        if debug_dict is not None:
                debug_dict['failed']['regression_comp'][file1] = out
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
    debug = False

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

    # Dictionary recording status of all files
    # Useful for debugging
    debug_dict = {
        'passed':[],
        'failed':{
            'pflotran_runtime':{},
            'pyflotran_runtime':{},
            'regression_comp':{}
        }
    }

    try:
        cmds = get_test_command(pflotran_testlog)
    except IOError:
        cmds = {}

    # Iterate over each file...
    for file in file_list:
        # Try reading
        try:
            print('\033[94m'+file+'\033[0m')
            tmp_file = os.path.join(os.path.dirname(file),tmp_out)

            _replace = True
            if file.split('/')[-1] in table_files:
                _replace = False

            pdata(file,replace_external_files=_replace).write(tmp_file)
            print('\033[94m'+tmp_file+'\033[0m')
        except Exception as e:
            print('\033[91mPyFLOTRAN RUNTIME ERROR:\033[0m Could not parse \'%s\'' % file)
            fail_list.append(file)
            print(e)
            debug_dict['failed']['pyflotran_runtime'][file] = e
            continue

        file_prefix = file.split('/')[-1].split('.')[0]
        if file_prefix in cmds.keys():
            cmd = cmds[file_prefix]

        os.chdir(current_dir)
        status = regression_diff(file,tmp_file,verbose=verbose,
                                 json_diff_dir=json_diff_dir,
                                 debug_dict=debug_dict,
                                 check_integrity=True,
                                 cmd=cmd)

        if status == False:
            pass
            #sys.exit(1)

        if status:
            success_list.append(file)
        else:
            fail_list.append(file)

        #os.remove(tmp_file)

    print("Number of successful regressions:   %d" % len(success_list))
    print("Number of failed regressions:       %d" % len(fail_list))

    # Write out complete pass/fail
    if debug:
        with open('DEBUG_LOG','w') as f:
            f.write("Passed:       %d" % len(success_list) + '\n' + 
                    "Failed:       %d" % len(fail_list) + '\n\n' + 
                    json.dumps(debug_dict))

    return debug_dict

def cleanup():
    files = check_file('find ' + pflotran_dir + \
                       '/regression_tests/ -type f -name "*pyflotran.in"')
    for file in files:
        print(file)
        run_popen('rm -f ' + file)

def validation(files):

    for file in files:
        if file != '':
            # print file
            if file.replace('/', ' ').split()[-1] in tests_list:
                read_and_write(file)

    # print results
    print('Successful files:')
    print(success_files)
    print('Read fail files:')
    print(read_fail_files)
    print('Write fail files:')
    print(write_fail_files)

    print('Number of successful files: ' + str(len(success_files)))
    print('Number of read fail files: ' + str(len(read_fail_files)))
    print('Number of write fail files: ' + str(len(write_fail_files)))


# Array of multiline literals for use in argparse descriptions.
arg_help = ['''
Performs a two-proged analysis on PyFLOTRAN files.
In the 'validation' approach, multiple PFLOTRAN input files are read into PyFLOTRAN and 
written out. This should be considered a 'first-pass' for verifying I/O, functional, and class
capabilities of PyFLOTRAN.

The second and more rigorous test is to perform a regression analysis on output from PFLOTRAN:
this is done by a comparitive analysis on PFLOTRAN output from a 'gold standard' input file and
from a PyFLOTRAN generated input file.''',
'''Run both validation and regression tests.''',
'''Run *only* the comparative regression analysis.''',
'''Run *only* the PyFLOTRAN I/O validation.''',
'''Run a comparative regression analysis on a single PFLOTRAN input file.''',
'''Compare read and written PFLOTRAN input files for missing keywords.''']

if (__name__ == '__main__'):

    cleanup()

    files_all = check_file('find ' + pflotran_dir + \
                       '/regression_tests/ -type f -name "*.in"')

    files = []
    for file in files_all:
        if file != '':
            # print file
            if file.replace('/', ' ').split()[-1] in tests_list:
                files.append(file)


    # Parse arguments
    parser = argparse.ArgumentParser(description=arg_help[0])
    parser.add_argument('-a','--all', action='store_true', help=arg_help[1])
    parser.add_argument('-r','--regression', action='store_true', help=arg_help[2])
    parser.add_argument('-v','--validation', action='store_true', help=arg_help[3])
    parser.add_argument('-s','--single', type=str, help=arg_help[4])
    parser.add_argument('-d','--diff',type=str, help=arg_help[5])

    #debug - delete
    parser.add_argument('-rw','--readwrite',type=str)

    args = parser.parse_args()

    if args.all:
        validation(files)
        regression_validation(files,verbose=False)
    elif args.regression:
        regression_validation(files,verbose=False)
    elif args.validation:
        validation(files)
    elif args.single is not None:
        regression_validation([args.single],verbose=True)
    elif args.diff is not None:
        tmp_file = os.path.join(os.path.dirname(args.diff),'temp.in')
        pdata(args.diff).write(tmp_file)
        print('Missing keywords:')
        check_keywords(args.diff,tmp_file)
    elif args.readwrite is not None:
        tmp_file = os.path.join(os.path.dirname(args.readwrite),'temp.in')
        pdata(args.readwrite).write(tmp_file)
    else:
        parser.print_help()

