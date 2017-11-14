from pdata import *
import subprocess
import shlex

try:
    pflotran_dir = os.environ['PFLOTRAN_DIR']
except KeyError:
    print('PFLOTRAN_DIR must point to PFLOTRAN installation' +
          'directory and be defined in system environment variables.')
    sys.exit(1)

def run_popen(cmd):
    process = subprocess.Popen(
        shlex.split(cmd), stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    out, err = process.communicate()
    files = out.split('\n')
    return files


def read(file):
    try:
        dat = pdata(file)
    except:
        print('Error in reading ' + file)
        pass


def read_and_write(file):
    success_files = []
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
    except:
        print('Error in reading: ' + file)
        pass
    return success_files


files = run_popen('find ' + pflotran_dir +
                  '/regression_tests/ -type f -name "*.in"')

for file in files:
    if file != '':
        # print file
        success_files = read_and_write(file)

