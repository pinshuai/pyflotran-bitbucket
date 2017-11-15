import pickle

file = 'pflotran-tests-2017-11-15_13-12-07.testlog'
write_file = 'regression_input_filenames.pickle'

count = 0
tests = []
print '--> Reading'
with open(file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        #        print line
        if '... ' in line:
            if len(line.split()) == 1:
                tests.append(line.replace('.', ' ').split()[0] + '.in')
                count += 1

print '--> Tests are:'
print tests
print '--> Total tests: ' + str(count)
print '--> Writing test input files'

with open(write_file, 'wb') as f:
    pickle.dump(tests, f)
