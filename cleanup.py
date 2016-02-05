import os
import glob

for FILE in glob.glob('tests/*/*tec'):
    os.remove(FILE)

for FILE in glob.glob('tests/*/*out'):
    os.remove(FILE)

for FILE in glob.glob('tests/*/*.h5'):
    os.remove(FILE)

for FILE in glob.glob('tests/*/*.regression'):
    os.remove(FILE)
    
for FILE in glob.glob('tests/*/multi_*/*'):
    os.remove(FILE)

for FILE in glob.glob('tests/*/*_new.*'):
    os.remove(FILE)
