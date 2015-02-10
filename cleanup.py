import os
import glob

for file in glob.glob('tests/*/*tec'):
	os.system('rm -f' + ' ' + file)	

for file in glob.glob('tests/*/*out'):
	os.system('rm -f' + ' ' + file)	

for file in glob.glob('tests/*/*h5'):
	os.system('rm -f' + ' ' + file)	
	
