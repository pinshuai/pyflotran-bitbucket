import filecmp
import unittest
import os

dir = os.path.dirname(os.path.realpath(__file__))

def compare_mphase():
    """Return True if pyflotran reads mphase.in correctly."""
   
    os.system('ls ' + dir) 
    os.system('python ' + dir + '/mphase_read.py >& /dev/null')
    return  filecmp.cmp(dir + '/mphase2.in', dir + '/mphase.gold')

class mphase_read(unittest.TestCase):
    """Test for reading mphase."""

    def test_mphase_read(self):
        """Test for reading mphase"""
        self.assertTrue(compare_mphase())
	os.system('rm -f ' + dir + '/mphase2.in')
	os.system('rm -f ' + dir + '/mphase2.out')
	os.system('rm -f ' + dir + '/mphase2*.tec')
	os.system('rm -f ' + dir + '/mphase2*.h5')
	os.system('rm -f ' + dir + '/mphase2*.dat')
	os.system('rm -f ' + dir + '/mphase2*.regression')

if __name__ == '__main__':
    unittest.main()
