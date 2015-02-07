import filecmp
import unittest
import os

try:
  pyflotran_dir = os.environ['PYFLOTRAN_DIR']
except KeyError:
  print('PYFLOTRAN_DIR must point to PFLOTRAN installation directory and be defined in system environment variables.')
  sys.exit(1)

test_dir = '/tests/mphase/'

def compare_mphase():
    """Return True if pyflotran reads mphase.in correctly."""
    os.system('python ' + pyflotran_dir + test_dir + 'mphase_read.py >& /dev/null')
    filecmp.cmp(pyflotran_dir + test_dir + 'mphase2.in', pyflotran_dir + test_dir + 'mphase.gold')

    return True

class mphase_read(unittest.TestCase):
    """Test for reading mphase."""

    def test_mphase_read(self):
        """Test for reading mphase"""
        self.assertTrue(compare_mphase())

if __name__ == '__main__':
    unittest.main()
