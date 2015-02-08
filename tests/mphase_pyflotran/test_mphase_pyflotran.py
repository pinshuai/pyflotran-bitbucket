import filecmp
import unittest
import os

dir = os.path.dirname(os.path.realpath(__file__))

def compare_mphase():
    """Return True if pyflotran reads mphase.in correctly."""
    os.system('python ' + dir + '/mphase.py >& /dev/null')
    return  filecmp.cmp(dir + '/mphase.in', dir + '/mphase.gold')


class mphase_read(unittest.TestCase):
    """Test for writing to PFLOTRAN input from PyFLOTRAN input for mphase."""

    def test_mphase_read(self):
        """Test for writing to PFLOTRAN input from PyFLOTRAN input for mphase."""
        self.assertTrue(compare_mphase())

if __name__ == '__main__':
    unittest.main()
