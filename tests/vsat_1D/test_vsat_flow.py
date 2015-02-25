import filecmp
import unittest
import os

dir = os.path.dirname(os.path.realpath(__file__))

def compare_vsat():
    """Return True if pyflotran runs vsat_read.py in correctly."""
    os.system('python ' + dir + '/vsat_read.py >& /dev/null ')
    return  filecmp.cmp(dir + '/vsat_flow2.in', dir + '/vsat_flow.gold')

class vsat_read(unittest.TestCase):
    """Test for reading vsat_1D."""

    def test_vsat_read(self):
        """Test for writing to PFLOTRAN input from PyFLOTRAN input for vsat 1D """
        self.assertTrue(compare_vsat())
	os.system('rm -f ' + dir + '/vsat_flow2.in')

if __name__ == '__main__':
    unittest.main()
