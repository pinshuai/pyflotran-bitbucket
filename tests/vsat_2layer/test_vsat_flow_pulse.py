import filecmp
import unittest
import os

dir = os.path.dirname(os.path.realpath(__file__))

def compare_vsat_2layer():
    """Return True if pyflotran runs vsat_2layer_read.py in correctly."""
    os.system('python ' + dir + '/vsat_2layer_read.py >& /dev/null ')
    return  filecmp.cmp(dir + '/vsat_flow2.in', dir + '/vsat_flow.gold')

class vsat_2layer(unittest.TestCase):
    """Test for reading vsat_2layer."""

    def test_vsat_2layer_read(self):
        """Test for writing to PFLOTRAN input from PyFLOTRAN input for vsat 1D 2layer"""
        self.assertTrue(compare_vsat_2layer())
	os.system('rm -f ' + dir + '/vsat_flow2.in')

if __name__ == '__main__':
    unittest.main()
