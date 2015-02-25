import filecmp
import unittest
import os

dir = os.path.dirname(os.path.realpath(__file__))

def compare_vsat_pulse():
    """Return True if pyflotran runs vsat_pulse_read.py in correctly."""
    os.system('python ' + dir + '/vsat_pulse_read.py >& /dev/null ')
    return  filecmp.cmp(dir + '/vsat_flow2.in', dir + '/vsat_flow.gold')

class vsat_pulse(unittest.TestCase):
    """Test for reading vsat_pulse."""

    def test_vsat_pulse_read(self):
        """Test for writing to PFLOTRAN input from PyFLOTRAN input for vsat 1D pulse"""
        self.assertTrue(compare_vsat_pulse())

if __name__ == '__main__':
    unittest.main()
