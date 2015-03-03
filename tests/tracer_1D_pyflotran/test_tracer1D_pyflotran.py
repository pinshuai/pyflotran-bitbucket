import filecmp
import unittest
import os

dir = os.path.dirname(os.path.realpath(__file__))

def compare_tracer1D():
    """Return True if pyflotran runs tracer_1D_pyflotran.py in correctly."""
    os.system('python ' + dir + '/tracer_1D_pyflotran.py >& /dev/null ')
    return  filecmp.cmp('tracer_1D.in', dir + '/tracer_1D.gold')

class tracer1D_read(unittest.TestCase):
    """Test for reading tracer1D."""

    def test_tracer1D_read(self):
        """Test for writing to PFLOTRAN input from PyFLOTRAN input for tracer 1D"""
        self.assertTrue(compare_tracer1D())
	os.system('rm -f ' + './tracer_1D.in')
	os.system('rm -f ' + './tracer_1D.out')
	os.system('rm -f ' + './tracer_1D*.tec')
	os.system('rm -f ' + './tracer_1D*.regression')


if __name__ == '__main__':
    unittest.main()
