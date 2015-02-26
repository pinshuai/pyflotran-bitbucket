import filecmp
import unittest
import os

dir = os.path.dirname(os.path.realpath(__file__))

def compare_tracer1D():
    """Return True if pyflotran reads tracer_1D.in correctly."""
    os.system('python ' + dir + '/tracer_1D_read.py >& /dev/null')
    return  filecmp.cmp(dir + '/tracer_1D_SC_2.in', dir + '/tracer_1D.gold')

class tracer1D_read(unittest.TestCase):
    """Test for reading tracer1D."""

    def test_tracer1D_read(self):
        """Test for reading tracer 1D"""
        self.assertTrue(compare_tracer1D())
	os.system('rm -f ' + dir + '/tracer_1D_SC_2.in')
	os.system('rm -f ' + dir + '/tracer_1D_SC_2*.tec')
	os.system('rm -f ' + dir + '/tracer_1D_SC_2*.out')
	os.system('rm -f ' + dir + '/tracer_1D_SC_2*.regression')

if __name__ == '__main__':
    unittest.main()
