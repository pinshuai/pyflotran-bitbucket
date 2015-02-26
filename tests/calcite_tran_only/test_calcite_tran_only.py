import filecmp
import unittest
import os

dir = os.path.dirname(os.path.realpath(__file__))

def compare_calcitetranonly():
    """Return True if pyflotran runs calcite_tran_read.py in correctly."""
    os.system('python ' + dir + '/calcite_tran_only_read.py >& /dev/null ')
    return  filecmp.cmp(dir + '/calcite_tran_only_2.in', dir + '/calcite_tran_only.gold')

class calcitetranonly_read(unittest.TestCase):
    """Test for reading calcitetranonly."""

    def test_calcitetranonly_read(self):
        """Test for reading calcite tran only"""
        self.assertTrue(compare_calcitetranonly())
	os.system('rm -f ' + dir + '/calcite_tran_only_2.in')
	os.system('rm -f ' + dir + '/calcite_tran_only_2.out')
	os.system('rm -f ' + dir + '/calcite_tran_only_2*.tec')
	os.system('rm -f ' + dir + '/calcite_tran_only_2*.regression')


if __name__ == '__main__':
    unittest.main()
