import filecmp
import unittest
import os

dir = os.path.dirname(os.path.realpath(__file__))

def compare_mphase():
    """Return True if pyflotran writes files in multiple directories correctly."""
    os.system('python ' + dir + '/mphase_multiple.py >& /dev/null')
    dir1 = '/mphase-run1'
    dir2 = '/mphase-run2'

    res1 =  filecmp.cmp(dir + dir1 + '/mphase.in', dir + '/mphase.gold')
    res2 =  filecmp.cmp(dir + dir2 + '/mphase.in', dir + '/mphase.gold')

    os.system('rm -rf ' +  dir + dir1)
    os.system('rm -rf ' +  dir + dir2)

    res = res1 and res2
 
    return res

class mphase_read(unittest.TestCase):
    """Test for writing to multiple directories from PyFLOTRAN input for mphase."""

    def test_mphase_multi(self):
        """Test for writing to multiple directories from PyFLOTRAN input for mphase."""
        self.assertTrue(compare_mphase())

if __name__ == '__main__':
    unittest.main()
