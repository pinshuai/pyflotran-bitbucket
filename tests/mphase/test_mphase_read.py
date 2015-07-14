import filecmp
import unittest
import os

dir = os.path.dirname(os.path.realpath(__file__))

class mphase_read(unittest.TestCase):
    """Test for reading mphase."""

    def setUp(self):
        os.system('python ' + dir + '/mphase_read.py > /dev/null 2>&1')

    def test_mphase_read(self):
        """Test for reading mphase"""
        gold = ''
        test = ''
        with open('mphase2.in', 'r') as f:
            line = f.readline()
            if not 'CO2_DATABASE' in line:
                test += line

        with open('mphase.gold', 'r') as f:
            line = f.readline()
            if not 'CO2_DATABASE' in line:
                gold += line

        self.assertEqual(gold, test)

    def tearDown(self):
        os.system('rm -f ' + dir + '/mphase2.in')
        os.system('rm -f ' + dir + '/mphase2.out')
        os.system('rm -f ' + dir + '/mphase2*.tec')
        os.system('rm -f ' + dir + '/mphase2*.h5')
        os.system('rm -f ' + dir + '/mphase2*.dat')
        os.system('rm -f ' + dir + '/mphase2*.regression')

if __name__ == '__main__':
    unittest.main()
