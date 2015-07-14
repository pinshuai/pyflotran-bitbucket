import unittest
import os

dir = os.path.dirname(os.path.realpath(__file__))
dir1 = '/mphase-run1'
dir2 = '/mphase-run2'


class mphase_read(unittest.TestCase):
    """Test for reading mphase."""

    def setUp(self):
        os.system('python ' + dir + '/mphase_multiple.py > /dev/null 2>&1')

    def test_mphase_read(self):
        """Test for reading mphase"""

        gold = ''
        test = ''

        with open(dir + dir1 + '/mphase2.in', 'r') as f:
            line = f.readline()
            if not 'CO2_DATABASE' in line:
                test += line

        with open(dir + '/mphase.gold', 'r') as f:
            line = f.readline()
            if not 'CO2_DATABASE' in line:
                gold += line

        self.assertEqual(gold, test)

        gold = ''
        test = ''

        with open(dir + dir2 + '/mphase2.in', 'r') as f:
            line = f.readline()
            if not 'CO2_DATABASE' in line:
                test += line

        with open(dir + '/mphase.gold', 'r') as f:
            line = f.readline()
            if not 'CO2_DATABASE' in line:
                gold += line

        self.assertEqual(gold, test)

    def tearDown(self):
        os.system('rm -rf ' + dir + dir1)
        os.system('rm -rf ' + dir + dir2)


if __name__ == '__main__':
    unittest.main()
