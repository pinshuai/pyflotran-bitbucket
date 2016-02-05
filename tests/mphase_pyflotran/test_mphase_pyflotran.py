import unittest
import os

dir = os.path.dirname(os.path.realpath(__file__))


class mphase_read(unittest.TestCase):
    """Test for mphase pyflotran."""

    def setUp(self):
        os.system('python ' + dir + '/mphase.py > /dev/null 2>&1')

    def test_mphase_read(self):
        """Test for mphase pyflotran"""
        gold = ''
        test = ''
        with open(dir + '/mphase.in', 'r') as f:
            line = f.readline()
            if 'CO2_DATABASE' not in line:
                test += line

        with open(dir + '/mphase.gold', 'r') as f:
            line = f.readline()
            if 'CO2_DATABASE' not in line:
                gold += line

        self.assertEqual(gold, test)

    def tearDown(self):
        os.system('rm -f ' + dir + '/mphase.in')
        os.system('rm -f ' + dir + '/mphase.out')
        os.system('rm -f ' + dir + '/mphase*.tec')
        os.system('rm -f ' + dir + '/mphase*.h5')
        os.system('rm -f ' + dir + '/mphase*.dat')
        os.system('rm -f ' + dir + '/mphase*.regression')


if __name__ == '__main__':
    unittest.main()
