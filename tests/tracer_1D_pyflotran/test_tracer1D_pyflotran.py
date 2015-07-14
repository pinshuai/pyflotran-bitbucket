import unittest
import os

dir = os.path.dirname(os.path.realpath(__file__))

class tracer_read(unittest.TestCase):
    """Test for running tracer_1D pyflotran"""

    def setUp(self):
        os.system('python ' + dir + '/tracer_1D_pyflotran.py > /dev/null 2>&1')

    def test_mphase_read(self):
        """Test for running tracer_1D pyflotran"""
        gold = ''
        test = ''
        with open(dir + '/tracer_1D.in', 'r') as f:
            test += f.read()

        with open(dir + '/tracer_1D.gold', 'r') as f:
            gold += f.read()

        self.assertEqual(gold, test)

    def tearDown(self):
        os.system('rm -f ' + dir + '/tracer_1D.in')
        os.system('rm -f ' + dir + '/tracer_1D*.tec')
        os.system('rm -f ' + dir + '/tracer_1D*.h5')
        os.system('rm -f ' + dir + '/tracer_1D.out')


if __name__ == '__main__':
    unittest.main()