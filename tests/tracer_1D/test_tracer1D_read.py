import unittest
import os

dir = os.path.dirname(os.path.realpath(__file__))

class tracer_read(unittest.TestCase):
    """Test for reading tracer_1D"."""

    def setUp(self):
        os.system('python ' + dir + '/tracer_1D_read.py > /dev/null 2>&1')

    def test_mphase_read(self):
        """Test for reading tracer_1D"""
        gold = ''
        test = ''
        with open('tracer_1D_SC_2.in', 'r') as f:
            test += f.read()

        with open('tracer_1D.gold', 'r') as f:
            gold += f.read()

        self.assertEqual(gold, test)

    def tearDown(self):
        os.system('rm -f ' + dir + '/tracer_1D_SC_2.in')


if __name__ == '__main__':
    unittest.main()