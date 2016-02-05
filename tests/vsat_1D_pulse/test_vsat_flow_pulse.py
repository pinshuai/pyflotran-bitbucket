import unittest
import os

dir = os.path.dirname(os.path.realpath(__file__))


class vsat_read(unittest.TestCase):
    """Test for reading vsat 1D pulse."""

    def setUp(self):
        os.system('python ' + dir + '/vsat_pulse_read.py > /dev/null 2>&1')

    def test_vsat_flow_read(self):
        """Test for reading vsat pulse"""
        gold = ''
        test = ''
        with open(dir + '/vsat_flow2.in', 'r') as f:
            test += f.read()

        with open(dir + '/vsat_flow.gold', 'r') as f:
            gold += f.read()

        self.assertEqual(gold, test)

    def tearDown(self):
        os.system('rm -f ' + dir + '/vsat_flow2.in')


if __name__ == '__main__':
    unittest.main()
