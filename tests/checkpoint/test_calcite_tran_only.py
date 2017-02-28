import unittest
import os

dir = os.path.dirname(os.path.realpath(__file__))


class vsat_read(unittest.TestCase):
    """Test for running checkpoint test""" 

    def setUp(self):
        os.system('python ' + dir + '/checkpoint.py > /dev/null 2>&1')

    def test_checkpoint(self):
        """Test for running checkpoint test"""
        gold = ''
        test = ''
        with open(dir + '/calcite_tran_only.in', 'r') as f:
            line = f.readline()
            if 'DATABASE' not in line:
                test += line

        with open(dir + '/calcite_tran_only.gold', 'r') as f:
            line = f.readline()
            if 'DATABASE' not in line:
                gold += line

        self.assertEqual(gold, test)

    def tearDown(self):
        os.system('rm -f ' + dir + '/calcite_tran_only.in')
        os.system('rm -f ' + dir + '/calcite_tran_only.out')
        os.system('rm -f ' + dir + '/calcite_tran_only*.tec')
        os.system('rm -f ' + dir + '/calcite_tran_only*.regression')


if __name__ == '__main__':
    unittest.main()
