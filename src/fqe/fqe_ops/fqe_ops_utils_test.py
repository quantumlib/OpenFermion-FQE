#   Copyright 2019 Quantum Simulation Technologies Inc.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import unittest

from fqe.fqe_ops import fqe_ops_utils


class TestFqeUtils(unittest.TestCase):


    def test_validate_rdm_string(self):
        """
        """
        rdm1 = '1^ 2'
        rdm2 = '7^ 8 23 1^'
        rdm3 = '0^ 1 2^ 3 4^ 5'
        rdm4 = '10^ 9^ 8^ 7^ 6 5 4 3'
        self.assertEqual(fqe_ops_utils.validate_rdm_string(rdm1, 1), 'element')
        self.assertEqual(fqe_ops_utils.validate_rdm_string(rdm2, 2), 'element')
        self.assertEqual(fqe_ops_utils.validate_rdm_string(rdm3, 3), 'element')
        self.assertEqual(fqe_ops_utils.validate_rdm_string(rdm4, 4), 'element')
        rdm1 = 'i^ j'
        rdm2 = 'k f^ l t^'
        rdm3 = 'x^ q w^ k u^ m'
        rdm4 = 'v^ b^ n^ m^ d f g h'
        self.assertEqual(fqe_ops_utils.validate_rdm_string(rdm1, 1), 'tensor')
        self.assertEqual(fqe_ops_utils.validate_rdm_string(rdm2, 2), 'tensor')
        self.assertEqual(fqe_ops_utils.validate_rdm_string(rdm3, 3), 'tensor')
        self.assertEqual(fqe_ops_utils.validate_rdm_string(rdm4, 4), 'tensor')


if __name__ == '__main__':
    unittest.main()
