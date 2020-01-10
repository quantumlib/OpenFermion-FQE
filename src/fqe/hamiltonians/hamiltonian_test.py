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
"""Base class for Hamiltonians in the fqe
"""

import unittest

from fqe.hamiltonians import hamiltonian


class TestHamiltonian(unittest.TestCase):
    """Test class for the base Hamiltonian
    """


    def test_general(self):
        """The base Hamiltonian initializes some common vaiables
        """
        class Test(hamiltonian.Hamiltonian):
            """A testing dummy class
            """

            def __init__(self, conserve_number):
                super().__init__(conserve_number=conserve_number)

            def dim(self) -> int:
                return 1


            def rank(self) -> int:
                return 1


        test = Test(True)
        self.assertEqual(test.dim(), 1)
        self.assertEqual(test.rank(), 1)
        self.assertEqual(test.e_0(), 0. + 0.j)
        self.assertFalse(test.quadratic())
        self.assertFalse(test.diagonal())
        self.assertFalse(test.diagonal_coulomb())
        self.assertTrue(test.conserve_number())
