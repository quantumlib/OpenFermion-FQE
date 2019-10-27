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

"""Test cases for general hamiltonian initialization
"""

import unittest

import numpy

from fqe.hamiltonians import general_hamiltonian

class GenralHamiltonianTest(unittest.TestCase):
    """Test cases for the General Hamiltonian class
    """


    def test_general_hamiltonian(self):
        """Make sure that the Hamiltonian can be initialized
        """
        symmh = [[[1, 2], 1.0, False]]
        symmg = [[[1, 2, 3, 4], 1.0, False]]
        ham = general_hamiltonian.General(0.0,
                                          numpy.zeros((2, 2),
                                                      dtype=numpy.complex64),
                                          numpy.ones((2, 2, 2, 2),
                                                     dtype=numpy.complex64),
                                          0.0, symmh, symmg)
        self.assertEqual(ham.identity(), 'General')
        ham.potential = 1.0
        self.assertEqual(ham.potential, 1.0)
        self.assertTrue(numpy.allclose(ham.h1e, numpy.zeros((2, 2),
                                                            dtype=numpy.complex64)))
        ham.mu_c = -1.0
        self.assertEqual(ham.mu_c, -1.0)
        self.assertTrue(numpy.allclose(ham.h_mu,
                                       numpy.identity(2,
                                                      dtype=numpy.complex64)))
        self.assertTrue(numpy.allclose(ham.g2e,
                                       numpy.ones((2, 2, 2, 2),
                                                  dtype=numpy.complex64)))
