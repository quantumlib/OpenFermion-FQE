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

"""Test cases for Quadratic hamiltonian initialization
"""

import unittest

import numpy

from fqe.hamiltonians import quadratic_hamiltonian

class QuadraticHamiltonianTest(unittest.TestCase):
    """Tests for the Qudratic Hamiltonian Class
    """


    def setUp(self):
        """Set a local type
        """
        self._type = numpy.complex64


    def test_quadratic_hamiltonian(self):
        """Make sure that the Hamiltonian can be initialized
        """
        symmh = [[[1, 2], 1.0, False]]
        ham = quadratic_hamiltonian.Quadratic(0.0,
                                              numpy.zeros((2, 2),
                                                          dtype=self._type),
                                              0.0, symmh)
        self.assertEqual(ham.identity(), 'Quadratic')
        ham.potential = 1.0
        self.assertEqual(ham.potential, 1.0)
        self.assertTrue(numpy.allclose(ham.h1e,
                                       numpy.zeros((2, 2),
                                                   dtype=self._type)))
        ham.mu_c = -1.0
        self.assertEqual(ham.mu_c, -1.0)
        self.assertTrue(numpy.allclose(ham.h_mu,
                                       numpy.identity(2, dtype=self._type)))
