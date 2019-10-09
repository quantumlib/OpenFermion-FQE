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

import numpy

import unittest

from fqe.hamiltonians.general_hamiltonian import General

class GenralHamiltonianTest(unittest.TestCase):


    def test_general_hamiltonian(self):
        """Make sure that the Hamiltonian can be initialized
        """
        symmh = [[[1, 2], 1.0, False]]
        symmg = [[[1, 2, 3, 4], 1.0, False]]
        Ham = General(0.0, numpy.zeros((2, 2), dtype=numpy.complex64),
                      numpy.ones((2, 2, 2, 2), dtype=numpy.complex64), 0.0,
                      symmh, symmg )
        self.assertEqual(Ham.identity, 'General')
        Ham.e = 1.0
        self.assertEqual(Ham.e, 1.0)
        self.assertTrue(numpy.allclose(Ham.h, numpy.zeros((2, 2),
                                                          dtype=numpy.complex64)))
        Ham.mu = -1.0
        self.assertEqual(Ham.mu, -1.0)
        self.assertTrue(numpy.allclose(Ham.h, 
                        numpy.identity(2, dtype=numpy.complex64)))
        self.assertTrue(numpy.allclose(Ham.g,
                        numpy.ones((2, 2, 2, 2), dtype=numpy.complex64)))
