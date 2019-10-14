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

""" Hamiltonian class for the OpenFermion-FQE.
"""

import unittest

from .general_hamiltonian import Hamiltonian

import abc
from six import add_metaclass

class DummyHamiltonian(Hamiltonian):
    """A test class for Hamiltonians
    """


    @property
    def identity(self):
        """This is a dummy
        """
        return 'Dummy'


class HamiltonianTest(unittest.TestCase):


    def test_hamiltonian_cant_instantiate(self):
        self.assertRaises(TypeError, Hamiltonian)


    def test_hamiltonian_dummy(self):
        dummy = DummyHamiltonian()
        self.assertEqual(dummy.identity, 'Dummy')
