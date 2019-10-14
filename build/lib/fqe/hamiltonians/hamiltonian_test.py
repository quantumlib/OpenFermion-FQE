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

from fqe.hamiltonians import hamiltonian

class DummyHamiltonian(hamiltonian.Hamiltonian):
    """A test class for Hamiltonians
    """


    def __init__(self):
        """Write over base class
        """


    def identity(self):
        """This is a dummy
        """
        return 'Dummy'


class HamiltonianTest(unittest.TestCase):
    """Tests for the Hamiltonian base class
    """


    def test_hamiltonian_cant_instantiate(self):
        """An unnamed Hamiltonian should not be created
        """
        self.assertRaises(TypeError, hamiltonian.Hamiltonian)


    def test_hamiltonian_dummy(self):
        """Check that the Hamiltonian has a name
        """
        dummy = DummyHamiltonian()
        self.assertEqual(dummy.identity(), 'Dummy')
