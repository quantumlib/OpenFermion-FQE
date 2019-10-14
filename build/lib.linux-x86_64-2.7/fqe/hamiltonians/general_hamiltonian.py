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

"""The most generally defined Hamiltonian
"""

import numpy

from fqe.hamiltonians.hamiltonian import Hamiltonian
from fqe.tensor.tensor_utils import confirm_symmetry


class General(Hamiltonian):


    def __init__(self, pot, h1e, g2e, chem, symmh, symmg):
        """This hamiltonian will have a potential, one body and two body terms.
        Any symmetries defined for the system must be explicitly added.

        Args:
            pot (complex) - a complex scalar 
            h1p (numpy.array(dim=2, dtype=complex64)) - matrix elements for
                single particle states
            g2p (numpy.array(dim=4, dtype=complex64)) - matrix elements for
                two particle states
            symmh (numpy.array(dim=4, dtype=complex64)) - matrix elements for
                two particle states
        """
        self._pot = None
        self._mu = None
        self._h1e = None
        self._g2e = None
        self.e = pot
        self.set_h(h1e, symmh)
        self.mu = chem
        self.set_g(g2e, symmg)


    @property
    def identity(self):
        """This is the most generic and general hamiltonian supported.
        """
        return 'General'


    @property
    def e(self):
        return self._pot


    @property
    def mu(self):
        return self._mu


    @property
    def h(self):
        return self._h1e


    @property
    def g(self):
        return self._g2e


    @e.setter
    def e(self, pot):
        self._pot = pot


    @mu.setter
    def mu(self, chem):
        self._mu = chem
        self._h1e -= self._mu*numpy.identity(self._h1e.shape[0])


    def set_h(self, h1e, symmh):
        confirm_symmetry(h1e, symmh)
        self._h1e = h1e


    def set_g(self, g2e, symmg):
        confirm_symmetry(g2e, symmg)
        self._g2e = g2e
