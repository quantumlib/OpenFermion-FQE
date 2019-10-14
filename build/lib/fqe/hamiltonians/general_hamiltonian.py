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

from fqe.hamiltonians import hamiltonian
from fqe.tensor import tensor_utils


class General(hamiltonian.Hamiltonian):
    """A general hamiltonian class.
    """


    def __init__(self, pot, h1e, g2e, chem, symmh, symmg):
        """This hamiltonian has a potential, one body and two body terms.
        Any symmetries defined for the system must be explicitly added.

        Args:
            pot (complex) - a complex scalar
            h1e (numpy.array(dim=2, dtype=complex64)) - matrix elements for
                single particle states
            g2e (numpy.array(dim=4, dtype=complex64)) - matrix elements for
                two particle states
            chem (double) - a value for the chemical potential
            symmh (list[list[int], double, bool]) - symmetry permutations for
                the one body matrix elements
            symmg (list[list[int], double, bool]) - symmetry permutations for
                the two body matrix elements
        """
        self._potential = pot
        self._h1e = None
        self._g2e = None
        self.set_h(h1e, symmh)
        self._mu = chem
        self.set_g(g2e, symmg)


    def identity(self):
        """This is the most generic and general hamiltonian supported.
        """
        return 'General'


    @property
    def potential(self):
        """Constant potential
        """
        return self._potential


    @property
    def mu_c(self):
        """Chemical potential
        """
        return self._mu


    @property
    def h1e(self):
        """One electron hamiltonian matrix elements
        """
        return self._h1e


    @property
    def h_mu(self):
        """One electron hamiltonian matrix elements
        """
        return self._h1e - self._mu*numpy.identity(self._h1e.shape[0])


    @property
    def g2e(self):
        """Two electron hamiltonian matrix elements
        """
        return self._g2e


    @potential.setter
    def potential(self, pot):
        self._potential = pot


    @mu_c.setter
    def mu_c(self, chem):
        self._mu = chem


    def set_h(self, h1e, symmh):
        """Set the one particle matrix elements with optional symmetry passed
        """
        tensor_utils.confirm_symmetry(h1e, symmh)
        self._h1e = h1e


    def set_g(self, g2e, symmg):
        """Set the two particle matrix elements with optional symmetry passed
        """
        tensor_utils.confirm_symmetry(g2e, symmg)
        self._g2e = g2e
