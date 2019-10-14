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

"""Quadratic Hamitlonian
"""

import numpy

from .hamiltonian import Hamiltonian
from fqe.tensor import tensor_utils


class Quadratic(Hamiltonian):


    def __init__(self, pot, h1e, chem, symmh):
        self._pot = None
        self._mu = None
        self._h1e = None
        self.e = pot
        self.set_h(h1e, symmh)
        self.mu = chem


    @property
    def identity(self):
        """This is a quadratic Hamiltonian.  
        """
        return 'Quadratic'


    @property
    def e(self):
        return self._pot


    @property
    def mu(self):
        return self._mu


    @property
    def h(self):
        return self._h1e


    @e.setter
    def e(self, pot):
        self._pot = pot


    @mu.setter
    def mu(self, chem):
        self._mu = chem
        self._h1e -= self._mu*numpy.identity(self._h1e.shape[0])


    def set_h(self, h1e, symmh):
        tensor_utils.confirm_symmetry(h1e, symmh)
        self._h1e = h1e
