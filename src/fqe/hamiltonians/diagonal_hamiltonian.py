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

""" Diagonal Hamiltonian
"""

import numpy

from fqe.hamiltonians import hamiltonian


class Diagonal(hamiltonian.Hamiltonian):
    """Diagonal Hamiltonian class.
    """


    def __init__(self, hdiag, conserve_number=True) -> None:
        super().__init__(conserve_number=conserve_number)

        if hdiag.ndim != 1:
            raise ValueError('Incorrect diemsion passed for Diagonal' +
                             ' Hamiltonian Elements')
        self._hdiag = hdiag
        self._dim = self._hdiag.shape[0]


    def dim(self) -> int:
        """
        """
        return self._dim


    @property
    def h1e(self) -> numpy.ndarray:
        """
        """
        h1e = numpy.zeros((self._dim, self._dim)).astype(self._hdiag.dtype)

        for i in range(self._dim):
            h1e[i, i] = self._hdiag[i]

        return h1e


    def rank(self) -> int:
        """
        """
        return 2


    def diagonal(self) -> bool:
        """
        """
        return True


    def tensor(self, rank):
        """
        """
        if rank > 2:
            raise ValueError('Diagonal Hamiltonian does not have greater' \
                             ' than rank 2 elements')
        return self.h1e


    def tensors(self):
        """
        """
        return tuple([self.h1e])


    def quadratic(self) -> bool:
        return True


    def iht(self, time, full=False):
        """
        """
        if full:
            return tuple([-1.j*time*self.h1e])

        return tuple([-1.j*time*self._hdiag])
