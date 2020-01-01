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

"""Generalized Spin Orbita; Hamiltonian
"""

import numpy
from numpy import linalg

from fqe.hamiltonians import hamiltonian, hamiltonian_utils


class GSOHamiltonian(hamiltonian.Hamiltonian):
    """
    """


    def __init__(self, tensor, conserve_number=True) -> None:
        """
        """
        super().__init__(conserve_number)

        self._tensor = {}
        for rank in range(len(tensor)):
            if tensor[rank].ndim % 2:
                raise ValueError('Odd rank tensor not supported in Hamiltonians')

            self._tensor[2*(rank + 1)] = tensor[rank]

        if not self._tensor:
            raise ValueError('No matrix elements passed into' \
                             ' the general hamiltonian')

        self._quadratic = False
        if len(self._tensor) == 1:
            if 2 in self._tensor.keys():
                self._quadratic = True

        self._dim = list(self._tensor.values())[0].shape[0]


    def iht(self, time, full=True):
        """
        """
        iht_mat = []
        for rank in range(len(self._tensor)):
            iht_mat.append(-1.j*time*self._tensor[2*(rank + 1)])

        return tuple(iht_mat)


    def dim(self) -> int:
        return self._dim


    def rank(self):
        """
        """
        return 2*len(self._tensor)


    def tensor(self, rank):
        """
        """
        return self._tensor[rank]


    def tensors(self):
        """
        """
        out = []
        for rank in range(len(self._tensor)):
            out.append(self._tensor[2*(rank + 1)])
        return tuple(out)


    def quadratic(self) -> bool:
        return self._quadratic


    def calc_diag_transform(self):
        """
        """
        _, trans = linalg.eigh(self._tensor[2])
        return trans


    def transform(self, trans):
        """Using the transformation stored, mutate the hamiltonian to
        diagonal
        """
        return trans.conj().T @ self._tensor[2] @ trans
