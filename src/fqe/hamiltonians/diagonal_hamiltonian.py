#   Copyright 2020 Google

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

from typing import Tuple

import copy

import numpy

from fqe.hamiltonians import hamiltonian


class Diagonal(hamiltonian.Hamiltonian):
    """Diagonal Hamiltonian class.
    """


    def __init__(self,
                 hdiag: numpy.array,
                 e_0: complex = 0. + 0.j) -> None:
        """
        Args:
            hdiag (numpy.array) - a variable length tuple containg between \
                one and four numpy.arrays of increasing rank.  The tensors \
                contain the n-body hamiltonian elements.  Tensors up to the \
                highest order must be included even if the lower terms are full \
                of zeros.

            e_0 (complex) - this is a scalar potential associated with the \
                Hamiltonian.
        """

        super().__init__(e_0=e_0)

        if hdiag.ndim != 1:
            raise ValueError('Incorrect diemsion passed for Diagonal' +
                             ' Hamiltonian Elements')
        self._hdiag = hdiag
        self._dim = self._hdiag.shape[0]


    def dim(self) -> int:
        """Dim is the orbital dimension of the Hamiltonian arrays.
        """
        return self._dim


    def rank(self) -> int:
        """This returns the rank of the largest tensor.
        """
        return 2


    def diagonal(self) -> bool:
        """Indicate the that the Hamiltonian is diagonal
        """
        return True


    def quadratic(self) -> bool:
        """Flag to indicate this is a quadratic hamiltonian
        """
        return True


    def diag_values(self) -> numpy.ndarray:
        """Return the diagonal values packed into a single dimension
        """
        return self._hdiag


    def iht(self, time: float) -> Tuple[numpy.ndarray, ...]:
        """Return the matrices of the Hamiltonian prepared for time evolution.
        """
        out = copy.deepcopy(self)
        out._hdiag *= -1.j*time
        return out
