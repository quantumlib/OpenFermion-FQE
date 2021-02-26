#   Copyright 2020 Google LLC

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
"""Defines the DiagonalCoulomb Hamiltonian class."""
from typing import Dict

import numpy as np

from fqe.hamiltonians import hamiltonian


class DiagonalCoulomb(hamiltonian.Hamiltonian):
    """The diagonal coulomb Hamiltonian is characterized as being a two-body
    operator with a specific structure such that it is the product of two
    number operators. It is generally written as

    .. math::
        \\hat{H} = E_0 + \\sum_r f_r \\hat{n}_r
            + \\sum_{rs} v_{rs} \\hat{n}_r \\hat{n}_s

    where n is a number operator. Note that this Hamiltonian is diagonal
    in the Slater determinant space,

    .. math::
        \\langle I|\\hat{H}|J\\rangle = p_I \\delta_{IJ}

    where p is an appropriate factor.
    """

    def __init__(self, h2e: np.ndarray, e_0: complex = 0.0 + 0.0j) -> None:
        """Initialize a DiagonalCoulomb Hamiltonian.

        Args:
            h2e: either (1) a dense rank-2 array that contains the diagonal
                 elements :math:`|v_{rs}|` above, or (2) a dense rank-4 array
                 in the format used for two-body operator in the dense
                 Hamiltonian code.

            e_0: Scalar potential associated with the Hamiltonian.
        """

        super().__init__(e_0=e_0)
        diag = np.zeros(h2e.shape[0], dtype=h2e.dtype)
        self._dim = h2e.shape[0]
        self._tensor: Dict[int, np.ndarray] = {}

        if h2e.ndim == 2:
            self._tensor[1] = diag
            self._tensor[2] = h2e

        elif h2e.ndim == 4:
            for k in range(self._dim):
                diag[k] += h2e[k, k, k, k]

            vij = np.zeros((self._dim, self._dim), dtype=h2e.dtype)
            for i in range(self._dim):
                for j in range(self._dim):
                    vij[i, j] -= h2e[i, j, i, j]

            self._tensor[1] = diag
            self._tensor[2] = vij

    def dim(self) -> int:
        """Returns is the orbital dimension of the Hamiltonian arrays."""
        return self._dim

    def diagonal_coulomb(self) -> bool:
        """Returns whether or not the Hamiltonian is diagonal_coulomb."""
        return True

    def rank(self) -> int:
        """Returns the rank of the largest tensor."""
        return 4

    def iht(self, time: float):
        """Returns the matrices of the Hamiltonian prepared for time evolution.

        Args:
            time: The time step.
        """
        iht_mat = []
        for rank in range(len(self._tensor)):
            iht_mat.append(-1.0j * time * self._tensor[rank + 1])

        return tuple(iht_mat)
