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
"""Defines the DiagonalHamiltonian class."""

import copy
from typing import TYPE_CHECKING

import numpy as np

from fqe.hamiltonians import hamiltonian

if TYPE_CHECKING:
    from numpy import ndarray as Nparray


class Diagonal(hamiltonian.Hamiltonian):
    """
    One-body diagonal Hamiltonian class. Diagonal Hamiltonians are defined as
    those that are diagonal in the Slater determinant space, namely,

    .. math::
        \\langle I|\\hat{H}|J\\rangle = p_I \\delta_{IJ}

    where I and J are Slater determinants, and p is some phase. Generally
    such Hamiltonians can be written as

    .. math::
        \\hat{H} =  = E_0 + \\sum_r h_{rr} a_r^\\dagger a_r
    """

    def __init__(self, hdiag: 'Nparray', e_0: complex = 0.0 + 0.0j) -> None:
        """
        Args:
            hdiag: A rank-1 numpy.array that contains the diagonal part of the
                   1-body Hamiltonian elements.
            e_0: Scalar potential associated with the Hamiltonian.
        """

        super().__init__(e_0=e_0)

        if hdiag.ndim != 1:
            raise ValueError(
                "Incorrect dimension passed for DiagonalHamiltonian elements. "
                f"Must have hdiag.ndim = 1 but hdiag.ndim = {hdiag.ndim}.")
        self._hdiag = hdiag
        self._dim = self._hdiag.shape[0]

    def dim(self) -> int:
        """Returns the orbital dimension of the Hamiltonian arrays."""
        return self._dim

    def rank(self) -> int:
        """Returns the rank of the largest tensor."""
        return 2

    def diagonal(self) -> bool:
        """Returns whether or not the Hamiltonian is diagonal."""
        return True

    def quadratic(self) -> bool:
        """Returns whether or not the Hamiltonian is quadratic."""
        return True

    def diag_values(self) -> np.ndarray:
        """Returns the diagonal values packed into a single dimension."""
        return self._hdiag

    def iht(self, time: float) -> 'Diagonal':
        """Returns the matrices of the Hamiltonian prepared for time evolution.

        Args:
            time: The time step.
        """
        out = copy.deepcopy(self)
        out._hdiag *= -1.0j * time
        return out
