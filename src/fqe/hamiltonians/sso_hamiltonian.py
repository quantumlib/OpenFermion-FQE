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
"""Defines Hamiltonian class for the Spin-conserving Spin Orbital Hamiltonian.
"""
# Pylint does not like zeros_like initializer from numpy
# pylint: disable=unsupported-assignment-operation

from typing import Dict, Tuple

import numpy as np

from fqe.hamiltonians import hamiltonian


class SSOHamiltonian(hamiltonian.Hamiltonian):
    """The SSO Hamiltonian is characterized by having distinct alpha and beta
    terms and no alpha/beta mixing blocks. An example is a part of
    non-relativistic molecular Hamiltonians in the trotterized algorithms.
    """

    def __init__(self,
                 tensors: Tuple[np.ndarray, ...],
                 e_0: complex = 0.0 + 0.0j) -> None:
        """Initializes an SSOHamiltonian.

        Arguments:
            tensors: Variable length tuple containg between one and four
                     numpy.arrays of increasing rank. The tensors contain the
                     n-body hamiltonian elements. Tensors up to the highest
                     order must be included even if the lower terms are full of
                     zeros.
            e_0: Scalar potential associated with the Hamiltonian.
        """

        super().__init__(e_0=e_0)
        self._tensor: Dict[int, np.ndarray] = {}

        for rank, matrix in enumerate(tensors):
            if not isinstance(matrix, np.ndarray):
                raise TypeError(
                    "Arg tensors should be a tuple of numpy.ndarray, but "
                    f"tensors[{rank}] = {type(tensors[rank])}.")
            # TODO: Raise error instead of assert clause.
            assert (matrix.ndim % 2) == 0

            self._tensor[2 * (rank + 1)] = matrix

        assert self._tensor, (
            "No matrix elements passed into the SSOHamiltonian.")

        self._quadratic = False
        if len(self._tensor) == 1:
            if 2 in self._tensor.keys():
                self._quadratic = True

        self._dim = list(self._tensor.values())[0].shape[0]

    def iht(self, time: float) -> Tuple[np.ndarray, ...]:
        """Return the matrices of the Hamiltonian prepared for time evolution.

        Args:
            time: The time step.
        """
        iht_mat = []
        for rank in range(len(self._tensor)):
            iht_mat.append(-1.0j * time * self._tensor[2 * (rank + 1)])

        return tuple(iht_mat)

    def dim(self) -> int:
        """Returns the orbital dimension of the Hamiltonian arrays."""
        return self._dim

    def rank(self) -> int:
        """Returns the rank of the largest tensor."""
        return 2 * len(self._tensor)

    def tensor(self, rank: int) -> np.ndarray:
        """Returns a single nbody tensor based on its rank.

        Args:
            rank: Indexes the single nbody tensor to return.
        """
        return self._tensor[rank]

    def tensors(self) -> Tuple[np.ndarray, ...]:
        """Returns all tensors in order of their rank."""
        out = []
        for rank in range(len(self._tensor)):
            out.append(self._tensor[2 * (rank + 1)])
        return tuple(out)

    def quadratic(self) -> bool:
        """Returns True if the Hamiltonian is quadratic, else False."""
        return self._quadratic

    def calc_diag_transform(self) -> np.ndarray:
        """Performs a unitary digaonlizing transformation of the one body term
        and returns that transformation.
        """
        assert self._tensor[2].shape[0] % 2 == 0
        norb = self._tensor[2].shape[0] // 2
        _, trans1 = np.linalg.eigh(self._tensor[2][:norb, :norb])
        _, trans2 = np.linalg.eigh(self._tensor[2][norb:, norb:])
        trans = np.zeros_like(self._tensor[2])
        trans[:norb, :norb] = trans1[:, :]
        trans[norb:, norb:] = trans2[:, :]
        return trans

    def transform(self, trans: np.ndarray) -> np.ndarray:
        """Tranforms the one body term using the provided matrix.

        Args:
            trans: Unitary transformation.

        Returns:
            Transformed one-body Hamiltonian as a numpy.ndarray.
        """
        return trans.conj().T @ self._tensor[2] @ trans
