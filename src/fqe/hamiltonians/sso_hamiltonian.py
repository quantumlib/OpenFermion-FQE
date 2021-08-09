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
import numpy

from fqe.hamiltonians import hamiltonian
from fqe.util import tensors_equal


class SSOHamiltonian(hamiltonian.Hamiltonian):
    """The SSO Hamiltonian is characterized by having distinct alpha and beta
    terms and no alpha/beta mixing blocks. An example is a part of
    non-relativistic molecular Hamiltonians in the trotterized algorithms.
    """

    def __init__(self,
                 tensors: Tuple[numpy.ndarray, ...],
                 e_0: complex = 0.0 + 0.0j) -> None:
        """Initializes an SSOHamiltonian.

        Arguments:
            tensors (Tuple[numpy.ndarray, ...]): Variable length tuple containg \
                between one and four numpy.arrays of increasing rank. \
                The tensors contain the n-body hamiltonian elements. \
                Tensors up to the highest order must be included even if \
                the lower terms are full of zeros.

            e_0 (complex): Scalar potential associated with the Hamiltonian.
        """

        super().__init__(e_0=e_0)
        self._tensor: Dict[int, numpy.ndarray] = {}

        for rank, matrix in enumerate(tensors):
            if not isinstance(matrix, numpy.ndarray):
                raise TypeError(
                    "Arg tensors should be a tuple of numpy.ndarray, but "
                    f"tensors[{rank}] = {type(tensors[rank])}.")
            if matrix.ndim % 2:
                raise ValueError("input tensor has an odd rank")

            self._tensor[2 * (rank + 1)] = matrix

        assert self._tensor, (
            "No matrix elements passed into the SSOHamiltonian.")

        self._quadratic = False
        if len(self._tensor) == 1:
            if 2 in self._tensor.keys():
                self._quadratic = True

        self._dim = list(self._tensor.values())[0].shape[0]

    def __eq__(self, other: object) -> bool:
        """ Comparison operator
        Args:
            other: SSOHamiltonian to be compared against

        Returns:
            (bool): True if equal, otherwise False
        """
        if not isinstance(other, SSOHamiltonian):
            return NotImplemented
        else:
            return self.e_0() == other.e_0() \
                and tensors_equal(self._tensor, other._tensor)

    def iht(self, time: float) -> Tuple[numpy.ndarray, ...]:
        """Returns the matrices of the Hamiltonian prepared for time evolution.

        Args:
            time (float): time associated with the time propagation

        Returns:
            Tuple[numpy.ndarray, ...]: tuple of arrays to be used in time propagation
        """
        iht_mat = []
        for rank in range(len(self._tensor)):
            iht_mat.append(-1.0j * time * self._tensor[2 * (rank + 1)])

        return tuple(iht_mat)

    def dim(self) -> int:
        """
        Returns:
            (int): the orbital dimension of the Hamiltonian arrays.
        """
        return self._dim

    def rank(self) -> int:
        """
        Returns:
            (int): the rank of the largest tensor.
        """
        return 2 * len(self._tensor)

    def tensor(self, rank: int) -> numpy.ndarray:
        """Returns a single nbody tensor based on its rank.

        Args:
            rank (int): rank of the single nbody tensor to return.

        Returns:
            numpy.ndarray: corresponding numpy array
        """
        return self._tensor[rank]

    def tensors(self) -> Tuple[numpy.ndarray, ...]:
        """
        Returns:
            Tuple[numpy.ndarray, ...]: all tensors in order of their rank.
        """
        out = []
        for rank in range(len(self._tensor)):
            out.append(self._tensor[2 * (rank + 1)])
        return tuple(out)

    def quadratic(self) -> bool:
        """
        Returns:
            bool: whether or not the Hamiltonian is quadratic.
        """
        return self._quadratic

    def calc_diag_transform(self) -> numpy.ndarray:
        """Performs a unitary digaonlizing transformation of the one body term
        and returns that transformation.

        Returns:
            numpy.ndarray: unitary transformation matrix
        """
        assert self._tensor[2].shape[0] % 2 == 0
        norb = self._tensor[2].shape[0] // 2
        _, trans1 = numpy.linalg.eigh(self._tensor[2][:norb, :norb])
        _, trans2 = numpy.linalg.eigh(self._tensor[2][norb:, norb:])
        trans = numpy.zeros_like(self._tensor[2])
        trans[:norb, :norb] = trans1[:, :]
        trans[norb:, norb:] = trans2[:, :]
        return trans

    def transform(self, trans: numpy.ndarray) -> numpy.ndarray:
        """Tranforms the one body term using the provided matrix.

        Args:
            trans (numpy.ndarray): Unitary transformation.

        Returns:
            numpy.ndarray: Transformed one-body Hamiltonian as a numpy.ndarray.
        """
        return trans.conj().T @ self._tensor[2] @ trans
