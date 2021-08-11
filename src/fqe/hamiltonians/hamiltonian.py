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
"""Defines the base Hamiltonian class for OpenFermion-FQE."""

#  The base class Hamiltonian currently support dense and sparse hamiltonians.
#  All the code is funtional but there may be some minor type errors with
#  type hinting as sparse and dense return different types in a few places.

from abc import ABCMeta, abstractmethod
from typing import Any, Tuple, Dict

import numpy


class Hamiltonian(metaclass=ABCMeta):
    """Abstract class to mediate the functions of Hamiltonian with the
    emulator.
    """

    def __init__(self, e_0: complex = 0.0 + 0.0j):
        """All hamiltonians share two basic types of information.

        Args:
            e_0: The scalar part of the Hamiltonian
        """
        self._conserve_number = True
        self._e_0 = e_0

    @abstractmethod
    def dim(self) -> int:
        """
        Returns:
            (int): the orbital dimension of the Hamiltonian arrays.
        """
        return 0

    def calc_diag_transform(self) -> numpy.ndarray:
        """Performs a unitary digaonlizing transformation of the one-body term
        and returns that transformation.

        Returns:
            (numpy.ndarray): Matrix representation of the transformation
        """
        return numpy.empty(0)

    @abstractmethod
    def rank(self) -> int:
        """
        Returns:
            (int): the rank of the largest tensor
        """
        return 0

    def quadratic(self) -> bool:
        """
        Returns:
            (bool): True if the Hamiltonian is quadratic, else False.
        """
        return False

    def diagonal(self) -> bool:
        """
        Returns:
            (bool): True if the Hamiltonian is diagonal, else False.
        """
        return False

    def diagonal_coulomb(self) -> bool:
        """
        Returns:
            (bool): True if the Hamiltonian is diagonal coloumb, else False
        """
        return False

    def conserve_number(self) -> bool:
        """
        Returns:
            (bool): True if the Hamiltonian is number conserving, else False
        """
        return self._conserve_number

    def e_0(self):
        """
        Returns:
            (complex): the scalar potential of the Hamiltonian
        """
        return self._e_0

    def iht(self, time: float) -> Any:
        """Return the matrices of the Hamiltonian prepared for time evolution.

        Args:
            time: The time step.

        Returns:
            Tuple[numpy.ndarray, ...]: tuple of arrays to be used in time \
                propagation
        """
        return tuple()

    def tensors(self) -> Tuple[numpy.ndarray, ...]:
        """
        Returns:
            Tuple[numpy.ndarray, ...]: tuple of tensors
        """
        return tuple()

    def diag_values(self) -> numpy.ndarray:
        """Returns the diagonal values packed into a single dimension."""
        return numpy.empty(0)

    def transform(self, trans: numpy.ndarray) -> numpy.ndarray:
        """Tranform the one body term using the provided matrix.

        Note: Care must be taken that this function does not transform the
        higher-body terms even if they exist.

        Args:
            trans: Unitary transformation.

        Returns:
            numpy.ndarray: Transformed one-body Hamiltonian as a numpy.ndarray.
        """
        return numpy.empty(0)
