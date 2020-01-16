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
""" Hamiltonian class for the OpenFermion-FQE.
"""
#The base class Hamiltonian currently support dense and sparse hamiltonians.
#All the code is funtional but there may be some minor type errors with
#type hinting as sparse and dense return different types in a few places.
#This will be corrected in a future version.

from typing import Tuple
from abc import ABCMeta, abstractmethod

import numpy


class Hamiltonian(metaclass=ABCMeta):
    """ Abstract class to mediate the functions of Hamiltonian with the
    emulator.  Since the structure of the Hamiltonian may contain symmetries
    which can greatly speed up operations that act up on the object, defining
    unique classes for each case can be a key towards making the code more
    efficient.
    """

    def __init__(self, e_0: complex = 0. + 0.j):
        """All hamiltonians share two basic types of information.

        Args:
            e_0 (complex) - the scalar part of the Hamiltonian

        Members:
            self._conserve_number (bool) - whether the Hamiltonian conserves \
                the number symmetry
        """
        self._conserve_number = True 
        self._e_0 = e_0


    @abstractmethod
    def dim(self) -> int:
        """Return the dimension of the hamiltonian
        """
        return 0


    def calc_diag_transform(self) -> numpy.ndarray:
        """Perform a unitary digaonlizing transformation of the one body term
        and return that transformation.
        """
        return numpy.empty(0)



    @abstractmethod
    def rank(self) -> int:
        """Return the dimension of the hamiltonian
        """
        return 0


    def quadratic(self) -> bool:
        """Flag to define the Hamiltonian as quadratic
        """
        return False


    def diagonal(self) -> bool:
        """Flag to define the Hamiltonian as diagonal
        """
        return False


    def diagonal_coulomb(self) -> bool:
        """Flag to define the Hamiltonian as diagonal coulomb
        """
        return False


    def conserve_number(self) -> bool:
        """Flag to define the Hamiltonian as number conserving
        """
        return self._conserve_number


    def e_0(self):
        """Return the scalar potential of the hamiltonian
        """
        return self._e_0


    def iht(self, time: float) -> Tuple[numpy.ndarray, ...]:
        """Return the matrices of the Hamiltonian prepared for time evolution.

        Args:
            time (float) - the time step
        """
        return tuple()


    def tensors(self) -> Tuple[numpy.ndarray, ...]:
        """All tensors are returned in order of their rank.
        """
        return tuple()


    def diag_values(self) -> numpy.ndarray:
        """Return the diagonal values packed into a single dimension
        """
        return numpy.empty(0)


    def transform(self, trans: numpy.ndarray) -> numpy.ndarray:
        """Tranform the one body term using the passed in matrix.
        Care must be taken that this function does not transform the higher-body
        terms even if they exist.

        Args:
            trans (numpy.ndarray) - unitary transformation

        Returns:
            (numpy.ndarray) - transformed one-body Hamiltonian
        """
        return numpy.empty(0)
