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
"""Defines the SparseHamiltonian object."""

# The protected member access does not
# mutate passed in data and so can be exposed.
# pylint: disable=protected-access
# pylint: disable=too-many-locals

import copy
from typing import List, Union, Tuple

from openfermion import FermionOperator
from openfermion.transforms import normal_ordered

from fqe.hamiltonians import hamiltonian, hamiltonian_utils


class SparseHamiltonian(hamiltonian.Hamiltonian):
    """The Sparse Hamiltonian is characterized by having only one or a few
    elements which are non-zero. This can provide advantages for certain
    operations where single element access is preferred.
    """

    def __init__(
            self,
            operators: Union[FermionOperator, str],
            conserve_spin: bool = True,
            e_0: complex = 0.0 + 0.0j,
    ) -> None:
        """Initializes a SparseHamiltonian.

        Args:
            operators (FermionOperator or str): Operator with a coefficient \
                in the FermionOperator format.

            conserve_spin (bool): Whether or not to conserve the Sz symmetry.

            e_0 (complex): Scalar part of the Hamiltonian.
        """
        if isinstance(operators, str):
            ops = FermionOperator(operators, 1.0)
        else:
            ops = operators

        ops = normal_ordered(ops)

        work = ops.terms.pop((), None)
        if work is not None:
            e_0 += work

        super().__init__(e_0=e_0)

        self._operators: List[
            Tuple[complex, List[Tuple[int, int]], List[Tuple[int, int]]]] = []
        self._conserve_spin = conserve_spin

        self._rank = 0
        for prod in ops.terms:
            self._rank = max(self._rank, len(prod))

        for oper in ops.get_operators():
            (
                coeff,
                phase,
                alpha_block,
                beta_block,
            ) = hamiltonian_utils.gather_nbody_spin_sectors(oper)

            alpha_out: List[Tuple[int, int]] = []
            beta_out: List[Tuple[int, int]] = []
            for alpha in alpha_block:
                alpha_out.append((alpha[0] // 2, alpha[1]))
            for beta in beta_block:
                beta_out.append((beta[0] // 2, beta[1]))
            self._operators.append((coeff * phase, alpha_out, beta_out))

    def __eq__(self, other: object) -> bool:
        """ Comparison operator

        Args:
            other: SparseHamiltonian to be compared against

        Returns:
            True if equal, otherwise False
        """
        if not isinstance(other, SparseHamiltonian):
            return NotImplemented
        else:
            return self.e_0() == other.e_0() \
                and self._conserve_spin == other._conserve_spin \
                and self._operators == other._operators

    def dim(self):
        """Dim is the orbital dimension of the Hamiltonian arrays.
        This function should not be used with SparseHamiltonian
        """
        raise NotImplementedError

    def rank(self) -> int:
        """
        Returns:
            (int): the rank of the largest tensor.
        """
        return self._rank

    def nterms(self) -> int:
        """
        Returns:
            (int): the number of non-zero elements in the Hamiltonian.
        """
        return len(self._operators)

    def is_individual(self) -> bool:
        """
        Returns:
            (bool): True if this Hamiltonian consists of an individual \
                operator plus its Hermitian conjugate.
        """
        nterm = 0
        for (_, alpha, beta) in self._operators:
            daga = []
            dagb = []
            undaga = []
            undagb = []
            for oper in alpha:
                if oper[1] == 1:
                    daga.append(oper[0])
                else:
                    undaga.append(oper[0])
            for oper in beta:
                if oper[1] == 1:
                    dagb.append(oper[0])
                else:
                    undagb.append(oper[0])
            if daga == undaga and dagb == undagb:
                nterm += 2
            else:
                nterm += 1
        return nterm < 3

    def iht(self, time: float) -> 'SparseHamiltonian':
        """Return the matrices of the Hamiltonian prepared for time evolution.

        Args:
            time (float): The time step.

        Returns:
            SparseHamiltonian: Hamiltonian to be used in time evolution
        """
        out = copy.deepcopy(self)
        for index in range(len(out._operators)):
            (coeff, alpha, beta) = out._operators[index]
            out._operators[index] = (-coeff * 1.0j * time, alpha, beta)
        return out

    def terms(
            self
    ) -> List[Tuple[complex, List[Tuple[int, int]], List[Tuple[int, int]]]]:
        """
        Returns:
            List[Tuple[complex, List[Tuple[int, int]], List[Tuple[int, int]]]]:
                the operators that comprise the SparseHamiltonian.
        """
        return self._operators

    def terms_hamiltonian(self) -> List['SparseHamiltonian']:
        """
        Returns:
            List['SparseHamiltonian']: a list of terms where each of the terms is \
                represented as a SparseHamiltonian.
        """
        out = []
        for current in self._operators:
            tmp = copy.deepcopy(self)
            tmp._operators = [current]
            out.append(tmp)
        return out
