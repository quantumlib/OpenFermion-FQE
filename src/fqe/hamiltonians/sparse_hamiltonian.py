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
"""Hamiltonian class for the Sparse Hamiltonain.
"""
#The protected member access does not mutate passed in data and so can be exposed.
#pylint: disable=protected-access

from typing import Any, List, Union, Tuple
import copy

from openfermion import FermionOperator
from openfermion.utils import normal_ordered

from fqe.hamiltonians import hamiltonian, hamiltonian_utils


class SparseHamiltonian(hamiltonian.Hamiltonian):
    """The Sparse Hamiltonian is characterized by having only one or a few
    elements which are non-zero.  This can provide advantages for certain
    operations where single element access is preferred.
    """

    def __init__(self,
                 norb: int,
                 operators: Union[FermionOperator, str],
                 conserve_spin: bool = True,
                 e_0: complex = 0. + 0.j) -> None:
        """
        Args:
            norb (int) - the number of orbitals

            operators(Union[FermionOperator, str]) - operator with a \
                coefficient in the FermionOperator format.

            conserve_spin (bool) - whether or not to conserve the Sz symmetry

            e_0 (complex) - scalar part of the Hamiltonian
        """
        if isinstance(operators, str):
            operators = FermionOperator(operators, 1.0)

        operators = normal_ordered(operators)

        work = operators.terms.pop((), None)
        if work is not None:
            e_0 += work

        super().__init__(e_0=e_0)

        self._norb = norb
        self._operators: List[Tuple[complex,
                                    List[Tuple[int, int]],
                                    List[Tuple[int, int]]]] = []
        self._conserve_spin = conserve_spin
        self._matrix_data: List[List[Any]] = []

        for prod in operators.terms:
            work = []
            for ele in prod:
                if ele[0] % 2:
                    work.append(((ele[0] - 1) // 2) + self._norb)
                else:
                    work.append(ele[0] // 2)
            self._matrix_data.append([operators.terms[prod], work])

        self._rank = len(self._matrix_data[0][1])

        ops = list(operators.get_operators())

        for oper in ops:
            coeff, phase, alpha_block, beta_block = \
                hamiltonian_utils.gather_nbody_spin_sectors(oper)

            alpha_out: List[Tuple[int, int]] = []
            beta_out: List[Tuple[int, int]] = []
            for alpha in alpha_block:
                alpha_out.append((alpha[0]//2, alpha[1]))
            for beta in beta_block:
                beta_out.append((beta[0]//2, beta[1]))
            self._operators.append((coeff*phase, alpha_out, beta_out))


    def dim(self) -> int:
        """Dim is the orbital dimension of the Hamiltonian arrays.
        """
        if self._conserve_spin:
            return self._norb

        return 2*self._norb


    def rank(self) -> int:
        """This returns the rank of the largest tensor.
        """
        return self._rank


    def nterms(self) -> int:
        """Return the number of non-zero elements in the Hamiltonian
        """
        return len(self._operators)


    def is_individual(self) -> bool:
        """Returns if this Hamiltonian consists of an individual operator plus its
        Hermitian conjugate
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


    def iht(self, time_step: float) -> 'SparseHamiltonian':
        """
        this returns SparseHamiltonian so it can be properly handled in Wavefunction propagation
        """
        out = copy.deepcopy(self)
        for index in range(len(out._operators)):
            (coeff, alpha, beta) = out._operators[index]
            out._operators[index] = (-coeff * 1.0j * time_step, alpha, beta)
        return out


    def terms(self):
        """Return the operators that comprise the SparseHamiltonian
        """
        return self._operators


    def terms_hamiltonian(self) -> List['SparseHamiltonian']:
        """ returns all of the terms as an array of SparseHamiltonian
        """
        out = []
        for current in self._operators:
            tmp = copy.deepcopy(self)
            tmp._operators = [current]
            out.append(tmp)
        return out
