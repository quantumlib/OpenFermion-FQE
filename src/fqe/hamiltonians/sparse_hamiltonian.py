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

"""Sparse Hamiltonian Base Class
"""
import numpy
import copy
from numpy import linalg
from openfermion import FermionOperator
from typing import List

from fqe.hamiltonians import hamiltonian, hamiltonian_utils


class SparseHamiltonian(hamiltonian.Hamiltonian):


    def __init__(self, norb: int, operators: 'openfermion.FermionOperator', conserve_spin = True, conserve_number = True) -> None:
        """
        """
        super().__init__(conserve_number=conserve_number)

        self._norb = norb
        self._operators = []
        self._conserve_spin = conserve_spin 
        self._matrix_data = []

        for prod in operators.terms:
            self._matrix_data.append([operators.terms[prod],
                                      [((ele[0] - 1) // 2) + self._norb if ele[0] % 2 else ele[0] // 2 for ele in prod]])

        self._rank = len(self._matrix_data[0][1])
 
        ops = list(operators.get_operators())

        for oper in ops:
            coeff, phase, alpha_block, beta_block = hamiltonian_utils.gather_nbody_spin_sectors(oper)
            for alpha in alpha_block:
                alpha[0] = alpha[0]//2
            for beta in beta_block:
                beta[0] = beta[0]//2
            self._operators.append((coeff*phase, alpha_block, beta_block))


    def __repr__(self):
        return self.identity()


    def dim(self) -> int:
        return norb if self._conserve_spin else norb*2 


    def rank(self):
        """
        """
        return self._rank


    def nterms(self) -> int:
        return len(self._operators)


    def iht(self, time, full=True):
        """
        """
        iht_list = []

        for rank in range(2, self._rank + 1, 2):
            mat_dim = tuple([2*self._norb for _ in range(rank)])
            work = numpy.zeros(mat_dim, dtype=numpy.complex128)
            if rank == self._rank:
                work[tuple(self._matrix_data[0][1])] = self._matrix_data[0][0]
                work[tuple(self._matrix_data[1][1])] = self._matrix_data[1][0]
            iht_list.append(-1.j*time*work)

        return tuple(iht_list)


    def generated_unitary(self, time_step: float) -> 'SparseHamiltonian':
        """
        this returns SparseHamiltonian so it can be properly handled in Wavefunction propagation
        """
        out = copy.deepcopy(self)
        for index in range(len(out._operators)):
            (coeff, alpha, beta) = out._operators[index]
            out._operators[index] = (-coeff * 1.0j * time_step, alpha, beta)
        return out 


    def terms(self):
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
