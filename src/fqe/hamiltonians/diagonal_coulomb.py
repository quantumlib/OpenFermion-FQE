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

"""Diagonal Coulomb Hamiltonian Class
"""

from fqe.hamiltonians import hamiltonian

import numpy


class DiagonalCoulomb(hamiltonian.Hamiltonian):

    def __init__(self, h2e, conserve_number=True) -> None:
        super().__init__(conserve_number=conserve_number)
        diag = numpy.zeros(h2e.shape[0], dtype=h2e.dtype)
        self._dim = h2e.shape[0]
        self._tensor = {}

        if h2e.ndim == 2:
            self._tensor[1] = diag
            self._tensor[2] = h2e

        elif h2e.ndim == 4:
            for k in range(self._dim):
                diag[k] += h2e[k, k, k, k]
    
            vij = numpy.zeros((self._dim, self._dim), dtype=h2e.dtype)
            for i in range(self._dim):
                for j in range(self._dim):
                    vij[i, j] -= h2e[i, j, i, j] 

            self._tensor[1] = diag
            self._tensor[2] = vij


    def dim(self) -> int:
        """Return the dimension of the hamiltonian
        """
        return self._dim


    def diagonal_coulomb(self) -> bool:
        """
        """
        return True 


    def rank(self):
        """
        """
        return 4


    def tensor(self, rank):
        """
        """
        local_type = dtype=self._tensor[1].dtype
        if rank == 2:
            return numpy.zeros((self._dim, self._dim), dtype=local_type)

        h2e = numpy.zeros((self._dim, self._dim, self._dim, self._dim), dtype=local_type)
        for k in range(self._dim):
            h2e[k, k, k, k] += self._tensor[1][k]

        for i in range(self._dim):
            for j in range(self._dim):
                h2e[i, j, i, j] -= vij[i, j]
        return h2e


    def tensors(self):
        """
        """
        local_type = dtype=self._tensor[1].dtype
        h2e = numpy.zeros((self._dim, self._dim, self._dim, self._dim), dtype=local_type)

        for k in range(self._dim):
            h2e[k, k, k, k] += self._tensor[1][k]

        for i in range(self._dim):
            for j in range(self._dim):
                h2e[i, j, i, j] -= vij[i, j]

        return tuple([numpy.zeros((self._dim, self._dim), dtype=local_type), h2e])


    def iht(self, time):
        """
        """
        iht_mat = []
        for rank in range(len(self._tensor)):
            iht_mat.append(-1.j*time*self._tensor[rank + 1])

        return tuple(iht_mat)
