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
""" Fermionic Quantum Emulator data class for holding wavefunction data.
"""
#Expanding out simple iterator indexes is unnecessary
#pylint: disable=invalid-name
#imports are ungrouped for type hinting
#pylint: disable=ungrouped-imports
#numpy.zeros_like initializer is not accepted
#pylint: disable=unsupported-assignment-operation
#pylint: disable=too-many-lines
#pylint: disable=too-many-locals
#pylint: disable=too-many-branches
#pylint: disable=too-many-arguments
import copy
import itertools
from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy
from scipy.special import binom

from fqe.bitstring import integer_index, get_bit, count_bits_above
from fqe.bitstring import set_bit, unset_bit, reverse_integer_index
from fqe.util import rand_wfn, validate_config
from fqe.fci_graph import FciGraph
from fqe.fci_graph_set import FciGraphSet

if TYPE_CHECKING:
    from numpy import ndarray as Nparray
    from fqe.fqe_data_set import FqeDataSet


class FqeData:
    """This is a basic data structure for use in the FQE.
    """

    def __init__(self,
                 nalpha: int,
                 nbeta: int,
                 norb: int,
                 fcigraph: Optional[FciGraph] = None,
                 dtype=numpy.complex128) -> None:
        """The FqeData structure holds the wavefunction for a particular
        configuration and provides an interace for accessing the data through
        the fcigraph functionality.

        Args:
            nalpha (int) - the number of alpha electrons
            nbeta (int) - the number of beta electrons
            norb (int) - the number of spatial orbitals
            fcigraph (optional, ...)
        """
        validate_config(nalpha, nbeta, norb)

        if not (fcigraph is None) and (nalpha != fcigraph.nalpha() or
                                       nbeta != fcigraph.nbeta() or
                                       norb != fcigraph.norb()):
            raise ValueError("FciGraph does not match other parameters")

        if fcigraph is None:
            self._core = FciGraph(nalpha, nbeta, norb)
        else:
            self._core = fcigraph
        self._dtype = dtype
        self._low_thresh = 0.3
        self._nele = self._core.nalpha() + self._core.nbeta()
        self._m_s = self._core.nalpha() - self._core.nbeta()
        self.coeff = numpy.zeros((self._core.lena(), self._core.lenb()),
                                 dtype=self._dtype)

    def __getitem__(self, key: Tuple[int, int]) -> complex:
        """Get an item from the fqe data structure by using the knowles-handy
        pointers.
        """
        return self.coeff[self._core.index_alpha(key[0]),
                          self._core.index_beta(key[1])]

    def __setitem__(self, key: Tuple[int, int], value: complex) -> None:
        """Set an element in the fqe data strucuture
        """
        self.coeff[self._core.index_alpha(key[0]),
                   self._core.index_beta(key[1])] = value

    def get_fcigraph(self) -> 'FciGraph':
        """
        Returns the underlying FciGraph object
        """
        return self._core

    def apply_diagonal_inplace(self, array: 'Nparray') -> None:
        """Iterate over each element and perform apply operation in place
        """
        beta_ptr = 0

        if array.size == 2 * self.norb():
            beta_ptr = self.norb()

        elif array.size != self.norb():
            raise ValueError('Non-diagonal array passed' \
                             ' into apply_diagonal_array')

        alpha = []
        for alp_cnf in range(self._core.lena()):
            diag_ele = 0.0
            for ind in integer_index(self._core.string_alpha(alp_cnf)):
                diag_ele += array[ind]
            alpha.append(diag_ele)

        beta = []
        for bet_cnf in range(self._core.lenb()):
            diag_ele = 0.0
            for ind in integer_index(self._core.string_beta(bet_cnf)):
                diag_ele += array[beta_ptr + ind]
            beta.append(diag_ele)

        for alp_cnf in range(self._core.lena()):
            for bet_cnf in range(self._core.lenb()):
                self.coeff[alp_cnf, bet_cnf] *= alpha[alp_cnf] + beta[bet_cnf]

    def evolve_diagonal(self, array: 'Nparray',
                        inplace: bool = False) -> 'Nparray':
        """Iterate over each element and return the exponential scaled
        contribution.
        """
        beta_ptr = 0

        if array.size == 2 * self.norb():
            beta_ptr = self.norb()

        elif array.size != self.norb():
            raise ValueError('Non-diagonal array passed' \
                             ' into apply_diagonal_array')

        if inplace:
            data = self.coeff
        else:
            data = numpy.copy(self.coeff).astype(numpy.complex128)

        for alp_cnf in range(self._core.lena()):
            diag_ele = 0.0
            for ind in integer_index(self._core.string_alpha(alp_cnf)):
                diag_ele += array[ind]

            if diag_ele != 0.0:
                data[alp_cnf, :] *= numpy.exp(diag_ele)

        for bet_cnf in range(self._core.lenb()):
            diag_ele = 0.0
            for ind in integer_index(self._core.string_beta(bet_cnf)):
                diag_ele += array[beta_ptr + ind]

            if diag_ele:
                data[:, bet_cnf] *= numpy.exp(diag_ele)

        return data

    def diagonal_coulomb(self,
                         diag: 'Nparray',
                         array: 'Nparray',
                         inplace: bool = False) -> 'Nparray':
        """Iterate over each element and return the scaled wavefunction.
        """
        if inplace:
            data = self.coeff
        else:
            data = numpy.copy(self.coeff)

        alpha_occ = []
        alpha_diag = []
        for alp_cnf in range(self.lena()):
            occ = integer_index(self._core.string_alpha(alp_cnf))
            alpha_occ.append(occ)
            diag_ele = 0.0
            for ind in occ:
                diag_ele += diag[ind]
                for jnd in occ:
                    diag_ele += array[ind, jnd]
            alpha_diag.append(diag_ele)

        beta_occ = []
        beta_diag = []
        for bet_cnf in range(self.lenb()):
            occ = integer_index(self._core.string_beta(bet_cnf))
            beta_occ.append(occ)
            diag_ele = 0.0
            for ind in occ:
                diag_ele += diag[ind]
                for jnd in occ:
                    diag_ele += array[ind, jnd]
            beta_diag.append(diag_ele)

        aarrays = numpy.empty((array.shape[1],), dtype=array.dtype)
        for alp_cnf in range(self.lena()):
            aarrays[:] = 0.0
            for ind in alpha_occ[alp_cnf]:
                aarrays[:] += array[ind, :]
            for bet_cnf in range(self.lenb()):
                diag_ele = 0.0
                for ind in beta_occ[bet_cnf]:
                    diag_ele += aarrays[ind]
                diag_ele = diag_ele * 2.0 + alpha_diag[alp_cnf] + beta_diag[
                    bet_cnf]
                data[alp_cnf, bet_cnf] *= numpy.exp(diag_ele)

        return data

    def apply(self, array: Tuple['Nparray']) -> 'Nparray':
        """
        API for application of dense operators (1- through 4-body operators) to
        the wavefunction self.
        """

        out = copy.deepcopy(self)
        out.apply_inplace(array)
        return out

    def apply_inplace(self, array: Tuple['Nparray', ...]) -> None:
        """
        API for application of dense operators (1- through 4-body operators) to
        the wavefunction self.
        """

        len_arr = len(array)
        assert 5 > len_arr > 0

        spatial = array[0].shape[0] == self.norb()
        if len_arr == 1:
            if spatial:
                self.coeff = self._apply_array_spatial1(array[0])
            else:
                self.coeff = self._apply_array_spin1(array[0])
        elif len_arr == 2:
            if spatial:
                self.coeff = self._apply_array_spatial12(array[0], array[1])
            else:
                self.coeff = self._apply_array_spin12(array[0], array[1])
        elif len_arr == 3:
            if spatial:
                self.coeff = self._apply_array_spatial123(
                    array[0], array[1], array[2])
            else:
                self.coeff = self._apply_array_spin123(array[0], array[1],
                                                       array[2])
        elif len_arr == 4:
            if spatial:
                self.coeff = self._apply_array_spatial1234(
                    array[0], array[1], array[2], array[3])
            else:
                self.coeff = self._apply_array_spin1234(array[0], array[1],
                                                        array[2], array[3])

    def _apply_array_spatial1(self, h1e: 'Nparray') -> 'Nparray':
        """
        API for application of 1-body spatial operators to the
        wavefunction self.  It returns array that corresponds to the
        output wave function data. If h1e only contains a single column,
        it goes to a special code path
        """
        assert h1e.shape == (self.norb(), self.norb())

        ncol = 0
        jorb = 0
        for j in range(self.norb()):
            if numpy.any(h1e[:, j]):
                ncol += 1
                jorb = j
            if ncol > 1:
                break

        if ncol > 1:
            dvec = self.calculate_dvec_spatial()
            out = numpy.einsum("ij,ijkl->kl", h1e, dvec)
        else:
            dvec = self.calculate_dvec_spatial_fixed_j(jorb)
            out = numpy.einsum("i,ikl->kl", h1e[:, jorb], dvec)

        return out

    def _apply_array_spin1(self, h1e: 'Nparray') -> 'Nparray':
        """
        API for application of 1-body spatial operators to the
        wavefunction self. It returns numpy.ndarray that corresponds to the
        output wave function data.
        """
        norb = self.norb()
        assert h1e.shape == (norb * 2, norb * 2)

        ncol = 0
        jorb = 0
        for j in range(self.norb() * 2):
            if numpy.any(h1e[:, j]):
                ncol += 1
                jorb = j
            if ncol > 1:
                break

        if ncol > 1:
            (dveca, dvecb) = self.calculate_dvec_spin()
            out = numpy.einsum("ij,ijkl->kl", h1e[:norb, :norb], dveca) \
                + numpy.einsum("ij,ijkl->kl", h1e[norb:, norb:], dvecb)
        else:
            dvec = self.calculate_dvec_spin_fixed_j(jorb)
            if jorb < norb:
                h1eview = h1e[:norb, jorb]
            else:
                h1eview = h1e[norb:, jorb]
            out = numpy.einsum("i,ikl->kl", h1eview, dvec)

        return out

    def _apply_array_spatial12(self, h1e: 'Nparray',
                               h2e: 'Nparray') -> 'Nparray':
        """
        API for application of 1- and 2-body spatial operators to the
        wavefunction self. It returns numpy.ndarray that corresponds to the
        output wave function data. Depending on the filling, it automatically
        chooses an efficient code.
        """
        norb = self.norb()
        assert h1e.shape == (norb, norb)
        assert h2e.shape == (norb, norb, norb, norb)
        nalpha = self.nalpha()
        nbeta = self.nbeta()

        thresh = self._low_thresh
        if nalpha < norb * thresh and nbeta < norb * thresh:
            graphset = FciGraphSet(2, 2)
            graphset.append(self._core)
            if nalpha - 2 >= 0:
                graphset.append(FciGraph(nalpha - 2, nbeta, norb))
            if nalpha - 1 >= 0 and nbeta - 1 >= 0:
                graphset.append(FciGraph(nalpha - 1, nbeta - 1, norb))
            if nbeta - 2 >= 0:
                graphset.append(FciGraph(nalpha, nbeta - 2, norb))
            return self._apply_array_spatial12_lowfilling(h1e, h2e)

        return self._apply_array_spatial12_halffilling(h1e, h2e)

    def _apply_array_spin12(self, h1e: 'Nparray', h2e: 'Nparray') -> 'Nparray':
        """
        API for application of 1- and 2-body spin-orbital operators to the
        wavefunction self.  It returns numpy.ndarray that corresponds to the
        output wave function data. Depending on the filling, it automatically
        chooses an efficient code.
        """
        norb = self.norb()
        assert h1e.shape == (norb * 2, norb * 2)
        assert h2e.shape == (norb * 2, norb * 2, norb * 2, norb * 2)
        nalpha = self.nalpha()
        nbeta = self.nbeta()

        thresh = self._low_thresh
        if nalpha < norb * thresh and nbeta < norb * thresh:
            graphset = FciGraphSet(2, 2)
            graphset.append(self._core)
            if nalpha - 2 >= 0:
                graphset.append(FciGraph(nalpha - 2, nbeta, norb))
            if nalpha - 1 >= 0 and nbeta - 1 >= 0:
                graphset.append(FciGraph(nalpha - 1, nbeta - 1, norb))
            if nbeta - 2 >= 0:
                graphset.append(FciGraph(nalpha, nbeta - 2, norb))
            return self._apply_array_spin12_lowfilling(h1e, h2e)

        return self._apply_array_spin12_halffilling(h1e, h2e)

    def _apply_array_spatial12_halffilling(self, h1e: 'Nparray',
                                           h2e: 'Nparray') -> 'Nparray':
        """
        Standard code to calculate application of 1- and 2-body spatial
        operators to the wavefunction self. It returns numpy.ndarray that
        corresponds to the output wave function data.
        """
        h1e = copy.deepcopy(h1e)
        h2e = numpy.moveaxis(copy.deepcopy(h2e), 1, 2) * (-1.0)
        norb = self.norb()
        for k in range(norb):
            h1e[:, :] -= h2e[:, k, k, :]

        if numpy.iscomplex(h1e).any() or numpy.iscomplex(h2e).any():
            dvec = self.calculate_dvec_spatial()
            out = numpy.einsum("ij,ijkl->kl", h1e, dvec)
            dvec = numpy.einsum("ijkl,klmn->ijmn", h2e, dvec)
            out += self._calculate_coeff_spatial_with_dvec(dvec)
        else:
            nij = norb * (norb + 1) // 2
            h1ec = numpy.zeros((nij), dtype=self._dtype)
            h2ec = numpy.zeros((nij, nij), dtype=self._dtype)
            for i in range(norb):
                for j in range(i + 1):
                    ijn = j + i * (i + 1) // 2
                    h1ec[ijn] = h1e[i, j]
                    for k in range(norb):
                        for l in range(k + 1):
                            kln = l + k * (k + 1) // 2
                            h2ec[ijn, kln] = h2e[i, j, k, l]
            dvec = self.calculate_dvec_spatial_compressed()
            out = numpy.einsum("i,ikl->kl", h1ec, dvec)
            dvec = numpy.einsum("ik,kmn->imn", h2ec, dvec)
            for i in range(self.norb()):
                for j in range(self.norb()):
                    ijn = min(i, j) + max(i, j) * (max(i, j) + 1) // 2
                    work = self._core.alpha_map(j, i)
                    for source, target, parity in work:
                        out[source, :] += dvec[ijn, target, :] * parity
                    work = self._core.beta_map(j, i)
                    for source, target, parity in work:
                        out[:, source] += dvec[ijn, :, target] * parity

        return out

    def _apply_array_spin12_halffilling(self, h1e: 'Nparray',
                                        h2e: 'Nparray') -> 'Nparray':
        """
        Standard code to calculate application of 1- and 2-body spin-orbital
        operators to the wavefunction self. It returns numpy.ndarray that
        corresponds to the output wave function data.
        """
        h1e = copy.deepcopy(h1e)
        h2e = numpy.moveaxis(copy.deepcopy(h2e), 1, 2) * (-1.0)
        norb = self.norb()
        for k in range(norb * 2):
            h1e[:, :] -= h2e[:, k, k, :]

        (dveca, dvecb) = self.calculate_dvec_spin()
        out = numpy.einsum("ij,ijkl->kl", h1e[:norb, :norb], dveca) \
            + numpy.einsum("ij,ijkl->kl", h1e[norb:, norb:], dvecb)
        ndveca = numpy.einsum("ijkl,klmn->ijmn",
                              h2e[:norb, :norb, :norb, :norb], dveca) \
               + numpy.einsum("ijkl,klmn->ijmn",
                              h2e[:norb, :norb, norb:, norb:], dvecb)
        ndvecb = numpy.einsum("ijkl,klmn->ijmn",
                              h2e[norb:, norb:, :norb, :norb], dveca) \
               + numpy.einsum("ijkl,klmn->ijmn",
                              h2e[norb:, norb:, norb:, norb:], dvecb)
        out += self.calculate_coeff_spin_with_dvec((ndveca, ndvecb))
        return out

    def _apply_array_spatial12_lowfilling(self, h1e: 'Nparray',
                                          h2e: 'Nparray') -> 'Nparray':
        """
        Low-filling specialization of the code to calculate application of
        1- and 2-body spatial operators to the wavefunction self.  It returns
        numpy.ndarray that corresponds to the output wave function data.
        """
        out = self._apply_array_spatial1(h1e)

        norb = self.norb()
        nalpha = self.nalpha()
        nbeta = self.nbeta()
        lena = self.lena()
        lenb = self.lenb()
        nlt = norb * (norb + 1) // 2

        h2ecomp = numpy.zeros((nlt, nlt), dtype=self._dtype)
        for i in range(norb):
            for j in range(i + 1, norb):
                ijn = i + j * (j + 1) // 2
                for k in range(norb):
                    for l in range(k + 1, norb):
                        h2ecomp[ijn, k + l * (l + 1) // 2] = (h2e[i, j, k, l] -
                                                              h2e[i, j, l, k] -
                                                              h2e[j, i, k, l] +
                                                              h2e[j, i, l, k])

        if nalpha - 2 >= 0:
            alpha_map, _ = self._core.find_mapping(-2, 0)
            intermediate = numpy.zeros(
                (nlt, int(binom(norb, nalpha - 2)), lenb), dtype=self._dtype)
            for i in range(norb):
                for j in range(i + 1, norb):
                    ijn = i + j * (j + 1) // 2
                    for source, target, parity in alpha_map[(i, j)]:
                        work = self.coeff[source, :] * parity
                        intermediate[ijn, target, :] += work

            intermediate = numpy.einsum('ij,jmn->imn', h2ecomp, intermediate)

            for i in range(norb):
                for j in range(i + 1, norb):
                    ijn = i + j * (j + 1) // 2
                    for source, target, parity in alpha_map[(i, j)]:
                        out[source, :] -= intermediate[ijn, target, :] * parity

        if self.nalpha() - 1 >= 0 and self.nbeta() - 1 >= 0:
            alpha_map, beta_map = self._core.find_mapping(-1, -1)
            intermediate = numpy.zeros((norb, norb, int(binom(
                norb, nalpha - 1)), int(binom(norb, nbeta - 1))),
                                       dtype=self._dtype)

            for i in range(norb):
                for j in range(norb):
                    for sourcea, targeta, paritya in alpha_map[(i,)]:
                        sign = ((-1)**(nalpha - 1)) * paritya
                        for sourceb, targetb, parityb in beta_map[(j,)]:
                            work = self.coeff[sourcea, sourceb] * sign * parityb
                            intermediate[i, j, targeta, targetb] += 2 * work

            intermediate = numpy.einsum('ijkl,klmn->ijmn', h2e, intermediate)

            for i in range(norb):
                for j in range(norb):
                    for sourcea, targeta, paritya in alpha_map[(i,)]:
                        sign = ((-1)**nalpha) * paritya
                        for sourceb, targetb, parityb in beta_map[(j,)]:
                            work = intermediate[i, j, targeta, targetb] * sign
                            out[sourcea, sourceb] += work * parityb

        if self.nbeta() - 2 >= 0:
            _, beta_map = self._core.find_mapping(0, -2)
            intermediate = numpy.zeros((nlt, lena, int(binom(norb, nbeta - 2))),
                                       dtype=self._dtype)
            for i in range(norb):
                for j in range(i + 1, norb):
                    ijn = i + j * (j + 1) // 2
                    for source, target, parity in beta_map[(i, j)]:
                        work = self.coeff[:, source] * parity
                        intermediate[ijn, :, target] += work

            intermediate = numpy.einsum('ij,jmn->imn', h2ecomp, intermediate)

            for i in range(norb):
                for j in range(i + 1, norb):
                    ijn = i + j * (j + 1) // 2
                    for source, target, sign in beta_map[(min(i, j), max(i,
                                                                         j))]:
                        out[:, source] -= intermediate[ijn, :, target] * sign
        return out

    def _apply_array_spin12_lowfilling(self, h1e: 'Nparray',
                                       h2e: 'Nparray') -> 'Nparray':
        """
        Low-filling specialization of the code to calculate application of
        1- and 2-body spin-orbital operators to the wavefunction self. It
        returns numpy.ndarray that corresponds to the output wave function data.
        """
        out = self._apply_array_spin1(h1e)

        norb = self.norb()
        nalpha = self.nalpha()
        nbeta = self.nbeta()
        lena = self.lena()
        lenb = self.lenb()
        nlt = norb * (norb + 1) // 2

        h2ecompa = numpy.zeros((nlt, nlt), dtype=self._dtype)
        h2ecompb = numpy.zeros((nlt, nlt), dtype=self._dtype)
        for i in range(norb):
            for j in range(i + 1, norb):
                ijn = i + j * (j + 1) // 2
                for k in range(norb):
                    for l in range(k + 1, norb):
                        kln = k + l * (l + 1) // 2
                        h2ecompa[ijn, kln] = (h2e[i, j, k, l] -
                                              h2e[i, j, l, k] -
                                              h2e[j, i, k, l] + h2e[j, i, l, k])
                        ino = i + norb
                        jno = j + norb
                        kno = k + norb
                        lno = l + norb
                        h2ecompb[ijn, kln] = (h2e[ino, jno, kno, lno] -
                                              h2e[ino, jno, lno, kno] -
                                              h2e[jno, ino, kno, lno] +
                                              h2e[jno, ino, lno, kno])

        if nalpha - 2 >= 0:
            alpha_map, _ = self._core.find_mapping(-2, 0)
            intermediate = numpy.zeros(
                (nlt, int(binom(norb, nalpha - 2)), lenb), dtype=self._dtype)
            for i in range(norb):
                for j in range(i + 1, norb):
                    ijn = i + j * (j + 1) // 2
                    for source, target, parity in alpha_map[(i, j)]:
                        work = self.coeff[source, :] * parity
                        intermediate[ijn, target, :] += work

            intermediate = numpy.einsum('ij,jmn->imn', h2ecompa, intermediate)

            for i in range(norb):
                for j in range(i + 1, norb):
                    ijn = i + j * (j + 1) // 2
                    for source, target, parity in alpha_map[(i, j)]:
                        out[source, :] -= intermediate[ijn, target, :] * parity

        if self.nalpha() - 1 >= 0 and self.nbeta() - 1 >= 0:
            alpha_map, beta_map = self._core.find_mapping(-1, -1)
            intermediate = numpy.zeros((norb, norb, int(binom(
                norb, nalpha - 1)), int(binom(norb, nbeta - 1))),
                                       dtype=self._dtype)

            for i in range(norb):
                for j in range(norb):
                    for sourcea, targeta, paritya in alpha_map[(i,)]:
                        sign = ((-1)**(nalpha - 1)) * paritya
                        for sourceb, targetb, parityb in beta_map[(j,)]:
                            work = self.coeff[sourcea, sourceb] * sign * parityb
                            intermediate[i, j, targeta, targetb] += 2 * work

            intermediate = numpy.einsum('ijkl,klmn->ijmn',
                                        h2e[:norb, norb:, :norb, norb:],
                                        intermediate)

            for i in range(norb):
                for j in range(norb):
                    for sourcea, targeta, paritya in alpha_map[(i,)]:
                        paritya *= (-1)**nalpha
                        for sourceb, targetb, parityb in beta_map[(j,)]:
                            work = intermediate[i, j, targeta, targetb]
                            out[sourcea, sourceb] += work * paritya * parityb

        if self.nbeta() - 2 >= 0:
            _, beta_map = self._core.find_mapping(0, -2)
            intermediate = numpy.zeros((nlt, lena, int(binom(norb, nbeta - 2))),
                                       dtype=self._dtype)
            for i in range(norb):
                for j in range(i + 1, norb):
                    ijn = i + j * (j + 1) // 2
                    for source, target, parity in beta_map[(i, j)]:
                        work = self.coeff[:, source] * parity
                        intermediate[ijn, :, target] += work

            intermediate = numpy.einsum('ij,jmn->imn', h2ecompb, intermediate)

            for i in range(norb):
                for j in range(i + 1, norb):
                    ijn = i + j * (j + 1) // 2
                    for source, target, sign in beta_map[(min(i, j), max(i,
                                                                         j))]:
                        out[:, source] -= intermediate[ijn, :, target] * sign
        return out

    def _apply_array_spatial123(self, h1e: 'Nparray', h2e: 'Nparray', h3e: 'Nparray', \
                                dvec: 'Nparray' = None, evec: 'Nparray' = None) -> 'Nparray':
        """
        Code to calculate application of 1- through 3-body spatial operators to
        the wavefunction self. It returns numpy.ndarray that corresponds to the
        output wave function data.
        """
        norb = self.norb()
        assert h3e.shape == (norb, norb, norb, norb, norb, norb)
        assert not (dvec is None) ^ (evec is None)

        lena = self.lena()
        lenb = self.lenb()

        nh1e = numpy.copy(h1e)
        nh2e = numpy.copy(h2e)

        for i in range(norb):
            for j in range(norb):
                for k in range(norb):
                    nh2e[j, k, :, :] += (-h3e[k, j, i, i, :, :] -
                                         h3e[j, i, k, i, :, :] -
                                         h3e[j, k, i, :, i, :])
                nh1e[:, :] += h3e[:, i, j, i, j, :]

        out = self._apply_array_spatial12_halffilling(nh1e, nh2e)

        if dvec is None:
            dvec = self.calculate_dvec_spatial()
        if evec is None:
            evec = numpy.zeros((norb, norb, norb, norb, lena, lenb),
                               dtype=self._dtype)
            for i in range(norb):
                for j in range(norb):
                    tmp = dvec[i, j, :, :]
                    tmp2 = self._calculate_dvec_spatial_with_coeff(tmp)
                    evec[:, :, i, j, :, :] = tmp2[:, :, :, :]

        dvec = numpy.einsum('ikmjln,klmnxy->ijxy', h3e, evec)

        out -= self._calculate_coeff_spatial_with_dvec(dvec)
        return out

    def _apply_array_spin123(self,
                             h1e: 'Nparray',
                             h2e: 'Nparray',
                             h3e: 'Nparray',
                             dvec: Optional[Tuple['Nparray', 'Nparray']] = None,
                             evec: Optional[Tuple['Nparray', 'Nparray', 'Nparray', 'Nparray']] \
                                   = None) -> 'Nparray':
        """
        Code to calculate application of 1- through 3-body spin-orbital
        operators to the wavefunction self. It returns numpy.ndarray that
        corresponds to the output wave function data.
        """
        norb = self.norb()
        assert h3e.shape == (norb * 2, norb * 2, norb * 2, norb * 2, norb * 2,
                             norb * 2)
        assert not (dvec is None) ^ (evec is None)

        from1234 = (dvec is not None) and (evec is not None)

        lena = self.lena()
        lenb = self.lenb()

        nh1e = numpy.copy(h1e)
        nh2e = numpy.copy(h2e)

        for i in range(norb * 2):
            for j in range(norb * 2):
                for k in range(norb * 2):
                    nh2e[j, k, :, :] += (-h3e[k, j, i, i, :, :] -
                                         h3e[j, i, k, i, :, :] -
                                         h3e[j, k, i, :, i, :])

                nh1e[:, :] += h3e[:, i, j, i, j, :]

        out = self._apply_array_spin12_halffilling(nh1e, nh2e)

        if not from1234:
            (dveca, dvecb) = self.calculate_dvec_spin()
        else:
            dveca, dvecb = dvec  # type: ignore

        if not from1234:
            evecaa = numpy.zeros((norb, norb, norb, norb, lena, lenb),
                                 dtype=self._dtype)
            evecab = numpy.zeros((norb, norb, norb, norb, lena, lenb),
                                 dtype=self._dtype)
            evecba = numpy.zeros((norb, norb, norb, norb, lena, lenb),
                                 dtype=self._dtype)
            evecbb = numpy.zeros((norb, norb, norb, norb, lena, lenb),
                                 dtype=self._dtype)

            for i in range(norb):
                for j in range(norb):
                    tmp = self._calculate_dvec_spin_with_coeff(
                        dveca[i, j, :, :])
                    evecaa[:, :, i, j, :, :] = tmp[0][:, :, :, :]

                    tmp = self._calculate_dvec_spin_with_coeff(
                        dvecb[i, j, :, :])
                    evecab[:, :, i, j, :, :] = tmp[0][:, :, :, :]
                    evecbb[:, :, i, j, :, :] = tmp[1][:, :, :, :]
        else:
            evecaa, evecab, evecba, evecbb = evec  # type: ignore

        symfac = 2.0 if not from1234 else 1.0

        dveca = numpy.einsum('ikmjln,klmnxy->ijxy',
                             h3e[:norb, :norb, :norb, :norb, :norb, :norb],
                             evecaa) \
              + numpy.einsum('ikmjln,klmnxy->ijxy',
                             h3e[:norb, :norb, norb:, :norb, :norb, norb:],
                             evecab) * symfac \
              + numpy.einsum('ikmjln,klmnxy->ijxy',
                             h3e[:norb, norb:, norb:, :norb, norb:, norb:],
                             evecbb)

        dvecb = numpy.einsum('ikmjln,klmnxy->ijxy',
                             h3e[norb:, :norb, :norb, norb:, :norb, :norb],
                             evecaa) \
              + numpy.einsum('ikmjln,klmnxy->ijxy',
                             h3e[norb:, :norb, norb:, norb:, :norb, norb:],
                             evecab) * symfac \
              + numpy.einsum('ikmjln,klmnxy->ijxy',
                             h3e[norb:, norb:, norb:, norb:, norb:, norb:],
                             evecbb)

        if from1234:
            dveca += numpy.einsum('ikmjln,klmnxy->ijxy',
                                  h3e[:norb, norb:, :norb, :norb, norb:, :norb],
                                  evecba)
            dvecb += numpy.einsum('ikmjln,klmnxy->ijxy',
                                  h3e[norb:, norb:, :norb, norb:, norb:, :norb],
                                  evecba)

        out -= self.calculate_coeff_spin_with_dvec((dveca, dvecb))
        return out

    def _apply_array_spatial1234(self, h1e: 'Nparray', h2e: 'Nparray',
                                 h3e: 'Nparray', h4e: 'Nparray') -> 'Nparray':
        """
        Code to calculate application of 1- through 4-body spatial operators to
        the wavefunction self.  It returns numpy.ndarray that corresponds to the
        output wave function data.
        """
        norb = self.norb()
        assert h4e.shape == (norb, norb, norb, norb, norb, norb, norb, norb)
        lena = self.lena()
        lenb = self.lenb()

        nh1e = numpy.copy(h1e)
        nh2e = numpy.copy(h2e)
        nh3e = numpy.copy(h3e)

        for i in range(norb):
            for j in range(norb):
                for k in range(norb):
                    nh1e[:, :] -= h4e[:, j, i, k, j, i, k, :]
                    for l in range(norb):
                        nh2e[i, j, :, :] += (h4e[j, l, i, k, l, k, :, :] +
                                             h4e[i, j, l, k, l, k, :, :] +
                                             h4e[i, l, k, j, l, k, :, :] +
                                             h4e[j, i, k, l, l, k, :, :] +
                                             h4e[i, k, j, l, k, :, l, :] +
                                             h4e[j, i, k, l, k, :, l, :] +
                                             h4e[i, j, k, l, :, k, l, :])
                        nh3e[i, j, k, :, :, :] += (h4e[k, i, j, l, l, :, :, :] +
                                                   h4e[j, i, l, k, l, :, :, :] +
                                                   h4e[i, l, j, k, l, :, :, :] +
                                                   h4e[i, k, j, l, :, l, :, :] +
                                                   h4e[i, j, l, k, :, l, :, :] +
                                                   h4e[i, j, k, l, :, :, l, :])

        dvec = self.calculate_dvec_spatial()
        evec = numpy.zeros((norb, norb, norb, norb, lena, lenb),
                           dtype=self._dtype)

        for i in range(norb):
            for j in range(norb):
                tmp = dvec[i, j, :, :]
                tmp2 = self._calculate_dvec_spatial_with_coeff(tmp)
                evec[:, :, i, j, :, :] = tmp2[:, :, :, :]

        out = self._apply_array_spatial123(nh1e, nh2e, nh3e, dvec, evec)

        evec = numpy.einsum('ikmojlnp,mnopxy->ijklxy', h4e, evec)

        dvec2 = numpy.zeros(dvec.shape, dtype=self._dtype)
        for i in range(norb):
            for j in range(norb):
                dvec[:, :, :, :] = evec[i, j, :, :, :, :]
                cvec = self._calculate_coeff_spatial_with_dvec(dvec)
                dvec2[i, j, :, :] += cvec[:, :]

        out += self._calculate_coeff_spatial_with_dvec(dvec2)
        return out

    def _apply_array_spin1234(self, h1e: 'Nparray', h2e: 'Nparray',
                              h3e: 'Nparray', h4e: 'Nparray') -> 'Nparray':
        """
        Code to calculate application of 1- through 4-body spin-orbital
        operators to the wavefunction self. It returns numpy.ndarray that
        corresponds to the output wave function data.
        """
        norb = self.norb()
        tno = 2 * norb
        assert h4e.shape == (tno, tno, tno, tno, tno, tno, tno, tno)
        lena = self.lena()
        lenb = self.lenb()

        nh1e = numpy.copy(h1e)
        nh2e = numpy.copy(h2e)
        nh3e = numpy.copy(h3e)

        for i in range(norb * 2):
            for j in range(norb * 2):
                for k in range(norb * 2):
                    nh1e[:, :] -= h4e[:, j, i, k, j, i, k, :]
                    for l in range(norb * 2):
                        nh2e[i, j, :, :] += (h4e[j, l, i, k, l, k, :, :] +
                                             h4e[i, j, l, k, l, k, :, :] +
                                             h4e[i, l, k, j, l, k, :, :] +
                                             h4e[j, i, k, l, l, k, :, :] +
                                             h4e[i, k, j, l, k, :, l, :] +
                                             h4e[j, i, k, l, k, :, l, :] +
                                             h4e[i, j, k, l, :, k, l, :])
                        nh3e[i, j, k, :, :, :] += (h4e[k, i, j, l, l, :, :, :] +
                                                   h4e[j, i, l, k, l, :, :, :] +
                                                   h4e[i, l, j, k, l, :, :, :] +
                                                   h4e[i, k, j, l, :, l, :, :] +
                                                   h4e[i, j, l, k, :, l, :, :] +
                                                   h4e[i, j, k, l, :, :, l, :])

        (dveca, dvecb) = self.calculate_dvec_spin()
        evecaa = numpy.zeros((norb, norb, norb, norb, lena, lenb),
                             dtype=self._dtype)
        evecab = numpy.zeros((norb, norb, norb, norb, lena, lenb),
                             dtype=self._dtype)
        evecba = numpy.zeros((norb, norb, norb, norb, lena, lenb),
                             dtype=self._dtype)
        evecbb = numpy.zeros((norb, norb, norb, norb, lena, lenb),
                             dtype=self._dtype)
        for i in range(norb):
            for j in range(norb):
                tmp = self._calculate_dvec_spin_with_coeff(dveca[i, j, :, :])
                evecaa[:, :, i, j, :, :] = tmp[0][:, :, :, :]
                evecba[:, :, i, j, :, :] = tmp[1][:, :, :, :]

                tmp = self._calculate_dvec_spin_with_coeff(dvecb[i, j, :, :])
                evecab[:, :, i, j, :, :] = tmp[0][:, :, :, :]
                evecbb[:, :, i, j, :, :] = tmp[1][:, :, :, :]

        out = self._apply_array_spin123(nh1e, nh2e, nh3e, (dveca, dvecb),
                                        (evecaa, evecab, evecba, evecbb))

        estr = 'ikmojlnp,mnopxy->ijklxy'
        nevecaa = numpy.einsum(estr, h4e[:norb, :norb, :norb, :norb, \
                                         :norb, :norb, :norb, :norb], evecaa) \
                + 2.0 * numpy.einsum(estr, h4e[:norb, :norb, :norb, norb:, \
                                         :norb, :norb, :norb, norb:], evecab) \
                + numpy.einsum(estr, h4e[:norb, :norb, norb:, norb:, \
                                         :norb, :norb, norb:, norb:], evecbb)
        nevecab = numpy.einsum(estr, h4e[:norb, norb:, :norb, :norb, \
                                         :norb, norb:, :norb, :norb], evecaa) \
                + 2.0 * numpy.einsum(estr, h4e[:norb, norb:, :norb, norb:, \
                                         :norb, norb:, :norb, norb:], evecab) \
                + numpy.einsum(estr, h4e[:norb, norb:, norb:, norb:, \
                                         :norb, norb:, norb:, norb:], evecbb)
        nevecbb = numpy.einsum(estr, h4e[norb:, norb:, :norb, :norb, \
                                         norb:, norb:, :norb, :norb], evecaa) \
                + 2.0 * numpy.einsum(estr, h4e[norb:, norb:, :norb, norb:, \
                                         norb:, norb:, :norb, norb:], evecab) \
                + numpy.einsum(estr, h4e[norb:, norb:, norb:, norb:, \
                                         norb:, norb:, norb:, norb:], evecbb)

        dveca2 = numpy.zeros(dveca.shape, dtype=self._dtype)
        dvecb2 = numpy.zeros(dvecb.shape, dtype=self._dtype)
        for i in range(norb):
            for j in range(norb):
                dveca[:, :, :, :] = nevecaa[i, j, :, :, :, :]
                dvecb[:, :, :, :] = nevecab[i, j, :, :, :, :]
                cvec = self.calculate_coeff_spin_with_dvec((dveca, dvecb))
                dveca2[i, j, :, :] += cvec[:, :]

                dveca[:, :, :, :] = nevecab[:, :, i, j, :, :]
                dvecb[:, :, :, :] = nevecbb[i, j, :, :, :, :]
                cvec = self.calculate_coeff_spin_with_dvec((dveca, dvecb))
                dvecb2[i, j, :, :] += cvec[:, :]

        out += self.calculate_coeff_spin_with_dvec((dveca2, dvecb2))
        return out

    def apply_inplace_s2(self) -> None:
        """
        Apply the S squared operator to self.
        """
        norb = self.norb()
        orig = numpy.copy(self.coeff)
        s_z = (self.nalpha() - self.nbeta()) * 0.5
        self.coeff *= s_z + s_z * s_z + self.nbeta()

        if self.nalpha() != self.norb() and self.nbeta() != 0:
            dvec = numpy.zeros((norb, norb, self.lena(), self.lenb()),
                               dtype=self._dtype)
            for i in range(norb):
                for j in range(norb):
                    for source, target, parity in self.alpha_map(i, j):
                        dvec[i, j, target, :] += orig[source, :] * parity
            for i in range(self.norb()):
                for j in range(self.norb()):
                    for source, target, parity in self.beta_map(j, i):
                        self.coeff[:, source] -= dvec[j, i, :, target] * parity

    def apply_individual_nbody(self, coeff: complex, daga: List[int],
                               undaga: List[int], dagb: List[int],
                               undagb: List[int]) -> 'FqeData':
        """
        Apply function with an individual operator represented in arrays.
        It is assumed that the operator is spin conserving
        """
        assert len(daga) == len(undaga) and len(dagb) == len(undagb)

        alphamap = []
        betamap = []

        def make_mapping_each(alpha: bool) -> None:
            (dag, undag) = (daga, undaga) if alpha else (dagb, undagb)
            for index in range(self.lena() if alpha else self.lenb()):
                if alpha:
                    current = self._core.string_alpha(index)
                else:
                    current = self._core.string_beta(index)

                check = True
                for i in undag:
                    if not check:
                        break
                    check &= bool(get_bit(current, i))
                for i in dag:
                    if not check:
                        break
                    check &= i in undag or not bool(get_bit(current, i))
                if check:
                    parity = 0
                    for i in reversed(undag):
                        parity += count_bits_above(current, i)
                        current = unset_bit(current, i)
                    for i in reversed(dag):
                        parity += count_bits_above(current, i)
                        current = set_bit(current, i)
                    if alpha:
                        alphamap.append((index, self._core.index_alpha(current),
                                         (-1)**parity))
                    else:
                        betamap.append((index, self._core.index_beta(current),
                                        (-1)**parity))

        make_mapping_each(True)
        make_mapping_each(False)
        out = copy.deepcopy(self)
        out.coeff.fill(0.0)
        sourceb_vec = numpy.array([xx[0] for xx in betamap])
        targetb_vec = numpy.array([xx[1] for xx in betamap])
        parityb_vec = numpy.array([xx[2] for xx in betamap])

        if len(alphamap) == 0 or len(betamap) == 0:
            return out
        else:
            for sourcea, targeta, paritya in alphamap:
                out.coeff[targeta, targetb_vec] = \
                    coeff * paritya * numpy.multiply(
                        self.coeff[sourcea, sourceb_vec], parityb_vec)
            # # TODO: THIS SHOULD BE CHECKED THOROUGHLY
            # # NOTE: Apparently the meshgrid construction overhead
            # # slows down this line so it is a little slower than the previous
            # sourcea_vec = numpy.array([xx[0] for xx in alphamap])
            # targeta_vec = numpy.array([xx[1] for xx in alphamap])
            # paritya_vec = numpy.array([xx[2] for xx in alphamap])
            # target_xi, target_yj = numpy.meshgrid(targeta_vec, targetb_vec)
            # source_xi, source_yj = numpy.meshgrid(sourcea_vec, sourceb_vec)
            # parity_xi, parity_yj = numpy.meshgrid(paritya_vec, parityb_vec)
            # out.coeff[target_xi, target_yj] = coeff * \
            #         (self.coeff[source_xi, source_yj] * parity_xi * parity_yj)

            return out

    def rdm1(self, bradata: Optional['FqeData'] = None) -> 'Nparray':
        """
        API for calculating 1-particle RDMs given a wave function. When bradata
        is given, it calculates transition RDMs. Depending on the filling, the
        code selects an optimal algorithm.
        """
        if bradata is not None:
            dvec2 = bradata.calculate_dvec_spatial()
        else:
            dvec2 = self.calculate_dvec_spatial()
        return (numpy.einsum('jikl,kl->ij', dvec2.conj(), self.coeff),)

    def rdm12(self, bradata: Optional['FqeData'] = None) -> numpy.ndarray:
        """
        API for calculating 1- and 2-particle RDMs given a wave function.
        When bradata is given, it calculates transition RDMs. Depending on the
        filling, the code selects an optimal algorithm.
        """
        norb = self.norb()
        nalpha = self.nalpha()
        nbeta = self.nbeta()

        thresh = self._low_thresh
        if nalpha < norb * thresh and nbeta < norb * thresh:
            graphset = FciGraphSet(2, 2)
            graphset.append(self._core)
            if nalpha - 2 >= 0:
                graphset.append(FciGraph(nalpha - 2, nbeta, norb))
            if nalpha - 1 >= 0 and nbeta - 1 >= 0:
                graphset.append(FciGraph(nalpha - 1, nbeta - 1, norb))
            if nbeta - 2 >= 0:
                graphset.append(FciGraph(nalpha, nbeta - 2, norb))
            return self._rdm12_lowfilling(bradata)

        return self._rdm12_halffilling(bradata)

    def _rdm12_halffilling(self, bradata: Optional['FqeData'] = None
                          ) -> numpy.ndarray:
        """
        Standard code for calculating 1- and 2-particle RDMs given a
        wavefunction. When bradata is given, it calculates transition RDMs.
        """
        dvec = self.calculate_dvec_spatial()
        dvec2 = dvec if bradata is None else bradata.calculate_dvec_spatial()
        out1 = numpy.einsum('jikl,kl->ij', dvec2, self.coeff)
        out2 = numpy.einsum('jikl,mnkl->imjn', dvec2.conj(), dvec) * (-1.0)
        for i in range(self.norb()):
            out2[:, i, i, :] += out1[:, :]
        return out1, out2

    def _rdm12_lowfilling(self,
                          bradata: Optional['FqeData'] = None) -> numpy.ndarray:
        """
        Low-filling specialization of the code for Calculating 1- and 2-particle
        RDMs given a wave function. When bradata is given, it calculates
        transition RDMs.
        """
        norb = self.norb()
        nalpha = self.nalpha()
        nbeta = self.nbeta()
        lena = self.lena()
        lenb = self.lenb()
        nlt = norb * (norb + 1) // 2

        outpack = numpy.zeros((nlt, nlt), dtype=self.coeff.dtype)
        outunpack = numpy.zeros((norb, norb, norb, norb),
                                dtype=self.coeff.dtype)
        if nalpha - 2 >= 0:
            alpha_map, _ = self._core.find_mapping(-2, 0)

            def compute_intermediate0(coeff):
                tmp = numpy.zeros((nlt, int(binom(norb, nalpha - 2)), lenb),
                                  dtype=self.coeff.dtype)
                for i in range(norb):
                    for j in range(i + 1, norb):
                        for source, target, parity in alpha_map[(i, j)]:
                            tmp[i + j * (j + 1) //
                                2, target, :] += coeff[source, :] * parity
                return tmp

            inter = compute_intermediate0(self.coeff)
            inter2 = inter if bradata is None else compute_intermediate0(
                bradata.coeff)
            outpack += numpy.einsum('imn,kmn->ik', inter2.conj(), inter)

        if self.nalpha() - 1 >= 0 and self.nbeta() - 1 >= 0:
            alpha_map, beta_map = self._core.find_mapping(-1, -1)

            def compute_intermediate1(coeff):
                tmp = numpy.zeros((norb, norb, int(binom(
                    norb, nalpha - 1)), int(binom(norb, nbeta - 1))),
                                  dtype=self.coeff.dtype)
                for i in range(norb):
                    for j in range(norb):
                        for sourcea, targeta, paritya in alpha_map[(i,)]:
                            paritya *= (-1)**(nalpha - 1)
                            for sourceb, targetb, parityb in beta_map[(j,)]:
                                work = coeff[sourcea,
                                             sourceb] * paritya * parityb
                                tmp[i, j, targeta, targetb] += work
                return tmp

            inter = compute_intermediate1(self.coeff)
            inter2 = inter if bradata is None else compute_intermediate1(
                bradata.coeff)
            outunpack += numpy.einsum('ijmn,klmn->ijkl', inter2.conj(), inter)

        if self.nbeta() - 2 >= 0:
            _, beta_map = self._core.find_mapping(0, -2)

            def compute_intermediate2(coeff):
                tmp = numpy.zeros((nlt, lena, int(binom(norb, nbeta - 2))),
                                  dtype=self.coeff.dtype)
                for i in range(norb):
                    for j in range(i + 1, norb):
                        for source, target, parity in beta_map[(i, j)]:
                            tmp[i + j * (j + 1) //
                                2, :, target] += coeff[:, source] * parity

                return tmp

            inter = compute_intermediate2(self.coeff)
            inter2 = inter if bradata is None else compute_intermediate2(
                bradata.coeff)
            outpack += numpy.einsum('imn,kmn->ik', inter2.conj(), inter)

        out = numpy.zeros_like(outunpack)
        for i in range(norb):
            for j in range(norb):
                ij = min(i, j) + max(i, j) * (max(i, j) + 1) // 2
                parityij = 1.0 if i < j else -1.0
                for k in range(norb):
                    for l in range(norb):
                        parity = parityij * (1.0 if k < l else -1.0)
                        out[i, j, k,
                            l] -= outunpack[i, j, k, l] + outunpack[j, i, l, k]
                        mnkl, mxkl = min(k, l), max(k, l)
                        work = outpack[ij, mnkl + mxkl * (mxkl + 1) // 2]
                        out[i, j, k, l] -= work * parity

        return self.rdm1(bradata)[0], out

    def rdm123(self,
               bradata: Optional['FqeData'] = None,
               dvec: 'Nparray' = None,
               dvec2: 'Nparray' = None,
               evec2: 'Nparray' = None) -> 'Nparray':
        """
        Calculates 1- through 3-particle RDMs given a wave function. When
        bradata is given, it calculates transition RDMs.
        """
        norb = self.norb()
        if dvec is None:
            dvec = self.calculate_dvec_spatial()
        if dvec2 is None:
            if bradata is None:
                dvec2 = dvec
            else:
                dvec2 = bradata.calculate_dvec_spatial()
        out1 = numpy.einsum('jikl,kl->ij', dvec2.conj(), self.coeff)
        out2 = numpy.einsum('jikl,mnkl->imjn', dvec2.conj(), dvec) * (-1.0)
        for i in range(norb):
            out2[:, i, i, :] += out1[:, :]

        def make_evec(current_dvec: 'Nparray') -> 'Nparray':
            current_evec = numpy.zeros(
                (norb, norb, norb, norb, self.lena(), self.lenb()),
                dtype=self._dtype)
            for i in range(norb):
                for j in range(norb):
                    tmp = current_dvec[i, j, :, :]
                    tmp2 = self._calculate_dvec_spatial_with_coeff(tmp)
                    current_evec[:, :, i, j, :, :] = tmp2[:, :, :, :]
            return current_evec

        if evec2 is None:
            evec2 = make_evec(dvec2)

        out3 = numpy.einsum('lkjimn,opmn->ikojlp', evec2.conj(), dvec) * (-1.0)
        for i in range(norb):
            out3[:, i, :, i, :, :] -= out2[:, :, :, :]
            out3[:, :, i, :, i, :] -= out2[:, :, :, :]
            for j in range(norb):
                out3[:, i, j, i, j, :] += out1[:, :]
                for k in range(norb):
                    out3[j, k, i, i, :, :] -= out2[k, j, :, :]
        return (out1, out2, out3)

    def rdm1234(self, bradata: Optional['FqeData'] = None) -> 'Nparray':
        """
        Calculates 1- through 4-particle RDMs given a wave function. When
        bradata is given, it calculates transition RDMs.
        """
        norb = self.norb()
        dvec = self.calculate_dvec_spatial()
        dvec2 = dvec if bradata is None else bradata.calculate_dvec_spatial()

        def make_evec(current_dvec: 'Nparray') -> 'Nparray':
            current_evec = numpy.zeros(
                (norb, norb, norb, norb, self.lena(), self.lenb()),
                dtype=self._dtype)
            for i in range(norb):
                for j in range(norb):
                    tmp = current_dvec[i, j, :, :]
                    tmp2 = self._calculate_dvec_spatial_with_coeff(tmp)
                    current_evec[:, :, i, j, :, :] = tmp2[:, :, :, :]
            return current_evec

        evec = make_evec(dvec)
        evec2 = evec if bradata is None else make_evec(dvec2)

        (out1, out2, out3) = self.rdm123(bradata, dvec, dvec2, evec2)

        out4 = numpy.einsum('lkjimn,opxymn->ikoxjlpy', evec2.conj(), evec)
        for i in range(norb):
            for j in range(norb):
                for k in range(norb):
                    out4[:, j, i, k, j, i, k, :] -= out1[:, :]
                    for l in range(norb):
                        out4[j, l, i, k, l, k, :, :] += out2[i, j, :, :]
                        out4[i, j, l, k, l, k, :, :] += out2[i, j, :, :]
                        out4[i, l, k, j, l, k, :, :] += out2[i, j, :, :]
                        out4[j, i, k, l, l, k, :, :] += out2[i, j, :, :]
                        out4[i, k, j, l, k, :, l, :] += out2[i, j, :, :]
                        out4[j, i, k, l, k, :, l, :] += out2[i, j, :, :]
                        out4[i, j, k, l, :, k, l, :] += out2[i, j, :, :]
                        out4[k, i, j, l, l, :, :, :] += out3[i, j, k, :, :, :]
                        out4[j, i, l, k, l, :, :, :] += out3[i, j, k, :, :, :]
                        out4[i, l, j, k, l, :, :, :] += out3[i, j, k, :, :, :]
                        out4[i, k, j, l, :, l, :, :] += out3[i, j, k, :, :, :]
                        out4[i, j, l, k, :, l, :, :] += out3[i, j, k, :, :, :]
                        out4[i, j, k, l, :, :, l, :] += out3[i, j, k, :, :, :]
        return (out1, out2, out3, out4)

    def calculate_dvec_spatial(self) -> 'Nparray':
        """Generate

        .. math::
            D^J_{ij} = \\sum_I \\langle J|a^\\dagger_i a_j|I\\rangle C_I

        using self.coeff as an input
        """
        return self._calculate_dvec_spatial_with_coeff(self.coeff)

    def calculate_dvec_spin(self) -> Tuple['Nparray', 'Nparray']:
        """Generate a pair of

        .. math::
            D^J_{ij} = \\sum_I \\langle J|a^\\dagger_i a_j|I\\rangle C_I

        using self.coeff as an input. Alpha and beta are seperately packed in
        the tuple to be returned
        """
        return self._calculate_dvec_spin_with_coeff(self.coeff)

    def calculate_dvec_spatial_fixed_j(self, jorb: int) -> 'Nparray':
        """Generate, for a fixed j,

        .. math::
            D^J_{ij} = \\sum_I \\langle J|a^\\dagger_i a_j|I\\rangle C_I

        using self.coeff as an input
        """
        return self._calculate_dvec_spatial_with_coeff_fixed_j(self.coeff, jorb)

    def calculate_dvec_spin_fixed_j(self, jorb: int) -> 'Nparray':
        """Generate a pair of the following, for a fixed j

        .. math::
            D^J_{ij} = \\sum_I \\langle J|a^\\dagger_i a_j|I\\rangle C_I

        using self.coeff as an input. Alpha and beta are seperately packed in
        the tuple to be returned
        """
        return self._calculate_dvec_spin_with_coeff_fixed_j(self.coeff, jorb)

    def _calculate_dvec_spatial_with_coeff(self, coeff: 'Nparray') -> 'Nparray':
        """Generate

        .. math::
            D^J_{ij} = \\sum_I \\langle J|a^\\dagger_i a_j|I\\rangle C_I

        """
        norb = self.norb()
        dvec = numpy.zeros((norb, norb, self.lena(), self.lenb()),
                           dtype=self._dtype)
        for i in range(norb):
            for j in range(norb):
                for source, target, parity in self.alpha_map(i, j):
                    dvec[i, j, target, :] += coeff[source, :] * parity
                for source, target, parity in self.beta_map(i, j):
                    dvec[i, j, :, target] += coeff[:, source] * parity
        return dvec

    def _calculate_dvec_spin_with_coeff(self, coeff: 'Nparray'
                                       ) -> Tuple['Nparray', 'Nparray']:
        """Generate

        .. math::

            D^J_{ij} = \\sum_I \\langle J|a^\\dagger_i a_j|I\\rangle C_I

        in the spin-orbital case
        """
        norb = self.norb()
        dveca = numpy.zeros((norb, norb, self.lena(), self.lenb()),
                            dtype=self._dtype)
        dvecb = numpy.zeros((norb, norb, self.lena(), self.lenb()),
                            dtype=self._dtype)
        for i in range(norb):
            for j in range(norb):
                # NOTE: alpha_map(i, j) == i^ j ladder ops
                # returns all connected basis states with parity
                for source, target, parity in self.alpha_map(i, j):
                    # source is the ket, target is the bra <S|i^ j|T> parity
                    # the ket has the coefficient associated with it! |T>C_{T}
                    # sum_{Ia, Ib} <JaJb|ia^ ja|IaIb>C(IaIb) =
                    # sum_{Ia, Ib} <Ja|ia^ ja| Ia> delta(Jb, Ib) C(IaIb)
                    dveca[i, j, target, :] += coeff[source, :] * parity
                for source, target, parity in self.beta_map(i, j):
                    # sum_{Ia, Ib} <JaJb|ib^ jb|IaIb>C(IaIb) =
                    # sum_{Ia, Ib} <Jb|ib^ jb| Ib> delta(Ja, Ia) C(IaIb)
                    dvecb[i, j, :, target] += coeff[:, source] * parity
        return (dveca, dvecb)

    def _calculate_dvec_spatial_with_coeff_fixed_j(self, coeff: 'Nparray',
                                                   jorb: int) -> 'Nparray':
        """Generate, for fixed j,

        .. math::
            D^J_{ij} = \\sum_I \\langle J|a^\\dagger_i a_j|I\\rangle C_I

        """
        norb = self.norb()
        assert (jorb < norb and jorb >= 0)
        dvec = numpy.zeros((norb, self.lena(), self.lenb()), dtype=self._dtype)
        for i in range(norb):
            for source, target, parity in self.alpha_map(i, jorb):
                dvec[i, target, :] += coeff[source, :] * parity
            for source, target, parity in self.beta_map(i, jorb):
                dvec[i, :, target] += coeff[:, source] * parity
        return dvec

    def _calculate_dvec_spin_with_coeff_fixed_j(self, coeff: 'Nparray',
                                                jorb: int) -> 'Nparray':
        """Generate, fixed j,

        .. math::

            D^J_{ij} = \\sum_I \\langle J|a^\\dagger_i a_j|I\\rangle C_I

        in the spin-orbital case
        """
        norb = self.norb()
        assert (jorb < norb * 2 and jorb >= 0)
        dvec = numpy.zeros((norb, self.lena(), self.lenb()), dtype=self._dtype)
        for i in range(norb):
            if jorb < norb:
                for source, target, parity in self.alpha_map(i, jorb):
                    dvec[i, target, :] += coeff[source, :] * parity
            else:
                for source, target, parity in self.beta_map(i, jorb - norb):
                    dvec[i, :, target] += coeff[:, source] * parity
        return dvec

    def _calculate_coeff_spatial_with_dvec(self, dvec: 'Nparray') -> 'Nparray':
        """Generate

        .. math::

            C_I = \\sum_J \\langle I|a^\\dagger_i a_j|J\\rangle D^J_{ij}
        """
        out = numpy.zeros(self.coeff.shape, dtype=self._dtype)
        for i in range(self.norb()):
            for j in range(self.norb()):
                for source, target, parity in self.alpha_map(j, i):
                    out[source, :] += dvec[i, j, target, :] * parity
                for source, target, parity in self.beta_map(j, i):
                    out[:, source] += dvec[i, j, :, target] * parity
        return out

    def calculate_dvec_spatial_compressed(self) -> 'Nparray':
        """Generate

        .. math::

            D^J_{i<j} = \\sum_I \\langle J|a^\\dagger_i a_j|I\\rangle C_I
        """
        norb = self.norb()
        nlt = norb * (norb + 1) // 2
        dvec = numpy.zeros((nlt, self.lena(), self.lenb()), dtype=self._dtype)
        for i in range(norb):
            for j in range(norb):
                ijn = min(i, j) + max(i, j) * (max(i, j) + 1) // 2
                for source, target, parity in self.alpha_map(i, j):
                    dvec[ijn, target, :] += self.coeff[source, :] * parity
                for source, target, parity in self.beta_map(i, j):
                    dvec[ijn, :, target] += self.coeff[:, source] * parity
        return dvec

    def calculate_coeff_spin_with_dvec(self, dvec: Tuple['Nparray', 'Nparray']
                                      ) -> 'Nparray':
        """Generate

        .. math::

            C_I = \\sum_J \\langle I|a^\\dagger_i a_j|J\\rangle D^J_{ij}
        """
        out = numpy.zeros(self.coeff.shape, dtype=self._dtype)
        for i in range(self.norb()):
            for j in range(self.norb()):
                for source, target, parity in self.alpha_map(j, i):
                    out[source, :] += dvec[0][i, j, target, :] * parity
                for source, target, parity in self.beta_map(j, i):
                    out[:, source] += dvec[1][i, j, :, target] * parity
        return out

    def evolve_inplace_individual_nbody_trivial(self, time: float,
                                                coeff: complex, opa: List[int],
                                                opb: List[int]) -> None:
        """
        This is the time evolution code for the cases where individual nbody
        becomes number operators (hence hat{T}^2 is nonzero) coeff includes
        parity due to sorting. opa and opb are integer arrays
        """
        n_a = len(opa)
        n_b = len(opb)
        coeff *= (-1)**(n_a * (n_a - 1) // 2 + n_b * (n_b - 1) // 2)

        amap = set()
        bmap = set()
        amask = reverse_integer_index(opa)
        bmask = reverse_integer_index(opb)
        for index in range(self.lena()):
            current = self._core.string_alpha(index)
            if (~current) & amask == 0:
                amap.add(index)
        for index in range(self.lenb()):
            current = self._core.string_beta(index)
            if (~current) & bmask == 0:
                bmap.add(index)

        factor = numpy.exp(-time * numpy.real(coeff) * 2.j)
        lamap = list(amap)
        lbmap = list(bmap)
        if len(lamap) != 0 and len(lbmap) != 0:
            xi, yi = numpy.meshgrid(lamap, lbmap, indexing='ij')
            self.coeff[xi, yi] *= factor

    def evolve_inplace_individual_nbody_nontrivial(
            self, time: float, coeff: complex, daga: List[int],
            undaga: List[int], dagb: List[int], undagb: List[int]) -> None:
        """
        This code time-evolves a wave function with an individual n-body
        generator which is spin-conserving. It is assumed that hat{T}^2 = 0.
        Using :math:`TT = 0` and :math:`TT^\\dagger` is diagonal in the determinant
        space, one could evaluate as

        .. math::
            \\exp(-i(T+T^\\dagger)t)
                &= 1 + i(T+T^\\dagger)t - \\frac{1}{2}(TT^\\dagger + T^\\dagger T)t^2
                 - i\\frac{1}{6}(TT^\\dagger T + T^\\dagger TT^\\dagger)t^3 + \\cdots \\\\
                &= -1 + \\cos(t\\sqrt{TT^\\dagger}) + \\cos(t\\sqrt{T^\\dagger T})
                 - iT\\frac{\\sin(t\\sqrt{T^\\dagger T})}{\\sqrt{T^\\dagger T}}
                 - iT^\\dagger\\frac{\\sin(t\\sqrt{TT^\\dagger})}{\\sqrt{TT^\\dagger}}
        """

        def isolate_number_operators(dag: List[int], undag: List[int],
                                     dagwork: List[int], undagwork: List[int],
                                     number: List[int]) -> int:
            """
            Pair-up daggered and undaggered operators that correspond to the
            same spin-orbital and isolate them, because they have to be treated
            differently.
            """
            par = 0
            for current in dag:
                if current in undag:
                    index1 = dagwork.index(current)
                    index2 = undagwork.index(current)
                    par += len(dagwork) - (index1 + 1) + index2
                    dagwork.remove(current)
                    undagwork.remove(current)
                    number.append(current)
            return par

        dagworka = copy.deepcopy(daga)
        dagworkb = copy.deepcopy(dagb)
        undagworka = copy.deepcopy(undaga)
        undagworkb = copy.deepcopy(undagb)
        numbera: List[int] = []
        numberb: List[int] = []

        parity = 0
        parity += isolate_number_operators(daga, undaga, dagworka, undagworka,
                                           numbera)
        parity += isolate_number_operators(dagb, undagb, dagworkb, undagworkb,
                                           numberb)
        ncoeff = coeff * (-1)**parity

        # code for (TTd)
        phase = (-1)**((len(daga) + len(undaga)) * (len(dagb) + len(undagb)))
        (cosdata1,
         sindata1) = self.apply_cos_sin(time, ncoeff, numbera + dagworka,
                                        undagworka, numberb + dagworkb,
                                        undagworkb)

        work_cof = numpy.conj(coeff) * phase
        cosdata1.ax_plus_y(
            -1.0j,
            sindata1.apply_individual_nbody(work_cof, undaga, daga, undagb,
                                            dagb))
        # code for (TdT)
        (cosdata2,
         sindata2) = self.apply_cos_sin(time, ncoeff, numbera + undagworka,
                                        dagworka, numberb + undagworkb,
                                        dagworkb)
        cosdata2.ax_plus_y(
            -1.0j,
            sindata2.apply_individual_nbody(coeff, daga, undaga, dagb, undagb))

        self.coeff = cosdata1.coeff + cosdata2.coeff - self.coeff

    def apply_cos_sin(self, time: float, ncoeff: complex, opa: List[int],
                      oha: List[int], opb: List[int],
                      ohb: List[int]) -> Tuple['FqeData', 'FqeData']:
        """
        Utility internal function that performs part of the operations in
        evolve_inplace_individual_nbody_nontrivial.  Isolated because it is
        also used in the counterpart in FqeDataSet.
        """
        amap = set()
        bmap = set()
        apmask = reverse_integer_index(opa)
        ahmask = reverse_integer_index(oha)
        bpmask = reverse_integer_index(opb)
        bhmask = reverse_integer_index(ohb)
        for index in range(self.lena()):
            current = self._core.string_alpha(index)
            if ((~current) & apmask) == 0 and (current & ahmask) == 0:
                amap.add(index)
        for index in range(self.lenb()):
            current = self._core.string_beta(index)
            if ((~current) & bpmask) == 0 and (current & bhmask) == 0:
                bmap.add(index)

        absol = numpy.absolute(ncoeff)
        cosfactor = numpy.cos(time * absol)
        sinfactor = numpy.sin(time * absol) / absol

        cosdata = copy.deepcopy(self)
        sindata = copy.deepcopy(self)
        sindata.coeff.fill(0.0)
        lamap = list(amap)
        lbmap = list(bmap)
        if len(lamap) == 0 or len(lbmap) == 0:
            return (cosdata, sindata)
        else:
            xi, yi = numpy.meshgrid(lamap, lbmap, indexing='ij')
            cosdata.coeff[xi, yi] *= cosfactor
            sindata.coeff[xi, yi] = self.coeff[xi, yi] * sinfactor
            return (cosdata, sindata)

    def alpha_map(self, iorb: int, jorb: int) -> List[Tuple[int, int, int]]:
        """Access the mapping for a singlet excitation from the current
        sector for alpha orbitals
        """
        return self._core.alpha_map(iorb, jorb)

    def beta_map(self, iorb: int, jorb: int) -> List[Tuple[int, int, int]]:
        """Access the mapping for a singlet excitation from the current
        sector for beta orbitals
        """
        return self._core.beta_map(iorb, jorb)

    def ax_plus_y(self, sval: complex, other: 'FqeData') -> 'FqeData':
        """Scale and add the data in the fqedata structure

            = sval*coeff + other

        """
        assert hash(self) == hash(other)
        self.coeff += other.coeff * sval
        return self

    def __hash__(self):
        """Fqedata sructures are unqiue in nele, s_z and the dimension.
        """
        return hash((self._nele, self._m_s))

    def conj(self) -> None:
        """Conjugate the coefficients
        """
        numpy.conjugate(self.coeff, self.coeff)

    def lena(self) -> int:
        """Length of the alpha configuration space
        """
        return self._core.lena()

    def lenb(self) -> int:
        """Length of the beta configuration space
        """
        return self._core.lenb()

    def nalpha(self) -> int:
        """Number of alpha electrons
        """
        return self._core.nalpha()

    def nbeta(self) -> int:
        """Number of beta electrons
        """
        return self._core.nbeta()

    def n_electrons(self) -> int:
        """Particle number getter
        """
        return self._nele

    def generator(self):
        """Iterate over the elements of the sector as alpha string, beta string
        coefficient
        """
        for inda in range(self._core.lena()):
            alpha_str = self._core.string_alpha(inda)
            for indb in range(self._core.lenb()):
                beta_str = self._core.string_beta(indb)
                yield alpha_str, beta_str, self.coeff[inda, indb]

    def norb(self) -> int:
        """Number of beta electrons
        """
        return self._core.norb()

    def norm(self) -> float:
        """Return the norm of the the sector wavefunction
        """
        return numpy.linalg.norm(self.coeff)

    def print_sector(self, pformat=None, threshold=0.0001):
        """Iterate over the strings and coefficients and print then
        using the print format
        """
        if pformat is None:

            def print_format(astr, bstr):
                return '{0:b}:{1:b}'.format(astr, bstr)

            pformat = print_format

        print('Sector N = {} : S_z = {}'.format(self._nele, self._m_s))
        for inda in range(self._core.lena()):
            alpha_str = self._core.string_alpha(inda)
            for indb in range(self._core.lenb()):
                beta_str = self._core.string_beta(indb)
                if numpy.abs(self.coeff[inda, indb]) > threshold:
                    print('{} {}'.format(pformat(alpha_str, beta_str),
                                         self.coeff[inda, indb]))

    def beta_inversion(self):
        """Return the coefficients with an inversion of the beta strings.
        """
        return numpy.flip(self.coeff, 1)

    def scale(self, sval: complex):
        """ Scale the wavefunction by the value sval

        Args:
            sval (complex) - value to scale by

        Returns:
            nothing - Modifies the wavefunction in place
        """
        self.coeff = self.coeff.astype(numpy.complex128) * sval

    def fill(self, value: complex):
        """ Fills the wavefunction with the value specified
        """
        self.coeff.fill(value)

    def set_wfn(self,
                strategy: Optional[str] = None,
                raw_data: 'Nparray' = numpy.empty(0)) -> None:
        """Set the values of the fqedata wavefunction based on a strategy

        Args:
            strategy (string) - the procedure to follow to set the coeffs

            raw_data (numpy.array(dim(self.lena(), self.lenb()), \
                dtype=numpy.complex128)) - the values to use
                if setting from data.  If vrange is supplied, the first column
                in data will correspond to the first index in vrange

        Returns:
            nothing - modifies the wavefunction in place
        """

        strategy_args = ['ones', 'zero', 'random', 'from_data', 'hartree-fock']

        if strategy is None and raw_data.shape == (0,):
            raise ValueError('No strategy and no data passed.'
                             ' Cannot initialize')

        if strategy == 'from_data' and raw_data.shape == (0,):
            raise ValueError('No data passed to initialize from')

        if raw_data.shape != (0,) and strategy not in ['from_data', None]:
            raise ValueError('Inconsistent strategy for set_vec passed with'
                             'data')

        if strategy not in strategy_args:
            raise ValueError('Unknown Argument passed to set_vec')

        if strategy == 'from_data':
            chkdim = raw_data.shape
            if chkdim[0] != self.lena() or chkdim[1] != self.lenb():
                raise ValueError('Dim of data passed {},{} is not compatible' \
                                 ' with {},{}'.format(chkdim[0],
                                                      chkdim[1],
                                                      self.lena(),
                                                      self.lenb()))

        if strategy == 'ones':
            self.coeff.fill(1. + .0j)
        elif strategy == 'zero':
            self.coeff.fill(0. + .0j)
        elif strategy == 'random':
            self.coeff[:, :] = rand_wfn(self.lena(), self.lenb())
        elif strategy == 'from_data':
            self.coeff = numpy.copy(raw_data)
        elif strategy == 'hartree-fock':
            self.coeff.fill(0 + .0j)
            self.coeff[0, 0] = 1.

    def __copy__(self):
        # FCIGraph is passed as by reference
        new_data = FqeData(nalpha=self._core.nalpha(),
                           nbeta=self._core.nbeta(),
                           norb=self._core.norb(),
                           fcigraph=self._core,
                           dtype=self._dtype)
        new_data._low_thresh = self._low_thresh
        new_data.coeff[:, :] = self.coeff[:, :]
        return new_data

    def __deepcopy__(self, memodict={}):  # pylint: disable=dangerous-default-value
        # FCIGraph is passed as by reference
        new_data = FqeData(nalpha=self._core.nalpha(),
                           nbeta=self._core.nbeta(),
                           norb=self._core.norb(),
                           fcigraph=self._core,
                           dtype=self._dtype)
        new_data._low_thresh = self._low_thresh
        # NOTE: numpy.copy only okay for numeric type self.coeff
        # NOTE: Otherwise implement copy.deepcopy(self.coeff)
        new_data.coeff[:, :] = self.coeff[:, :]
        return new_data

    def get_spin_opdm(self):
        """estimate the alpha-alpha and beta-beta block of the 1-RDM"""
        dveca, dvecb = self.calculate_dvec_spin()
        alpha_opdm = numpy.einsum('ijkl,kl->ij', dveca, self.coeff.conj())
        beta_opdm = numpy.einsum('ijkl,kl->ij', dvecb, self.coeff.conj())
        return alpha_opdm, beta_opdm

    def get_ab_tpdm(self):
        """Get the alpha-beta block of the 2-RDM

        tensor[i, j, k, l] = <ia^ jb^ kb la>
        """
        dveca, dvecb = self.calculate_dvec_spin()
        tpdm_ab = numpy.einsum('liab,jkab->ijkl', dveca.conj(), dvecb)
        return tpdm_ab

    def get_aa_tpdm(self):
        """Get the alpha-alpha block of the 2-RDM

        tensor[i, j, k, l] = <ia^ ja^ ka la>
        """
        dveca, _ = self.calculate_dvec_spin()
        alpha_opdm = numpy.einsum('ijkl,kl->ij', dveca, self.coeff.conj())
        nik_njl_aa = numpy.einsum('kiab,jlab->ikjl', dveca.conj(), dveca)
        tensor_aa = numpy.einsum('il,jk->ikjl', alpha_opdm,
                                 numpy.eye(alpha_opdm.shape[0]))
        return alpha_opdm, (tensor_aa - nik_njl_aa).transpose(0, 2, 1, 3)

    def get_bb_tpdm(self):
        """Get the beta-beta block of the 2-RDM

        tensor[i, j, k, l] = <ib^ jb^ kb lb>
        """
        _, dvecb = self.calculate_dvec_spin()
        beta_opdm = numpy.einsum('ijkl,kl->ij', dvecb, self.coeff.conj())
        nik_njl_bb = numpy.einsum('kiab,jlab->ikjl', dvecb.conj(), dvecb)
        tensor_bb = numpy.einsum('il,jk->ikjl', beta_opdm,
                                 numpy.eye(beta_opdm.shape[0]))
        return beta_opdm, (tensor_bb - nik_njl_bb).transpose(0, 2, 1, 3)

    def get_openfermion_rdms(self):
        """
        Generate spin-rdms and return in openfermion format
        """
        opdm_a, tpdm_aa = self.get_aa_tpdm()
        opdm_b, tpdm_bb = self.get_bb_tpdm()
        tpdm_ab = self.get_ab_tpdm()
        nqubits = 2 * opdm_a.shape[0]
        tpdm = numpy.zeros((nqubits, nqubits, nqubits, nqubits),
                           dtype=tpdm_ab.dtype)
        opdm = numpy.zeros((nqubits, nqubits), dtype=opdm_a.dtype)
        opdm[::2, ::2] = opdm_a
        opdm[1::2, 1::2] = opdm_b
        # same spin
        tpdm[::2, ::2, ::2, ::2] = tpdm_aa
        tpdm[1::2, 1::2, 1::2, 1::2] = tpdm_bb

        # mixed spin
        tpdm[::2, 1::2, 1::2, ::2] = tpdm_ab
        tpdm[::2, 1::2, ::2, 1::2] = numpy.einsum('ijkl->ijlk', -tpdm_ab)
        tpdm[1::2, ::2, ::2, 1::2] = numpy.einsum('ijkl->jilk', tpdm_ab)
        tpdm[1::2, ::2, 1::2, ::2] = numpy.einsum('ijkl->ijlk',
                                                  -tpdm[1::2, ::2, ::2, 1::2])

        return opdm, tpdm

    def get_three_spin_blocks_rdm(self):
        r"""
        Generate 3-RDM in the spin-orbital basis.

        3-RDM has Sz spin-blocks (aaa, aab, abb, bbb).  The strategy is to
        use this blocking to generate the minimal number of p^ q r^ s t^ u
        blocks and then generate the other components of the 3-RDM through
        symmeterization.  For example,

        p^ r^ t^ q s u = -p^ q r^ s t^ u - d(q, r) p^ t^ s u + d(q, t)p^ r^ s u
                        - d(s, t)p^ r^ q u + d(q,r)d(s,t)p^ u

        It is formulated in this way so we can use the dvec calculation.

        Given:
        ~D(p, j, Ia, Ib)(t, u) = \sum_{Ka, Kb}\sum_{LaLb}<IaIb|p^ j|KaKb><KaKb|t^ u|LaLb>C(La,Lb)

        then:
        p^ q r^ s t^ u = \sum_{Ia, Ib}D(p, q, Ia, Ib).conj(), ~D(p, j, Ia, Ib)(t, u)

        Example:

        p, q, r, s, t, u = 5, 5, 0, 4, 5, 1

        .. code-block:: python

            tdveca, tdvecb = fqe_data._calculate_dvec_spin_with_coeff(dveca[5, 1, :, :])
            test_ccc = np.einsum('liab,ab->il', dveca.conj(), tdveca[0, 4, :, :])[5, 5]
        """
        norb = self.norb()
        # p^q r^s t^ u spin-blocks
        ckckck_aaa = numpy.zeros((norb, norb, norb, norb, norb, norb),
                                 dtype=self._dtype)
        ckckck_aab = numpy.zeros((norb, norb, norb, norb, norb, norb),
                                 dtype=self._dtype)
        ckckck_abb = numpy.zeros((norb, norb, norb, norb, norb, norb),
                                 dtype=self._dtype)
        ckckck_bbb = numpy.zeros((norb, norb, norb, norb, norb, norb),
                                 dtype=self._dtype)

        dveca, dvecb = self.calculate_dvec_spin()
        dveca_conj, dvecb_conj = dveca.conj().copy(), dvecb.conj().copy()
        opdm, tpdm = self.get_openfermion_rdms()
        krond = numpy.eye(opdm.shape[0] // 2)
        # alpha-alpha-alpha
        for t, u in itertools.product(range(self.norb()), repeat=2):
            tdveca_a, _ = self._calculate_dvec_spin_with_coeff(
                dveca[t, u, :, :])
            tdveca_b, tdvecb_b = self._calculate_dvec_spin_with_coeff(
                dvecb[t, u, :, :])
            for r, s in itertools.product(range(self.norb()), repeat=2):
                # p(:)^ q(:) r^ s t^ u
                # a-a-a
                pq_rdm = numpy.einsum('liab,ab->il',
                                      dveca_conj,
                                      tdveca_a[r, s, :, :],
                                      optimize=True)
                ckckck_aaa[:, :, r, s, t, u] = pq_rdm
                # a-a-b
                pq_rdm = numpy.einsum('liab,ab->il',
                                      dveca_conj,
                                      tdveca_b[r, s, :, :],
                                      optimize=True)
                ckckck_aab[:, :, r, s, t, u] = pq_rdm
                # a-b-b
                pq_rdm = numpy.einsum('liab,ab->il',
                                      dveca_conj,
                                      tdvecb_b[r, s, :, :],
                                      optimize=True)
                ckckck_abb[:, :, r, s, t, u] = pq_rdm
                # b-b-b
                pq_rdm = numpy.einsum('liab,ab->il',
                                      dvecb_conj,
                                      tdvecb_b[r, s, :, :],
                                      optimize=True)
                ckckck_bbb[:, :, r, s, t, u] = pq_rdm

        # p^ r^ t^ u s q = p^ q r^ s t^ u + d(q, r) p^ t^ s u - d(q, t)p^ r^ s u
        #                 + d(s, t)p^ r^ q u - d(q,r)d(s,t)p^ u
        ccckkk_aaa = numpy.einsum('pqrstu->prtusq', ckckck_aaa)
        ccckkk_aaa += numpy.einsum('qr,ptsu->prtusq', krond,
                                   tpdm[::2, ::2, ::2, ::2])
        ccckkk_aaa -= numpy.einsum('qt,prsu->prtusq', krond,
                                   tpdm[::2, ::2, ::2, ::2])
        ccckkk_aaa += numpy.einsum('st,prqu->prtusq', krond,
                                   tpdm[::2, ::2, ::2, ::2])
        ccckkk_aaa -= numpy.einsum('qr,st,pu->prtusq', krond, krond,
                                   opdm[::2, ::2])

        ccckkk_aab = numpy.einsum('pqrstu->prtusq', ckckck_aab)
        ccckkk_aab += numpy.einsum('qr,ptsu->prtusq', krond,
                                   tpdm[::2, 1::2, ::2, 1::2])

        ccckkk_abb = numpy.einsum('pqrstu->prtusq', ckckck_abb)
        ccckkk_abb += numpy.einsum('st,prqu->prtusq', krond,
                                   tpdm[::2, 1::2, ::2, 1::2])

        ccckkk_bbb = numpy.einsum('pqrstu->prtusq', ckckck_bbb)
        ccckkk_bbb += numpy.einsum('qr,ptsu->prtusq', krond,
                                   tpdm[1::2, 1::2, 1::2, 1::2])
        ccckkk_bbb -= numpy.einsum('qt,prsu->prtusq', krond,
                                   tpdm[1::2, 1::2, 1::2, 1::2])
        ccckkk_bbb += numpy.einsum('st,prqu->prtusq', krond,
                                   tpdm[1::2, 1::2, 1::2, 1::2])
        ccckkk_bbb -= numpy.einsum('qr,st,pu->prtusq', krond, krond,
                                   opdm[1::2, 1::2])

        return ccckkk_aaa, ccckkk_aab, ccckkk_abb, ccckkk_bbb

    def get_three_pdm(self):
        norbs = self.norb()
        ccckkk = numpy.zeros(tuple([2 * norbs] * 6), dtype=self._dtype)
        ccckkk_aaa, ccckkk_aab, ccckkk_abb, ccckkk_bbb = self.get_three_spin_blocks_rdm(
        )

        # same spin
        ccckkk[::2, ::2, ::2, ::2, ::2, ::2] = ccckkk_aaa
        ccckkk[1::2, 1::2, 1::2, 1::2, 1::2, 1::2] = ccckkk_bbb

        # different spin-aab
        # (aab,baa), (aab,aba), (aab,aab)
        # (aba,baa), (aba,aba), (aba,aab)
        # (baa,baa), (baa,aba), (baa,aab)
        ccckkk[::2, ::2, 1::2, 1::2, ::2, ::2] = ccckkk_aab
        ccckkk[::2, ::2, 1::2, ::2, 1::2, ::2] = numpy.einsum(
            'pqrstu->pqrtsu', -ccckkk_aab)
        ccckkk[::2, ::2, 1::2, ::2, ::2, 1::2] = numpy.einsum(
            'pqrstu->pqrtus', ccckkk_aab)

        ccckkk[::2, 1::2, ::2, 1::2, ::2, ::2] = numpy.einsum(
            'pqrstu->prqstu', -ccckkk_aab)
        ccckkk[::2, 1::2, ::2, ::2, 1::2, ::2] = numpy.einsum(
            'pqrstu->prqtsu', ccckkk_aab)
        ccckkk[::2, 1::2, ::2, ::2, ::2, 1::2] = numpy.einsum(
            'pqrstu->prqtus', -ccckkk_aab)

        ccckkk[1::2, ::2, ::2, 1::2, ::2, ::2] = numpy.einsum(
            'pqrstu->rpqstu', ccckkk_aab)
        ccckkk[1::2, ::2, ::2, ::2, 1::2, ::2] = numpy.einsum(
            'pqrstu->rpqtsu', -ccckkk_aab)
        ccckkk[1::2, ::2, ::2, ::2, ::2, 1::2] = numpy.einsum(
            'pqrstu->rpqtus', ccckkk_aab)

        # different spin-abb
        # (abb,bba), (abb,bab), (abb,abb)
        # (bab,bba), (bab,bab), (bab,abb)
        # (abb,bba), (abb,bab), (abb,abb)
        ccckkk[::2, 1::2, 1::2, 1::2, 1::2, ::2] = ccckkk_abb
        ccckkk[::2, 1::2, 1::2, 1::2, ::2, 1::2] = numpy.einsum(
            'pqrstu->pqrsut', -ccckkk_abb)
        ccckkk[::2, 1::2, 1::2, ::2, 1::2, 1::2] = numpy.einsum(
            'pqrstu->pqrust', ccckkk_abb)

        ccckkk[1::2, ::2, 1::2, 1::2, 1::2, ::2] = numpy.einsum(
            'pqrstu->qprstu', -ccckkk_abb)
        ccckkk[1::2, ::2, 1::2, 1::2, ::2, 1::2] = numpy.einsum(
            'pqrstu->qprsut', ccckkk_abb)
        ccckkk[1::2, ::2, 1::2, ::2, 1::2, 1::2] = numpy.einsum(
            'pqrstu->qprust', -ccckkk_abb)

        ccckkk[1::2, 1::2, ::2, 1::2, 1::2, ::2] = numpy.einsum(
            'pqrstu->qrpstu', ccckkk_abb)
        ccckkk[1::2, 1::2, ::2, 1::2, ::2, 1::2] = numpy.einsum(
            'pqrstu->qrpsut', -ccckkk_abb)
        ccckkk[1::2, 1::2, ::2, ::2, 1::2, 1::2] = numpy.einsum(
            'pqrstu->qrpust', ccckkk_abb)

        return ccckkk
