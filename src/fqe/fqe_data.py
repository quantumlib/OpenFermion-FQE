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

""" Fermionic Quantum Emulator data class for holding wavefunction data.
"""

import copy
from typing import List, Optional, Tuple

import numpy
from scipy.special import binom

from fqe.bitstring import integer_index, get_bit, count_bits_above
from fqe.bitstring import set_bit, unset_bit
from fqe.util import rand_wfn, validate_config
from fqe.fci_graph import FciGraph
from fqe.fci_graph_set import FciGraphSet


nparray = numpy.ndarray


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

        if not (fcigraph is None) and (nalpha != fcigraph.nalpha() or \
            nbeta != fcigraph.nbeta() or norb != fcigraph.norb()):
            raise ValueError("FciGraph does not match other parameters")

        if fcigraph is None:
            self._core = FciGraph(nalpha, nbeta, norb)
        else:
            self._core = fcigraph

        self._dtype = dtype
        self._nele = self._core.nalpha() + self._core.nbeta()
        self._m_s = self._core.nalpha() - self._core.nbeta()
        self.coeff = numpy.zeros((self._core.lena(), self._core.lenb()),
                                 dtype=self._dtype)


    def get_fcigraph(self) -> 'FciGraph':
        """
        Returns the underlying FciGraph object
        """
        return self._core


    def apply_diagonal_unitary_array(self,
                                     array: nparray) -> nparray:
        """Iterate over each element and return the exponential scaled
        contribution.
        """
        beta_ptr = 0

        if array.size == 2*self.norb():
            beta_ptr = self.norb()

        elif array.size != self.norb():
            raise ValueError('Non-diagonal array passed' \
                             ' into apply_diagonal_array')

        data = numpy.copy(self.coeff).astype(numpy.complex128)

        for alp_cnf in range(self._core.lena()):
            diag_ele = 0.0
            for ind in integer_index(self._core.string_alpha(alp_cnf)):
                diag_ele += array[ind]

            data[alp_cnf, :] *= numpy.exp(diag_ele)

        for bet_cnf in range(self._core.lenb()):
            diag_ele = 0.0
            for ind in integer_index(self._core.string_beta(bet_cnf)):
                diag_ele += array[beta_ptr + ind]

            data[:, bet_cnf] *= numpy.exp(diag_ele)

        return data


    def diagonal_coulomb(self,
                         diag: nparray,
                         array: nparray) -> nparray:
        """Iterate over each element and return the scaled wavefunction.
        """
        data = numpy.copy(self.coeff)

        beta_occ = []
        for bet_cnf in range(self.lenb()):
            beta_occ.append(integer_index(self._core.string_beta(bet_cnf)))

        for alp_cnf in range(self.lena()):
            alpha_occ = integer_index(self._core.string_alpha(alp_cnf))
            for bet_cnf in range(self._core.lenb()):
                occ = alpha_occ + beta_occ[bet_cnf]
                diag_ele = 0.0
                for ind in occ:
                    diag_ele += diag[ind]
                    for jnd in occ:
                        diag_ele += array[ind, jnd]

                data[alp_cnf, bet_cnf] *= numpy.exp(diag_ele)

        return data


    def apply(self, array: Tuple[nparray]) -> nparray:
        """
        API for application of dense operators (1- through 4-body operators) to
        the wavefunction self.
        """

        out = copy.deepcopy(self)
        out.apply_inplace(array)
        return out


    def apply_inplace(self, array: Tuple[nparray]) -> None:
        """
        API for application of dense operators (1- through 4-body operators) to
        the wavefunction self.
        """

        spatial = array[0].shape[0] == self.norb()
        if len(array) == 1:
            if spatial:
                self.coeff = self.apply_array_spatial1(array[0])
            else:
                self.coeff = self.apply_array_spin1(array[0])
        elif len(array) == 2:
            if spatial:
                self.coeff = self.apply_array_spatial12(array[0], array[1])
            else:
                self.coeff = self.apply_array_spin12(array[0], array[1])
        elif len(array) == 3:
            if spatial:
                self.coeff = self.apply_array_spatial123(array[0],
                                                         array[1],
                                                         array[2])
            else:
                self.coeff = self.apply_array_spin123(array[0],
                                                      array[1],
                                                      array[2])
        elif len(array) == 4:
            if spatial:
                self.coeff = self.apply_array_spatial1234(array[0],
                                                          array[1],
                                                          array[2],
                                                          array[3])
            else:
                self.coeff = self.apply_array_spin1234(array[0],
                                                       array[1],
                                                       array[2],
                                                       array[3])
        else:
            raise ValueError('unexpected array passed in FqeData apply_inplace')


    def apply_array_spatial1(self, h1e: nparray) -> nparray:
        """
        API for application of 1- and 2-body spatial operators to the
        wavefunction self.  It returns array that corresponds to the
        output wave function data.
        """
        assert h1e.shape == (self.norb(), self.norb())
        dvec = self.calculate_dvec_spatial()
        return numpy.einsum("ij,ijkl->kl", h1e, dvec)


    def apply_array_spin1(self, h1e: nparray) -> nparray:
        """
        API for application of 1- and 2-body spatial operators to the
        wavefunction self. It returns numpy.ndarray that corresponds to the
        output wave function data.
        """
        norb = self.norb()
        assert h1e.shape == (norb*2, norb*2)
        (dveca, dvecb) = self.calculate_dvec_spin()
        return numpy.einsum("ij,ijkl->kl", h1e[:norb, :norb], dveca) \
             + numpy.einsum("ij,ijkl->kl", h1e[norb:, norb:], dvecb)


    def apply_array_spatial12(self,
                              h1e: nparray,
                              h2e: nparray) -> nparray:
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


        thresh = 0.3
        if nalpha < norb * thresh and nbeta < norb * thresh:
            graphset = FciGraphSet(2, 2)
            graphset.append(self._core)
            if nalpha-2 >= 0:
                graphset.append(FciGraph(nalpha-2, nbeta, norb))
            if nalpha-1 >= 0 and nbeta-1 >= 0:
                graphset.append(FciGraph(nalpha-1, nbeta-1, norb))
            if nbeta-2 >= 0:
                graphset.append(FciGraph(nalpha, nbeta-2, norb))
            return self.apply_array_spatial12_lowfilling(h1e, h2e)

        return self.apply_array_spatial12_halffilling(h1e, h2e)


    def apply_array_spin12(self,
                           h1e: nparray,
                           h2e: nparray) -> nparray:
        """
        API for application of 1- and 2-body spin-orbital operators to the
        wavefunction self.  It returns numpy.ndarray that corresponds to the
        output wave function data. Depending on the filling, it automatically
        chooses an efficient code.
        """
        norb = self.norb()
        assert h1e.shape == (norb*2, norb*2)
        assert h2e.shape == (norb*2, norb*2, norb*2, norb*2)
        nalpha = self.nalpha()
        nbeta = self.nbeta()


        thresh = 0.3
        if nalpha < norb * thresh and nbeta < norb * thresh:
            graphset = FciGraphSet(2, 2)
            graphset.append(self._core)
            if nalpha-2 >= 0:
                graphset.append(FciGraph(nalpha-2, nbeta, norb))
            if nalpha-1 >= 0 and nbeta-1 >= 0:
                graphset.append(FciGraph(nalpha-1, nbeta-1, norb))
            if nbeta-2 >= 0:
                graphset.append(FciGraph(nalpha, nbeta-2, norb))
            return self.apply_array_spin12_lowfilling(h1e, h2e)

        return self.apply_array_spin12_halffilling(h1e, h2e)


    def apply_array_spatial12_halffilling(self,
                                          h1e: nparray,
                                          h2e: nparray) -> nparray:
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
            out += self.calculate_coeff_spatial_with_dvec(dvec)
        else:
            nij = norb*(norb+1)//2
            h1ec = numpy.zeros((nij), dtype=self._dtype)
            h2ec = numpy.zeros((nij, nij), dtype=self._dtype)
            for i in range(norb):
                for j in range(i+1):
                    ijn = j + i*(i+1)//2
                    h1ec[ijn] = h1e[i, j]
                    for k in range(norb):
                        for l in range(k+1):
                            kln = l + k*(k+1)//2
                            h2ec[ijn, kln] = h2e[i, j, k, l]
            dvec = self.calculate_dvec_spatial_compressed()
            out = numpy.einsum("i,ikl->kl", h1ec, dvec)
            dvec = numpy.einsum("ik,kmn->imn", h2ec, dvec)
            for i in range(self.norb()):
                for j in range(self.norb()):
                    ijn = min(i, j) + max(i, j)*(max(i, j)+1)//2
                    for source, target, parity in self.alpha_map(j, i):
                        out[source, :] += dvec[ijn, target, :] * parity
                    for source, target, parity in self.beta_map(j, i):
                        out[:, source] += dvec[ijn, :, target] * parity

        return out


    def apply_array_spin12_halffilling(self,
                                       h1e: nparray,
                                       h2e: nparray) -> nparray:
        """
        Standard code to calculate application of 1- and 2-body spin-orbital
        operators to the wavefunction self. It returns numpy.ndarray that
        corresponds to the output wave function data.
        """
        h1e = copy.deepcopy(h1e)
        h2e = numpy.moveaxis(copy.deepcopy(h2e), 1, 2) * (-1.0)
        norb = self.norb()
        for k in range(norb*2):
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


    def apply_array_spatial12_lowfilling(self,
                                         h1e: nparray,
                                         h2e: nparray) -> nparray:
        """
        Low-filling specialization of the code to calculate application of
        1- and 2-body spatial operators to the wavefunction self.  It returns
        numpy.ndarray that corresponds to the output wave function data.
        """
        out = self.apply_array_spatial1(h1e)

        norb = self.norb()
        nalpha = self.nalpha()
        nbeta = self.nbeta()
        lena = self.lena()
        lenb = self.lenb()
        nlt = norb*(norb+1)//2

        h2ecomp = numpy.zeros((nlt, nlt), dtype=self._dtype)
        for i in range(norb):
            for j in range(i+1, norb):
                ijn = i+j*(j+1)//2
                for k in range(norb):
                    for l in range(k+1, norb):
                        h2ecomp[ijn, k+l*(l+1)//2] = (h2e[i, j, k, l]
                                                      - h2e[i, j, l, k]
                                                      - h2e[j, i, k, l]
                                                      + h2e[j, i, l, k])

        if nalpha-2 >= 0:
            alpha_map, _ = self._core.find_mapping(-2, 0)
            intermediate = numpy.zeros((nlt,
                                        int(binom(norb, nalpha-2)),
                                        lenb),
                                       dtype=self._dtype)
            for i in range(norb):
                for j in range(i+1, norb):
                    ijn = i+j*(j+1) // 2
                    for source, target, parity in alpha_map[(i, j)]:
                        work = self.coeff[source, :] * parity
                        intermediate[ijn, target, :] += work

            intermediate = numpy.einsum('ij,jmn->imn', h2ecomp, intermediate)

            for i in range(norb):
                for j in range(i+1, norb):
                    ijn = i+j*(j+1)//2
                    for source, target, parity in alpha_map[(i, j)]:
                        out[source, :] -= intermediate[ijn, target, :] * parity

        if self.nalpha()-1 >= 0 and self.nbeta()-1 >= 0:
            alpha_map, beta_map = self._core.find_mapping(-1, -1)
            intermediate = numpy.zeros((norb,
                                        norb,
                                        int(binom(norb, nalpha-1)),
                                        int(binom(norb, nbeta-1))),
                                       dtype=self._dtype)

            for i in range(norb): #alpha
                for j in range(norb):
                    for sourcea, targeta, paritya in alpha_map[(i,)]:
                        sign = ((-1) ** (nalpha - 1)) * paritya
                        for sourceb, targetb, parityb in beta_map[(j,)]:
                            work = self.coeff[sourcea, sourceb] * sign * parityb
                            intermediate[i, j, targeta, targetb] += 2 * work

            intermediate = numpy.einsum('ijkl,klmn->ijmn', h2e, intermediate)

            for i in range(norb):
                for j in range(norb):
                    for sourcea, targeta, paritya in alpha_map[(i,)]:
                        sign = ((-1) ** nalpha) * paritya
                        for sourceb, targetb, parityb in beta_map[(j,)]:
                            work = intermediate[i, j, targeta, targetb] * sign
                            out[sourcea, sourceb] += work * parityb

        if self.nbeta()-2 >= 0:
            _, beta_map = self._core.find_mapping(0, -2)
            intermediate = numpy.zeros((nlt, lena, int(binom(norb, nbeta-2))),
                                       dtype=self._dtype)
            for i in range(norb):
                for j in range(i+1, norb):
                    ijn = i+j*(j+1)//2
                    for source, target, parity in beta_map[(i, j)]:
                        work = self.coeff[:, source] * parity
                        intermediate[ijn, :, target] += work

            intermediate = numpy.einsum('ij,jmn->imn', h2ecomp, intermediate)

            for i in range(norb):
                for j in range(i+1, norb):
                    ijn = i+j*(j+1)//2
                    for source, target, sign in beta_map[(min(i, j), max(i, j))]:
                        out[:, source] -= intermediate[ijn, :, target] * sign
        return out



    def apply_array_spin12_lowfilling(self, h1e: nparray, h2e: nparray) -> nparray:
        """
        Low-filling specialization of the code to calculate application of
        1- and 2-body spin-orbital operators to the wavefunction self. It
        returns numpy.ndarray that corresponds to the output wave function data.
        """
        out = self.apply_array_spin1(h1e)

        norb = self.norb()
        nalpha = self.nalpha()
        nbeta = self.nbeta()
        lena = self.lena()
        lenb = self.lenb()
        nlt = norb*(norb+1)//2

        h2ecompa = numpy.zeros((nlt, nlt), dtype=self._dtype)
        h2ecompb = numpy.zeros((nlt, nlt), dtype=self._dtype)
        for i in range(norb):
            for j in range(i+1, norb):
                ijn = i+j*(j+1)//2
                for k in range(norb):
                    for l in range(k+1, norb):
                        kln = k+l*(l+1)//2
                        h2ecompa[ijn, kln] = (h2e[i, j, k, l]
                                              - h2e[i, j, l, k]
                                              - h2e[j, i, k, l]
                                              + h2e[j, i, l, k])
                        ino = i + norb
                        jno = j + norb
                        kno = k + norb
                        lno = l + norb
                        h2ecompb[ijn, kln] = (h2e[ino, jno, kno, lno]
                                              - h2e[ino, jno, lno, kno]
                                              - h2e[jno, ino, kno, lno]
                                              + h2e[jno, ino, lno, kno])

        if nalpha - 2 >= 0:
            alpha_map, _ = self._core.find_mapping(-2, 0)
            intermediate = numpy.zeros((nlt,
                                        int(binom(norb, nalpha-2)),
                                        lenb),
                                       dtype=self._dtype)
            for i in range(norb):
                for j in range(i+1, norb):
                    ijn = i+j*(j+1)//2
                    for source, target, parity in alpha_map[(i, j)]:
                        work = self.coeff[source, :] * parity
                        intermediate[ijn, target, :] += work

            intermediate = numpy.einsum('ij,jmn->imn', h2ecompa, intermediate)

            for i in range(norb):
                for j in range(i+1, norb):
                    ijn = i+j*(j+1)//2
                    for source, target, parity in alpha_map[(i, j)]:
                        out[source, :] -= intermediate[ijn, target, :] * parity

        if self.nalpha()-1 >= 0 and self.nbeta()-1 >= 0:
            alpha_map, beta_map = self._core.find_mapping(-1, -1)
            intermediate = numpy.zeros((norb,
                                        norb,
                                        int(binom(norb, nalpha-1)),
                                        int(binom(norb, nbeta-1))),
                                       dtype=self._dtype)

            for i in range(norb):
                for j in range(norb):
                    for sourcea, targeta, paritya in alpha_map[(i,)]:
                        sign = i((-1) ** (nalpha-1))*paritya
                        for sourceb, targetb, parityb in beta_map[(j,)]:
                            work = self.coeff[sourcea, sourceb] * sign * parityb
                            intermediate[i, j, targeta, targetb] += 2 * work

            intermediate = numpy.einsum('ijkl,klmn->ijmn',
                                        h2e[:norb, norb:, :norb, norb:],
                                        intermediate)

            for i in range(norb):
                for j in range(norb):
                    for sourcea, targeta, paritya in alpha_map[(i,)]:
                        paritya *= (-1) ** nalpha
                        for sourceb, targetb, parityb in beta_map[(j,)]:
                            work = intermediate[i, j, targeta, targetb]
                            out[sourcea, sourceb] += work * paritya * parityb

        if self.nbeta()-2 >= 0:
            _, beta_map = self._core.find_mapping(0, -2)
            intermediate = numpy.zeros((nlt,
                                        lena,
                                        int(binom(norb, nbeta-2))),
                                       dtype=self._dtype)
            for i in range(norb):
                for j in range(i+1, norb):
                    ijn = i+j*(j+1)//2
                    for source, target, parity in beta_map[(i, j)]:
                        work = self.coeff[:, source] * parity
                        intermediate[ijn, :, target] += work

            intermediate = numpy.einsum('ij,jmn->imn', h2ecompb, intermediate)

            for i in range(norb):
                for j in range(i+1, norb):
                    ijn = i+j*(j+1)//2
                    for source, target, sign in beta_map[(min(i, j), max(i, j))]:
                        out[:, source] -= intermediate[ijn, :, target] * sign
        return out


    def apply_array_spatial123(self, h1e: nparray, h2e: nparray, h3e: nparray, \
                               dvec: nparray = None, evec: nparray = None) -> nparray:
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
                    nh2e[j, k, :, :] += (- h3e[k, j, i, i, :, :]
                                         - h3e[j, i, k, i, :, :]
                                         - h3e[j, k, i, :, i, :])
                nh1e[:, :] += h3e[:, i, j, i, j, :]

        out = self.apply_array_spatial12_halffilling(nh1e, nh2e)

        if dvec is None:
            dvec = self.calculate_dvec_spatial()
        if evec is None:
            evec = numpy.zeros((norb, norb, norb, norb, lena, lenb),
                               dtype=self._dtype)
            for i in range(norb):
                for j in range(norb):
                    tmp = dvec[i, j, :, :]
                    tmp2 = self.calculate_dvec_spatial_with_coeff(tmp)
                    evec[:, :, i, j, :, :] = tmp2[:, :, :, :]

        dvec = numpy.einsum('ikmjln,klmnxy->ijxy', h3e, evec)

        out -= self.calculate_coeff_spatial_with_dvec(dvec)
        return out


    def apply_array_spin123(self,
                            h1e: nparray,
                            h2e: nparray,
                            h3e: nparray,
                            dvec: Optional[Tuple[nparray, nparray]] = None,
                            evec: Optional[Tuple[nparray, nparray, nparray, nparray]] \
                                  = None) -> nparray:
        """
        Code to calculate application of 1- through 3-body spin-orbital
        operators to the wavefunction self. It returns numpy.ndarray that
        corresponds to the output wave function data.
        """
        norb = self.norb()
        assert h3e.shape == (norb*2, norb*2, norb*2, norb*2, norb*2, norb*2)
        assert not (dvec is None) ^ (evec is None)

        from1234 = (dvec is not None) and (evec is not None)

        lena = self.lena()
        lenb = self.lenb()

        nh1e = numpy.copy(h1e)
        nh2e = numpy.copy(h2e)

        for i in range(norb*2):
            for j in range(norb*2):
                for k in range(norb*2):
                    nh2e[j, k, :, :] += (- h3e[k, j, i, i, :, :]
                                         - h3e[j, i, k, i, :, :]
                                         - h3e[j, k, i, :, i, :])

                nh1e[:, :] += h3e[:, i, j, i, j, :]

        out = self.apply_array_spin12_halffilling(nh1e, nh2e)

        if not from1234:
            (dveca, dvecb) = self.calculate_dvec_spin()
        else:
            dveca, dvecb = dvec[0], dvec[1]

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
                    tmp = self.calculate_dvec_spin_with_coeff(dveca[i, j, :, :])
                    evecaa[:, :, i, j, :, :] = tmp[0][:, :, :, :]

                    tmp = self.calculate_dvec_spin_with_coeff(dvecb[i, j, :, :])
                    evecab[:, :, i, j, :, :] = tmp[0][:, :, :, :]
                    evecbb[:, :, i, j, :, :] = tmp[1][:, :, :, :]
        else:
            evecaa, evecab, evecba, evecbb = evec[0], evec[1], evec[2], evec[3]

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


    def apply_array_spatial1234(self,
                                h1e: nparray,
                                h2e: nparray,
                                h3e: nparray,
                                h4e: nparray) -> nparray:
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
                        nh2e[i, j, :, :] += (h4e[j, l, i, k, l, k, :, :]
                                             + h4e[i, j, l, k, l, k, :, :]
                                             + h4e[i, l, k, j, l, k, :, :]
                                             + h4e[j, i, k, l, l, k, :, :]
                                             + h4e[i, k, j, l, k, :, l, :]
                                             + h4e[j, i, k, l, k, :, l, :]
                                             + h4e[i, j, k, l, :, k, l, :])
                        nh3e[i, j, k, :, :, :] += (h4e[k, i, j, l, l, :, :, :]
                                                   + h4e[j, i, l, k, l, :, :, :]
                                                   + h4e[i, l, j, k, l, :, :, :]
                                                   + h4e[i, k, j, l, :, l, :, :]
                                                   + h4e[i, j, l, k, :, l, :, :]
                                                   + h4e[i, j, k, l, :, :, l, :])

        dvec = self.calculate_dvec_spatial()
        evec = numpy.zeros((norb, norb, norb, norb, lena, lenb),
                           dtype=self._dtype)

        for i in range(norb):
            for j in range(norb):
                tmp = dvec[i, j, :, :]
                tmp2 = self.calculate_dvec_spatial_with_coeff(tmp)
                evec[:, :, i, j, :, :] = tmp2[:, :, :, :]

        out = self.apply_array_spatial123(nh1e, nh2e, nh3e, dvec, evec)

        evec = numpy.einsum('ikmojlnp,mnopxy->ijklxy', h4e, evec)

        dvec2 = numpy.zeros(dvec.shape, dtype=self._dtype)
        for i in range(norb):
            for j in range(norb):
                dvec[:, :, :, :] = evec[i, j, :, :, :, :]
                cvec = self.calculate_coeff_spatial_with_dvec(dvec)
                dvec2[i, j, :, :] += cvec[:, :]

        out += self.calculate_coeff_spatial_with_dvec(dvec2)
        return out


    def apply_array_spin1234(self,
                             h1e: nparray,
                             h2e: nparray,
                             h3e: nparray,
                             h4e: nparray) -> nparray:
        """
        Code to calculate application of 1- through 4-body spin-orbital
        operators to the wavefunction self. It returns numpy.ndarray that
        corresponds to the output wave function data.
        """
        norb = self.norb()
        tno = 2*norb
        assert h4e.shape == (tno, tno, tno, tno, tno, tno, tno, tno)
        lena = self.lena()
        lenb = self.lenb()

        nh1e = numpy.copy(h1e)
        nh2e = numpy.copy(h2e)
        nh3e = numpy.copy(h3e)

        for i in range(norb*2):
            for j in range(norb*2):
                for k in range(norb*2):
                    nh1e[:, :] -= h4e[:, j, i, k, j, i, k, :]
                    for l in range(norb*2):
                        nh2e[i, j, :, :] += (h4e[j, l, i, k, l, k, :, :]
                                             + h4e[i, j, l, k, l, k, :, :]
                                             + h4e[i, l, k, j, l, k, :, :]
                                             + h4e[j, i, k, l, l, k, :, :]
                                             + h4e[i, k, j, l, k, :, l, :]
                                             + h4e[j, i, k, l, k, :, l, :]
                                             + h4e[i, j, k, l, :, k, l, :])
                        nh3e[i, j, k, :, :, :] += (h4e[k, i, j, l, l, :, :, :]
                                                   + h4e[j, i, l, k, l, :, :, :]
                                                   + h4e[i, l, j, k, l, :, :, :]
                                                   + h4e[i, k, j, l, :, l, :, :]
                                                   + h4e[i, j, l, k, :, l, :, :]
                                                   + h4e[i, j, k, l, :, :, l, :])

        (dveca, dvecb) = self.calculate_dvec_spin()
        evecaa = numpy.zeros((norb, norb, norb, norb, lena, lenb), dtype=self._dtype)
        evecab = numpy.zeros((norb, norb, norb, norb, lena, lenb), dtype=self._dtype)
        evecba = numpy.zeros((norb, norb, norb, norb, lena, lenb), dtype=self._dtype)
        evecbb = numpy.zeros((norb, norb, norb, norb, lena, lenb), dtype=self._dtype)
        for i in range(norb):
            for j in range(norb):
                tmp = self.calculate_dvec_spin_with_coeff(dveca[i, j, :, :])
                evecaa[:, :, i, j, :, :] = tmp[0][:, :, :, :]
                evecba[:, :, i, j, :, :] = tmp[1][:, :, :, :]

                tmp = self.calculate_dvec_spin_with_coeff(dvecb[i, j, :, :])
                evecab[:, :, i, j, :, :] = tmp[0][:, :, :, :]
                evecbb[:, :, i, j, :, :] = tmp[1][:, :, :, :]

        out = self.apply_array_spin123(nh1e,
                                       nh2e,
                                       nh3e,
                                       (dveca, dvecb),
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
        self.coeff *= s_z + s_z*s_z + self.nbeta()

        if self.nalpha() != self.norb() and self.nbeta() != 0:
            dvec = numpy.zeros((norb, norb, self.lena(), self.lenb()),
                               dtype=self._dtype)
            for i in range(norb): #creation
                for j in range(norb): #annihilation
                    for source, target, parity in self.alpha_map(i, j):
                        dvec[i, j, target, :] += orig[source, :] * parity
            for i in range(self.norb()):
                for j in range(self.norb()):
                    for source, target, parity in self.beta_map(j, i):
                        self.coeff[:, source] -= dvec[j, i, :, target] * parity


    def apply_individual_nbody(self,
                               coeff: complex,
                               daga: List[int],
                               undaga: List[int],
                               dagb: List[int],
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
                        alphamap.append((index,
                                         self._core.index_alpha(current),
                                         (-1)**parity))
                    else:
                        betamap.append((index,
                                        self._core.index_beta(current),
                                        (-1)**parity))
        make_mapping_each(True)
        make_mapping_each(False)
        out = copy.deepcopy(self)
        out.coeff.fill(0.0)
        for sourcea, targeta, paritya in alphamap:
            for sourceb, targetb, parityb in betamap:
                work = coeff * self.coeff[sourcea, sourceb] * paritya * parityb
                out.coeff[targeta, targetb] = work
        return out


    def rdm1(self, bradata: 'FqeData' = None) -> nparray:
        """
        API for calculating 1-particle RDMs given a wave function. When bradata
        is given, it calculates transition RDMs. Depending on the filling, the
        code selects an optimal algorithm.
        """
        if bradata is not None:
            dvec2 = bradata.calculate_dvec_spatial()
        else:
            dvec2 = self.calculate_dvec_spatial()
        return (numpy.einsum('jikl,kl->ij', dvec2.conj(), self.coeff), )


    def rdm12(self, bradata: 'FqeData' = None) -> nparray:
        """
        API for calculating 1- and 2-particle RDMs given a wave function. When
        bradata is given, it calculates transition RDMs. Depending on the
        filling, the code selects an optimal algorithm.
        """
        norb = self.norb()
        nalpha = self.nalpha()
        nbeta = self.nbeta()


        thresh = 0.3
        if nalpha < norb * thresh and nbeta < norb * thresh:
            graphset = FciGraphSet(2, 2)
            graphset.append(self._core)
            if nalpha-2 >= 0:
                graphset.append(FciGraph(nalpha-2, nbeta, norb))
            if nalpha-1 >= 0 and nbeta-1 >= 0:
                graphset.append(FciGraph(nalpha-1, nbeta-1, norb))
            if nbeta-2 >= 0:
                graphset.append(FciGraph(nalpha, nbeta-2, norb))
            return self.rdm12_lowfilling(bradata)

        return self.rdm12_halffilling(bradata)


    def rdm12_halffilling(self, bradata: 'FqeData' = None) -> nparray:
        """
        Standard code for calculating 1- and 2-particle RDMs given a wave
        function. When bradata is given, it calculates transition RDMs.
        """
        dvec = self.calculate_dvec_spatial()
        dvec2 = dvec if bradata is None else bradata.calculate_dvec_spatial()
        out1 = numpy.einsum('jikl,kl->ij', dvec2, self.coeff)
        out2 = numpy.einsum('jikl,mnkl->imjn', dvec2.conj(), dvec) * (-1.0)
        for i in range(self.norb()):
            out2[:, i, i, :] += out1[:, :]
        return (out1, out2)


    def rdm12_lowfilling(self, bradata: 'FqeData' = None) -> nparray:
        """
        Low-filling specialization of the code for Calculating 1- and
        2-particle RDMs given a wave function. When bradata is given, it
        calculates transition RDMs.
        """
        norb = self.norb()
        nalpha = self.nalpha()
        nbeta = self.nbeta()
        lena = self.lena()
        lenb = self.lenb()
        nlt = norb*(norb+1)//2

        outpack = numpy.zeros((nlt, nlt), dtype=self._dtype)
        outunpack = numpy.zeros((norb, norb, norb, norb), dtype=self._dtype)
        if nalpha-2 >= 0:
            alpha_map, _ = self._core.find_mapping(-2, 0)

            def compute_intermediate0(coeff):
                tmp = numpy.zeros((nlt,
                                   int(binom(norb, nalpha-2)),
                                   lenb),
                                  dtype=self._dtype)
                for i in range(norb):
                    for j in range(i+1, norb):
                        ijn = i+j*(j+1)//2
                        for source, target, parity in alpha_map[(i, j)]:
                            tmp[ijn, target, :] += coeff[source, :] * parity
                return tmp

            inter = compute_intermediate0(self.coeff)
            if bradata is None:
                inter2 = inter
            else:
                inter2 = compute_intermediate0(bradata.coeff)

            outpack += numpy.einsum('imn,kmn->ik', inter2.conj(), inter)

        if self.nalpha()-1 >= 0 and self.nbeta()-1 >= 0:
            alpha_map, beta_map = self._core.find_mapping(-1, -1)

            def compute_intermediate1(coeff):
                tmp = numpy.zeros((norb,
                                   norb,
                                   int(binom(norb, nalpha-1)),
                                   int(binom(norb, nbeta-1))),
                                  dtype=self._dtype)
                for i in range(norb): #alpha
                    for j in range(norb):
                        for sourcea, targeta, paritya in alpha_map[(i,)]:
                            signa = ((-1) ** (nalpha-1))*paritya
                            for sourceb, targetb, parityb in beta_map[(j,)]:
                                work = coeff[sourcea, sourceb] * signa * parityb
                                tmp[i, j, targeta, targetb] += work
                return tmp

            inter = compute_intermediate1(self.coeff)
            if bradata is None:
                inter2 = inter
            else:
                inter2 = compute_intermediate1(bradata.coeff)

            outunpack += numpy.einsum('ijmn,klmn->ijkl', inter2.conj(), inter)

        if self.nbeta()-2 >= 0:
            _, beta_map = self._core.find_mapping(0, -2)

            def compute_intermediate2(coeff):
                tmp = numpy.zeros((nlt,
                                   lena,
                                   int(binom(norb, nbeta-2))),
                                  dtype=self._dtype)
                for i in range(norb):
                    for j in range(i+1, norb):
                        ijn = i+j*(j+1)//2
                        for source, target, parity in beta_map[(i, j)]:
                            tmp[ijn, :, target] += coeff[:, source] * parity

                return tmp

            inter = compute_intermediate2(self.coeff)
            if bradata is None:
                inter2 = inter
            else:
                inter2 = compute_intermediate2(bradata.coeff)
            outpack += numpy.einsum('imn,kmn->ik', inter2.conj(), inter)

        out = numpy.zeros(outunpack.shape, dtype=self._dtype)
        for i in range(norb):
            for j in range(norb):
                ijn = min(i, j) + max(i, j)*(max(i, j)+1)//2
                for k in range(norb):
                    for l in range(norb):
                        kln = min(k, l)+max(k, l)*(max(k, l)+1)//2
                        out[i, j, k, l] -= (outpack[ijn, kln]
                                            + outunpack[j, i, k, l]
                                            + outunpack[i, j, l, k])

        return (self.rdm1(bradata), out)


    def rdm123(self,
               bradata: 'FqeData' = None,
               dvec: nparray = None,
               dvec2: nparray = None,
               evec2: nparray = None) -> nparray:
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

        def make_evec(current_dvec: nparray) -> nparray:
            current_evec = numpy.zeros((norb,
                                        norb,
                                        norb,
                                        norb,
                                        self.lena(),
                                        self.lenb()),
                                       dtype=self._dtype)
            for i in range(norb):
                for j in range(norb):
                    tmp = current_dvec[i, j, :, :]
                    tmp2 = self.calculate_dvec_spatial_with_coeff(tmp)
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


    def rdm1234(self, bradata: 'FqeData' = None) -> nparray:
        """
        Calculates 1- through 4-particle RDMs given a wave function. When
        bradata is given, it calculates transition RDMs.
        """
        norb = self.norb()
        dvec = self.calculate_dvec_spatial()
        dvec2 = dvec if bradata is None else bradata.calculate_dvec_spatial()

        def make_evec(current_dvec: nparray) -> nparray:
            current_evec = numpy.zeros((norb,
                                        norb,
                                        norb,
                                        norb,
                                        self.lena(),
                                        self.lenb()),
                                       dtype=self._dtype)
            for i in range(norb):
                for j in range(norb):
                    tmp = current_dvec[i, j, :, :]
                    tmp2 = self.calculate_dvec_spatial_with_coeff(tmp)
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


    def calculate_dvec_spatial(self) -> nparray:
        """Generate
            D^{J}_ij = sum_I <J|a^i a_j|I> C_I
        using self.coeff as an input
        """
        return self.calculate_dvec_spatial_with_coeff(self.coeff)


    def calculate_dvec_spin(self) -> Tuple[nparray, nparray]:
        """Generate a pair of
            D^{J}_ij = sum_I <J|a^i a_j|I> C_I
        using self.coeff as an input. Alpha and beta are seperately packed in
        the tuple to be returned
        """
        return self.calculate_dvec_spin_with_coeff(self.coeff)


    def calculate_dvec_spatial_with_coeff(self, coeff: nparray) -> nparray:
        """Generate

            D^{J}_ij = sum_I <J|a^i a_j|I> C_I
        """
        norb = self.norb()
        dvec = numpy.zeros((norb, norb, self.lena(), self.lenb()), dtype=self._dtype)
        for i in range(norb):
            for j in range(norb):
                for source, target, parity in self.alpha_map(i, j):
                    dvec[i, j, target, :] += coeff[source, :] * parity
                for source, target, parity in self.beta_map(i, j):
                    dvec[i, j, :, target] += coeff[:, source] * parity
        return dvec


    def calculate_dvec_spin_with_coeff(self,
                                       coeff: nparray) -> Tuple[nparray, nparray]:
        """Generate

            D^{J}_ij = sum_I <J|a^i a_j|I> C_I

        in the spin-orbital case
        """
        norb = self.norb()
        dveca = numpy.zeros((norb, norb, self.lena(), self.lenb()), dtype=self._dtype)
        dvecb = numpy.zeros((norb, norb, self.lena(), self.lenb()), dtype=self._dtype)
        for i in range(norb):
            for j in range(norb):
                for source, target, parity in self.alpha_map(i, j):
                    dveca[i, j, target, :] += coeff[source, :] * parity
                for source, target, parity in self.beta_map(i, j):
                    dvecb[i, j, :, target] += coeff[:, source] * parity
        return (dveca, dvecb)


    def calculate_coeff_spatial_with_dvec(self, dvec: nparray) -> nparray:
        """Generate
            C_I = sum_J <I|a^i a_j|J>D^{J}_ij
        """
        out = numpy.zeros(self.coeff.shape, dtype=self._dtype)
        for i in range(self.norb()):
            for j in range(self.norb()):
                for source, target, parity in self.alpha_map(j, i):
                    out[source, :] += dvec[i, j, target, :] * parity
                for source, target, parity in self.beta_map(j, i):
                    out[:, source] += dvec[i, j, :, target] * parity
        return out


    def calculate_dvec_spatial_compressed(self) -> nparray:
        """Generate
            D^{J}_i<j = sum_I <J|a^i a_j|I> C_I
        """
        norb = self.norb()
        nlt = norb*(norb+1)//2
        dvec = numpy.zeros((nlt, self.lena(), self.lenb()),
                           dtype=self._dtype)
        for i in range(norb): #creation
            for j in range(norb): #annihilation
                ijn = min(i, j) + max(i, j)*(max(i, j)+1)//2
                for source, target, parity in self.alpha_map(i, j):
                    dvec[ijn, target, :] += self.coeff[source, :] * parity
                for source, target, parity in self.beta_map(i, j):
                    dvec[ijn, :, target] += self.coeff[:, source] * parity
        return dvec


    def calculate_coeff_spin_with_dvec(self,
                                       dvec: Tuple[nparray, nparray]) -> nparray:
        """Generate
            C_I = sum_J <I|a^i a_j|J>D^{J}_ij
        """
        out = numpy.zeros(self.coeff.shape, dtype=self._dtype)
        for i in range(self.norb()):
            for j in range(self.norb()):
                for source, target, parity in self.alpha_map(j, i):
                    out[source, :] += dvec[0][i, j, target, :] * parity
                for source, target, parity in self.beta_map(j, i):
                    out[:, source] += dvec[1][i, j, :, target] * parity
        return out


    def evolve_inplace_individual_nbody_trivial(self,
                                                time: float,
                                                coeff: complex,
                                                opa: List[int],
                                                opb: List[int]) -> None:
        """
        This is the time evolultion code for the cases where individual nbody
        becomes number operators (hence hat{T}^2 is nonzero) coeff includes
        parity due to sorting. opa and opb are integer arrays
        """
        n_a = len(opa)
        n_b = len(opb)
        coeff *= (-1)**(n_a*(n_a-1)//2 + n_b*(n_b-1)//2)

        amap = set()
        bmap = set()
        for index in range(self.lena()):
            current = self._core.string_alpha(index)
            check = True
            for i in opa:
                check &= bool(get_bit(current, i))
            if check:
                amap.add(index)
        for index in range(self.lenb()):
            current = self._core.string_beta(index)
            check = True
            for i in opb:
                check &= bool(get_bit(current, i))
            if check:
                bmap.add(index)

        factor = numpy.exp(-time * numpy.real(coeff) * 2.j)
        for i_a in amap:
            for i_b in bmap:
                self.coeff[i_a, i_b] *= factor


    def evolve_inplace_individual_nbody_nontrivial(self,
                                                   time: float,
                                                   coeff: complex,
                                                   daga: List[int],
                                                   undaga: List[int],
                                                   dagb: List[int],
                                                   undagb: List[int]) -> None:
        """
        This code time-evolves a wave function with an individual n-body
        generator which is spin-conserving. It is assumed that hat{T}^2 = 0.
        Using TT = 0 and TTd is diagonal in the determinant space, one could
        evaluate as

        exp(-i(T+Td)t) = 1 + i(T+Td)t - (TTd + TdT)t^2/2
                         - i(TTdT + TdTTd)t^3/6 + ...

                       = -1 + cos(sqrt(TTd)) + cos(sqrt(TdT))
                            - iT*sin(sqrt(TdT))/sqrt(TdT)
                            - iTd*sin(sqrt(TTd))/sqrt(TTd)
        """
        def isolate_number_operators(dag: List[int],
                                     undag: List[int],
                                     dagwork: List[int],
                                     undagwork: List[int],
                                     number: List[int]) -> int:
            """
            Pair-up daggered and undaggered operators that correspond to the
            same spin-orbital and isolate them, because they have to be treated
            differently.
            """
            par = 0
#            for i in range(len(dag)):
            for current in dag:
#                current = dag[i]
                if current in undag:
                    index1 = dagwork.index(current)
                    index2 = undagwork.index(current)
                    par += len(dagwork)-(index1+1) + index2
                    dagwork.remove(current)
                    undagwork.remove(current)
                    number.append(current)
            return par

        dagworka = copy.deepcopy(daga)
        dagworkb = copy.deepcopy(dagb)
        undagworka = copy.deepcopy(undaga)
        undagworkb = copy.deepcopy(undagb)
        numbera = []
        numberb = []

        parity = 0
        parity += isolate_number_operators(daga,
                                           undaga,
                                           dagworka,
                                           undagworka,
                                           numbera)
        parity += isolate_number_operators(dagb,
                                           undagb,
                                           dagworkb,
                                           undagworkb,
                                           numberb)
        ncoeff = coeff * (-1)**parity

        # code for (TTd)
        phase = (-1)**((len(daga)+len(undaga)) * (len(dagb)+len(undagb)))
        (cosdata1, sindata1) = self.apply_cos_sin(time,
                                                  ncoeff,
                                                  numbera + dagworka,
                                                  undagworka,
                                                  numberb + dagworkb,
                                                  undagworkb)

        work_cof = numpy.conj(coeff)*phase
        cosdata1.ax_plus_y(-1.0j,
                           sindata1.apply_individual_nbody(work_cof,
                                                           undaga,
                                                           daga,
                                                           undagb,
                                                           dagb))
        # code for (TdT)
        (cosdata2, sindata2) = self.apply_cos_sin(time,
                                                  ncoeff,
                                                  numbera + undagworka,
                                                  dagworka,
                                                  numberb + undagworkb,
                                                  dagworkb)
        cosdata2.ax_plus_y(-1.0j, sindata2.apply_individual_nbody(coeff,
                                                                  daga,
                                                                  undaga,
                                                                  dagb,
                                                                  undagb))

        self.coeff = cosdata1.coeff + cosdata2.coeff - self.coeff


    def apply_cos_sin(self,
                      time: float,
                      ncoeff: complex,
                      opa: List[int],
                      oha: List[int],
                      opb: List[int],
                      ohb: List[int]) -> Tuple['FqeData', 'FqeData']:
        """
        Utility internal function that performs part of the operations in
        evolve_inplace_individual_nbody_nontrivial.  Isolated because it is
        also used in the counterpart in FqeDataSet.
        """
        amap = set()
        bmap = set()
        for index in range(self.lena()):
            current = self._core.string_alpha(index)
            check = True
            for i in opa:
                check &= bool(get_bit(current, i))
            for i in oha:
                check &= not bool(get_bit(current, i))
            if check:
                amap.add(index)
        for index in range(self.lenb()):
            current = self._core.string_beta(index)
            check = True
            for i in opb:
                check &= bool(get_bit(current, i))
            for i in ohb:
                check &= not bool(get_bit(current, i))
            if check:
                bmap.add(index)

        absol = numpy.absolute(ncoeff)
        cosfactor = numpy.cos(time * absol)
        sinfactor = numpy.sin(time * absol) / absol

        cosdata = copy.deepcopy(self)
        sindata = copy.deepcopy(self)
        sindata.coeff.fill(0.0)
        for i_a in amap:
            for i_b in bmap:
                cosdata.coeff[i_a, i_b] *= cosfactor
                sindata.coeff[i_a, i_b] = self.coeff[i_a, i_b] * sinfactor
        return (cosdata, sindata)


    def alpha_map(self, iorb: int, jorb: int) -> Tuple[int, int, int]:
        """Access the mapping for a singlet excitation from the current
        sector for alpha orbitals
        """
        return self._core.alpha_map(iorb, jorb)


    def beta_map(self, iorb: int, jorb: int) -> Tuple[int, int, int]:
        """Access the mapping for a singlet excitation from the current
        sector for beta orbitals
        """
        return self._core.beta_map(iorb, jorb)


    def __iadd__(self, other: 'FqeData') -> 'FqeData':
        """Add coefficients together in place
        """
        assert hash(self) == hash(other)
        self.coeff += other.coeff
        return self


    def __add__(self, other: 'FqeData') -> 'FqeData':
        """Add coefficients together and return them
        """
        out = copy.deepcopy(self)
        out += other
        return out


    def __isub__(self, other: 'FqeData') -> 'FqeData':
        """subtract coefficients in place
        """
        assert hash(self) == hash(other)
        self.coeff -= other.coeff
        return self


    def __sub__(self, other: 'FqeData') -> 'FqeData':
        """Subtract coefficients and return them
        """
        out = copy.deepcopy(self)
        out -= other
        return out


    def ax_plus_y(self, sval: complex, other: 'FqeData') -> 'FqeData':
        """Scale and add the data in the fqedata structure

            = sval*coeff + other

        """
        assert hash(self) == hash(other)
        self.coeff += other.coeff * sval
        return self


    def __eq__(self, other: 'FqeData') -> bool:
        """Check the hash first since it can preclude checking every element
        """
        if hash(self) != hash(other):
            return False

        return numpy.allclose(self.coeff, other.coeff)


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


    def index_alpha(self, bit_string: int) -> int:
        """Retrieve the alpha index stored by it's bitstring

        Args:
            address (int) - a bitstring in the fci space

        Returns:
            The fqeindex into the sector for that bitsring
        """
        return self._core.index_alpha(bit_string)


    def index_beta(self, bit_string: int) -> int:
        """Retrieve the beta bitstring reprsentation stored at the address

        Args:
            address (int) - an integer pointing into the fcigraph

        Returns:
            (bistring) - an occupation representation of the configuration
        """
        return self._core.index_beta(bit_string)


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
                return '{}:{}'.format(astr, bstr)

            pformat = print_format

        print('Sector N = {} : S_z = {}'.format(self._nele, self._m_s))
        for inda in range(self._core.lena()):
            alpha_str = self._core.string_alpha(inda)
            for indb in range(self._core.lenb()):
                beta_str = self._core.string_beta(indb)
                if numpy.abs(self.coeff[inda, indb]) > threshold:
                    print('{} {}'.format(pformat(alpha_str, beta_str),
                                         self.coeff[inda, indb]))


    def scale(self, sval: complex):
        """ Scale the wavefunction by the value sval

        Args:
            sval (complex) - value to scale by

        Returns:
            nothing - Modifies the wavefunction in place
        """
        self.coeff = self.coeff.astype(numpy.complex128)*sval


    def fill(self, value: complex):
        """ Fills the wavefunction with the value specified
        """
        self.coeff.fill(value)


    def set_wfn(self, strategy: Optional[str] = None,
                raw_data: nparray = None) -> None:
        """Set the values of the fqedata wavefunction based on a strategy

        Args:
            strategy (string) - the procedure to follow to set the coeffs
            raw_data (numpy.array(dim(self.lena(), self.lenb()),
                                  dtype=numpy.complex128)) - the values to use
                if setting from data.  If vrange is supplied, the first column
                in data will correspond to the first index in vrange

        Returns:
            nothing - modifies the wavefunction in place
        """

        strategy_args = [
            'ones',
            'zero',
            'lowest',
            'random',
            'from_data'
        ]

        if strategy is None and raw_data is None:
            raise ValueError('No strategy and no data passed.'
                             ' Cannot initialize')

        if strategy == 'from_data' and raw_data is None:
            raise ValueError('No data passed to initialize from')

        if raw_data is not None and strategy not in ['from_data', None]:
            raise ValueError('Inconsistent strategy for set_vec passed with'
                             'data')

        if strategy not in strategy_args:
            raise ValueError('Unknown Argument passed to set_vec')

        if strategy == 'from_data':
            chkdim = raw_data.shape
            if chkdim[0] != self.lena() or chkdim[1] != self.lenb():
                raise ValueError('Dim in passed data is wrong')

        if strategy == 'ones':
            self.coeff.fill(1. + .0j)
        elif strategy == 'zero':
            self.coeff.fill(0. + .0j)
        elif strategy == 'lowest':
            self.coeff.fill(0. + .0j)
            self.coeff[0, 0] = 1. + .0j
        elif strategy == 'random':
            self.coeff[:, :] = rand_wfn(self.lena(), self.lenb())
        elif strategy == 'from_data':
            self.coeff = numpy.copy(raw_data)
