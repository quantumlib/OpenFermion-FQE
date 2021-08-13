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
#pylint: disable=dangerous-default-value
import copy
import itertools
from typing import List, Optional, Tuple, Callable, Iterator, \
                   TYPE_CHECKING

import numpy
from scipy.special import binom

from fqe.bitstring import integer_index, get_bit, count_bits_above
from fqe.bitstring import set_bit, unset_bit, reverse_integer_index
from fqe.util import rand_wfn, validate_config
from fqe.fci_graph import FciGraph
from fqe.fci_graph_set import FciGraphSet
import fqe.settings

from fqe.lib.fqe_data import _lm_apply_array1, _make_dvec_part, \
    _make_coeff_part, _make_dvec, _make_coeff, _diagonal_coulomb, \
    _lm_apply_array12_same_spin_opt, _lm_apply_array12_diff_spin_opt, \
    _apply_array12_lowfillingab, _apply_array12_lowfillingab2, \
    _apply_array12_lowfillingaa, _apply_array12_lowfillingaa2, \
    _apply_individual_nbody1_accumulate, _sparse_scale, \
    _evaluate_map_each, _make_Hcomp, \
    _sparse_apply_array1, _lm_apply_array1_alpha_column, \
    _apply_diagonal_inplace, _evolve_diagonal_inplace, _make_nh123, \
    _apply_diagonal_coulomb
from fqe.lib.linalg import _zimatadd, _transpose

if TYPE_CHECKING:
    from numpy import ndarray as Nparray
    from numpy import dtype as Dtype
    from fqe.fqe_data_set import FqeDataSet


class FqeData:
    """This is a basic data structure for use in the FQE.
    """

    def __init__(self,
                 nalpha: int,
                 nbeta: int,
                 norb: int,
                 fcigraph: Optional[FciGraph] = None,
                 dtype: 'Dtype' = numpy.complex128) -> None:
        """The FqeData structure holds the wavefunction for a particular
        configuration and provides an interace for accessing the data through
        the fcigraph functionality.

        Args:
            nalpha (int): the number of alpha electrons

            nbeta (int): the number of beta electrons

            norb (int): the number of spatial orbitals

            fcigraph (optional, FciGraph): the FciGraph to be used. When None, \
                it is computed here

            dtype (optional, Dtype): numpy.dtype of the underlying array
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

        if fqe.settings.use_accelerated_code:
            # Use the same C extension for both cases by default
            self._low_thresh = 0.0
        else:
            self._low_thresh = 0.3
        self._nele = self.nalpha() + self.nbeta()
        self._m_s = self.nalpha() - self.nbeta()
        self.coeff = numpy.zeros((self.lena(), self.lenb()), dtype=self._dtype)

    def __getitem__(self, key: Tuple[int, int]) -> complex:
        """Get an item from the fqe data structure by using the knowles-handy
        pointers.

        Args:
            key (Tuple[int, int]): a pair of alpha and beta strings

        Returns:
            complex: the value of the corresponding element
        """
        return self.coeff[self._core.index_alpha(key[0]),
                          self._core.index_beta(key[1])]

    def __setitem__(self, key: Tuple[int, int], value: complex) -> None:
        """Set an element in the fqe data strucuture

        Args:
            key (Tuple[int, int]): a pair of alpha and beta strings

            value: the value to be set
        """
        self.coeff[self._core.index_alpha(key[0]),
                   self._core.index_beta(key[1])] = value

    def __deepcopy__(self, memodict={}) -> 'FqeData':
        """Construct new FqeData that has the same coefficient

        Returns:
            FqeData: an object that is deepcopied from self
        """
        new_data = FqeData(nalpha=self.nalpha(),
                           nbeta=self.nbeta(),
                           norb=self._core.norb(),
                           fcigraph=self._core,
                           dtype=self._dtype)
        new_data._low_thresh = self._low_thresh
        new_data.coeff = self.coeff.copy()
        return new_data

    def get_fcigraph(self) -> 'FciGraph':
        """
        Returns the underlying FciGraph object

        Returns:
            FciGraph: underlying FciGraph object for this object
        """
        return self._core

    def apply_diagonal_inplace(self, array: 'Nparray') -> None:
        """Iterate over each element and perform apply operation in place

        Args:
            array (Nparray): a diagonal operator to be applied to self. The size \
                of this array is norb or 2*norb depending on the context
        """
        beta_ptr = 0

        if array.size == 2 * self.norb():
            beta_ptr = self.norb()

        elif array.size != self.norb():
            raise ValueError('Non-diagonal array passed'
                             ' into apply_diagonal_inplace')

        if not array.flags['C_CONTIGUOUS']:
            array = numpy.copy(array)

        if fqe.settings.use_accelerated_code:
            aarray = array[:self.norb()]
            barray = array[beta_ptr:]
            _apply_diagonal_inplace(self.coeff, aarray, barray,
                                    self._core.string_alpha_all(),
                                    self._core.string_beta_all())
        else:
            alpha = numpy.zeros((self._core.lena(),), dtype=numpy.complex128)
            beta = numpy.zeros((self._core.lenb(),), dtype=numpy.complex128)

            for alp_cnf in range(self._core.lena()):
                occupation = self._core.string_alpha(alp_cnf)
                diag_ele = 0.0
                for ind in integer_index(occupation):
                    diag_ele += array[ind]
                alpha[alp_cnf] = diag_ele
            for bet_cnf in range(self._core.lenb()):
                occupation = self._core.string_beta(bet_cnf)
                diag_ele = 0.0
                for ind in integer_index(occupation):
                    diag_ele += array[beta_ptr + ind]
                beta[bet_cnf] = diag_ele

            for alp_cnf in range(self._core.lena()):
                for bet_cnf in range(self._core.lenb()):
                    self.coeff[alp_cnf,
                               bet_cnf] *= alpha[alp_cnf] + beta[bet_cnf]

    def evolve_diagonal(self, array: 'Nparray',
                        inplace: bool = False) -> 'Nparray':
        """Iterate over each element and return the exponential scaled
        contribution.

        Args:
            array (Nparray): a diagonal operator using which time evolution is \
                performed. The size of this array is norb or 2*norb depending \
                on the context

            inplace (bool): toggle to specify if the result will be stored \
                in-place or out-of-place

        Returns:
            Nparray: the numpy array that contains the result. If inplace is \
                True, self.coeff is returned.
        """
        beta_ptr = 0

        if array.size == 2 * self.norb():
            beta_ptr = self.norb()

        elif array.size != self.norb():
            raise ValueError('Non-diagonal array passed into evolve_diagonal')

        if inplace:
            data = self.coeff
        else:
            data = numpy.copy(self.coeff).astype(numpy.complex128)

        if not array.flags['C_CONTIGUOUS']:
            array = numpy.copy(array)

        if fqe.settings.use_accelerated_code:
            aarray = array[:self.norb()]
            barray = array[beta_ptr:]
            _evolve_diagonal_inplace(data, aarray, barray,
                                     self._core.string_alpha_all(),
                                     self._core.string_beta_all())
        else:
            for alp_cnf in range(self._core.lena()):
                occupation = self._core.string_alpha(alp_cnf)
                diag_ele = 0.0
                for ind in integer_index(self._core.string_alpha(alp_cnf)):
                    diag_ele += array[ind]

                if diag_ele != 0.0:
                    data[alp_cnf, :] *= numpy.exp(diag_ele)

            for bet_cnf in range(self._core.lenb()):
                occupation = self._core.string_beta(bet_cnf)
                diag_ele = 0.0
                for ind in integer_index(occupation):
                    diag_ele += array[beta_ptr + ind]

                if diag_ele:
                    data[:, bet_cnf] *= numpy.exp(diag_ele)

        return data

    def apply_diagonal_coulomb(self,
                               diag: 'Nparray',
                               array: 'Nparray',
                               inplace: bool = False) -> 'Nparray':
        """Apply a diagonal Coulomb Hamiltonian represented by the arrays
        and return the resulting coefficient.

        Args:
            diag: one-body part of the diagonal elements in the 1-D format

            array: two-body part of the diagonal elements in the 2-D format. \
                The elements correspond to the coefficient for `n_i n_j`

            inplace (bool): toggle to specify if the result will be stored \
                in-place or out-of-place

        Returns:
            Nparray: the numpy array that contains the result. If inplace is \
                True, self.coeff is returned.
        """
        if inplace:
            data = self.coeff
        else:
            data = numpy.copy(self.coeff)

        if fqe.settings.use_accelerated_code:
            _apply_diagonal_coulomb(data, self._core.string_alpha_all(),
                                    self._core.string_beta_all(), diag, array,
                                    self.lena(), self.lenb(), self.nalpha(),
                                    self.nbeta(), self.norb())
        else:
            alpha = numpy.zeros((self._core.lena(),), dtype=numpy.complex128)
            beta = numpy.zeros((self._core.lenb(),), dtype=numpy.complex128)

            for alp_cnf in range(self._core.lena()):
                occupation = self._core.string_alpha(alp_cnf)
                diag_ele = 0.0
                for ind in integer_index(occupation):
                    diag_ele += diag[ind]
                    for jnd in integer_index(occupation):
                        diag_ele += array[ind, jnd]
                alpha[alp_cnf] = diag_ele

            for bet_cnf in range(self._core.lenb()):
                occupation = self._core.string_beta(bet_cnf)
                diag_ele = 0.0
                for ind in integer_index(occupation):
                    diag_ele += diag[ind]
                    for jnd in integer_index(occupation):
                        diag_ele += array[ind, jnd]
                beta[bet_cnf] = diag_ele

            aarrays = numpy.empty((array.shape[1],), dtype=array.dtype)
            for alp_cnf in range(self._core.lena()):
                aoccs = self._core.string_alpha(alp_cnf)
                aarrays[:] = 0.0
                for ind in integer_index(aoccs):
                    aarrays[:] += array[ind, :]
                    aarrays[:] += array[:, ind]
                for bet_cnf in range(self._core.lenb()):
                    ab = 0.0
                    boccs = self._core.string_beta(bet_cnf)
                    for jnd in integer_index(boccs):
                        ab += aarrays[jnd]
                    data[alp_cnf, bet_cnf] *= (ab + alpha[alp_cnf] +
                                               beta[bet_cnf])

        return data

    def evolve_diagonal_coulomb(self,
                                diag: 'Nparray',
                                array: 'Nparray',
                                inplace: bool = False) -> 'Nparray':
        """Perform time evolution using a diagonal Coulomb Hamiltonian represented
        by the input arrays and return the resulting coefficient.

        Args:
            diag: one-body part of the diagonal elements in the 1-D format

            array: two-body part of the diagonal elements in the 2-D format. \
                The elements correspond to the coefficient for `n_i n_j`

            inplace (bool): toggle to specify if the result will be stored \
                in-place or out-of-place

        Returns:
            Nparray: the numpy array that contains the result. If inplace is \
                True, self.coeff is returned.
        """
        if inplace:
            data = self.coeff
        else:
            data = numpy.copy(self.coeff)

        if fqe.settings.use_accelerated_code:
            _diagonal_coulomb(data, self._core.string_alpha_all(),
                              self._core.string_beta_all(), diag, array,
                              self.lena(), self.lenb(), self.nalpha(),
                              self.nbeta(), self.norb())
        else:
            diagexp = numpy.exp(diag)
            arrayexp = numpy.exp(array)

            alpha_occ = numpy.zeros((self.lena(), self.nalpha()), dtype=int)
            alpha_diag = numpy.zeros((self.lena(),), dtype=numpy.complex128)
            for a, astring in enumerate(self._core.string_alpha_all()):
                occ = integer_index(astring)
                alpha_occ[a, :] = occ
                diag_ele = 1.0
                for ind in occ:
                    diag_ele *= diagexp[ind]
                    for jnd in occ:
                        diag_ele *= arrayexp[ind, jnd]
                alpha_diag[a] = diag_ele

            beta_occ = numpy.zeros((self.lenb(), self.nbeta()), dtype=int)
            beta_diag = numpy.zeros((self.lenb(),), dtype=numpy.complex128)
            for b, bstring in enumerate(self._core.string_beta_all()):
                occ = integer_index(bstring)
                beta_occ[b, :] = occ
                diag_ele = 1.0
                for ind in occ:
                    diag_ele *= diagexp[ind]
                    for jnd in occ:
                        diag_ele *= arrayexp[ind, jnd]
                beta_diag[b] = diag_ele

            aarrays = numpy.empty((array.shape[1],), dtype=array.dtype)
            for a in range(self.lena()):
                aarrays[:] = 1.0
                for ind in alpha_occ[a]:
                    aarrays[:] *= arrayexp[ind, :]
                for b in range(self.lenb()):
                    diag_ele = 1.0
                    for ind in beta_occ[b]:
                        diag_ele *= aarrays[ind]
                    data[a, b] *= diag_ele * diag_ele * alpha_diag[
                        a] * beta_diag[b]

        return data

    def apply(self, array: Tuple['Nparray']) -> 'FqeData':
        """
        API for application of dense operators (1- through 4-body operators) to
        the wavefunction self. The result is stored out of place and returned.

        Args:
            array: (Tuple[Nparray]): numpy arrays that represent 1- through 4-body \
                Hamiltonian elements

        Returns:
            FqeData: the FqeData that contains the result of the Hamiltonian application
        """

        out = copy.deepcopy(self)
        out.apply_inplace(array)
        return out

    def apply_inplace(self, array: Tuple['Nparray', ...]) -> None:
        """
        API for application of dense operators (1- through 4-body operators) to
        the wavefunction self. The result is stored in-place.

        Args:
            array: (Tuple[Nparray]): numpy arrays that represent 1- through 4-body \
                Hamiltonian elements
        """

        len_arr = len(array)
        if (len_arr < 1 or len_arr > 4):
            raise ValueError("Number of operators in tuple must be "
                             "between 1 and 4.")

        # Get the first numpy array (i.e. a non-None object) in array
        # and use its dimensions to determine whether we have spatial or spin orbitals
        # Necessary since 1-body operator cam be absent
        array_for_dimensions = next(
            filter(lambda x: isinstance(x, numpy.ndarray), array), False)

        if isinstance(array_for_dimensions, bool):
            # this can only be False
            assert (not array_for_dimensions)
            return

        spatial = array_for_dimensions.shape[0] == self.norb()
        ## check correct dimensions in case of spin orbitals
        if not spatial and array_for_dimensions.shape[0] != 2 * self.norb():
            raise ValueError("Inconsistent number of spin-orbitals in "
                             "operators and wavefunction.")
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

        # Check if only one column of h1e is non-zero
        ncol = 0
        jorb = 0
        for j in range(self.norb()):
            if numpy.any(h1e[:, j]):
                ncol += 1
                jorb = j
            if ncol > 1:
                break

        def dense_apply_array_spatial1(self, h1e):
            out = numpy.zeros(self.coeff.shape, dtype=self._dtype)
            out_b = numpy.zeros(self.coeff.shape[::-1], dtype=self._dtype)
            cT = numpy.empty(self.coeff.shape[::-1], dtype=self._dtype)
            _transpose(cT, self.coeff)

            _lm_apply_array1(self.coeff,
                             h1e,
                             self._core._dexca,
                             self.lena(),
                             self.lenb(),
                             self.norb(),
                             True,
                             out=out)
            _lm_apply_array1(cT,
                             h1e,
                             self._core._dexcb,
                             self.lenb(),
                             self.lena(),
                             self.norb(),
                             True,
                             out=out_b)
            _zimatadd(out, out_b, 1.)
            return out

        if fqe.settings.use_accelerated_code:
            out = dense_apply_array_spatial1(self, h1e)
        else:
            if ncol > 1:
                dvec = self.calculate_dvec_spatial()
                out = numpy.tensordot(h1e, dvec, axes=((0, 1), (0, 1)))
            else:
                dvec = self.calculate_dvec_spatial_fixed_j(jorb)
                out = numpy.tensordot(h1e[:, jorb], dvec, axes=1)

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

        def dense_apply_array_spin1_lm(self, h1e):
            out = _lm_apply_array1(self.coeff, h1e[:norb, :norb],
                                   self._core._dexca, self.lena(), self.lenb(),
                                   self.norb(), True)
            _lm_apply_array1(self.coeff,
                             h1e[norb:, norb:],
                             self._core._dexcb,
                             self.lena(),
                             self.lenb(),
                             self.norb(),
                             False,
                             out=out)
            return out

        if fqe.settings.use_accelerated_code:
            out = dense_apply_array_spin1_lm(self, h1e)
        else:
            if ncol > 1:
                (dveca, dvecb) = self.calculate_dvec_spin()
                out = numpy.tensordot(h1e[:norb, :norb], dveca) \
                    + numpy.tensordot(h1e[norb:, norb:], dvecb)
            else:
                dvec = self.calculate_dvec_spin_fixed_j(jorb)
                if jorb < norb:
                    h1eview = h1e[:norb, jorb]
                else:
                    h1eview = h1e[norb:, jorb]
                out = numpy.tensordot(h1eview, dvec, axes=1)

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
        if fqe.settings.use_accelerated_code:
            return self._apply_array_spatial12_lm(h1e, h2e)
        else:
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
                dvec = self._calculate_dvec_spatial_compressed()
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

    def _apply_array_spatial12_lm(self, h1e: 'Nparray',
                                  h2e: 'Nparray') -> 'Nparray':
        """
        Low-memory version to apply_array_spatial12.
        No construction of dvec.
        """
        h1e = copy.deepcopy(h1e)
        h2e = numpy.moveaxis(copy.deepcopy(h2e), 1, 2) * (-1.0)
        h1e -= numpy.einsum('ikkj->ij', h2e)

        out = _lm_apply_array12_same_spin_opt(self.coeff, h1e, h2e,
                                              self._core._dexca, self.lena(),
                                              self.lenb(), self.norb())
        out += _lm_apply_array12_same_spin_opt(self.coeff.T, h1e, h2e,
                                               self._core._dexcb, self.lenb(),
                                               self.lena(), self.norb()).T

        _lm_apply_array12_diff_spin_opt(self.coeff,
                                        h2e + numpy.einsum('ijkl->klij', h2e),
                                        self._core._dexca,
                                        self._core._dexcb,
                                        self.lena(),
                                        self.lenb(),
                                        self.norb(),
                                        out=out)
        return out

    def _apply_array_spin12_halffilling(self, h1e: 'Nparray',
                                        h2e: 'Nparray') -> 'Nparray':
        """
        Standard code to calculate application of 1- and 2-body spin-orbital
        operators to the wavefunction self. It returns numpy.ndarray that
        corresponds to the output wave function data.
        """
        if fqe.settings.use_accelerated_code:
            #return self._apply_array_spin12_blocked(h1e, h2e)
            return self._apply_array_spin12_lm(h1e, h2e)
        else:
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
            out += self._calculate_coeff_spin_with_dvec((ndveca, ndvecb))
            return out

    def _apply_array_spin12_lm(self, h1e: 'Nparray',
                               h2e: 'Nparray') -> 'Nparray':
        """
        Low-memory version to apply_array_spin12.
        No construction of dvec.
        """
        h1e = copy.deepcopy(h1e)
        h2e = numpy.moveaxis(copy.deepcopy(h2e), 1, 2) * (-1.0)
        norb = self.norb()
        h1e -= numpy.einsum('ikkj->ij', h2e)

        out = _lm_apply_array12_same_spin_opt(self.coeff, h1e[:norb, :norb],
                                              h2e[:norb, :norb, :norb, :norb],
                                              self._core._dexca, self.lena(),
                                              self.lenb(), self.norb())
        out += _lm_apply_array12_same_spin_opt(self.coeff.T, h1e[norb:, norb:],
                                               h2e[norb:, norb:, norb:, norb:],
                                               self._core._dexcb, self.lenb(),
                                               self.lena(), self.norb()).T

        h2e_c = h2e[:norb, :norb, norb:, norb:] \
            + numpy.einsum('ijkl->klij', h2e[norb:, norb:, :norb, :norb])
        _lm_apply_array12_diff_spin_opt(self.coeff,
                                        h2e_c,
                                        self._core._dexca,
                                        self._core._dexcb,
                                        self.lena(),
                                        self.lenb(),
                                        self.norb(),
                                        out=out)
        return out

    def _apply_array_spatial12_lowfilling(self, h1e: 'Nparray',
                                          h2e: 'Nparray') -> 'Nparray':
        """
        Low-filling specialization of the code to calculate application of
        1- and 2-body spatial operators to the wavefunction self.  It returns
        numpy.ndarray that corresponds to the output wave function data.
        Wrapper to distinguish between C and Python functions
        """
        if fqe.settings.use_accelerated_code:
            return self._apply_array_spatial12_lowfilling_fast(h1e, h2e)
        else:
            return self._apply_array_spatial12_lowfilling_python(h1e, h2e)

    def _apply_array_spatial12_lowfilling_fast(self, h1e: 'Nparray',
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
        h2etemp = numpy.ascontiguousarray(h2e)
        _make_Hcomp(norb, nlt, h2etemp, h2ecomp)

        if nalpha - 2 >= 0:
            alpha_map, _ = self._core.find_mapping(-2, 0)
            alpha_array = self._to_array1(alpha_map, norb)
            intermediate = numpy.zeros(
                (nlt, int(binom(norb, nalpha - 2)), lenb), dtype=self._dtype)
            _apply_array12_lowfillingaa(self.coeff, alpha_array, intermediate)

            intermediate = numpy.tensordot(h2ecomp, intermediate, axes=1)

            _apply_array12_lowfillingaa2(intermediate, alpha_array, out)

        if self.nalpha() - 1 >= 0 and self.nbeta() - 1 >= 0:
            alpha_map, beta_map = self._core.find_mapping(-1, -1)
            nastates = int(binom(norb, nalpha - 1))
            nbstates = int(binom(norb, nbeta - 1))
            intermediate = numpy.zeros((norb, norb, nastates, nbstates),
                                       dtype=self._dtype)

            alpha_array = self._to_array2(alpha_map, norb)
            beta_array = self._to_array2(beta_map, norb)
            _apply_array12_lowfillingab(self.coeff, alpha_array, beta_array,
                                        nalpha, nbeta, intermediate)
            intermediate = numpy.tensordot(h2e, intermediate, axes=2)
            _apply_array12_lowfillingab2(alpha_array, beta_array, nalpha, nbeta,
                                         intermediate, out)

        if self.nbeta() - 2 >= 0:
            _, beta_map = self._core.find_mapping(0, -2)
            beta_array = self._to_array1(beta_map, norb)
            intermediate = numpy.zeros((nlt, lena, int(binom(norb, nbeta - 2))),
                                       dtype=self._dtype)
            _apply_array12_lowfillingaa(self.coeff,
                                        beta_array,
                                        intermediate,
                                        alpha=False)

            intermediate = numpy.tensordot(h2ecomp, intermediate, axes=1)

            _apply_array12_lowfillingaa2(intermediate,
                                         beta_array,
                                         out,
                                         alpha=False)
        return out

    def _to_array1(self, maps, norb):
        """Convert maps to arrays for passing to C code
        """
        nstate = len(maps[(0, 1)])
        nlt = norb * (norb + 1) // 2
        arrays = numpy.zeros((nlt, nstate, 3), dtype=numpy.int32)
        for i in range(norb):
            for j in range(i + 1, norb):
                ijn = i + j * (j + 1) // 2
                for k, data in enumerate(maps[(i, j)]):
                    arrays[ijn, k, 0] = data[0]
                    arrays[ijn, k, 1] = data[1]
                    arrays[ijn, k, 2] = data[2]
        return arrays

    def _to_array2(self, maps, norb):
        """Convert maps to arrays for passing to C code
        """
        nstate = len(maps[(0,)])
        arrays = numpy.zeros((norb, nstate, 3), dtype=numpy.int32)
        for i in range(norb):
            for k, data in enumerate(maps[(i,)]):
                arrays[i, k, 0] = data[0]
                arrays[i, k, 1] = data[1]
                arrays[i, k, 2] = data[2]
        return arrays

    def _apply_array_spatial12_lowfilling_python(self, h1e: 'Nparray',
                                                 h2e: 'Nparray') -> 'Nparray':
        """
        Low-filling specialization of the code to calculate application of
        1- and 2-body spatial operators to the wavefunction self.  It returns
        numpy.ndarray that corresponds to the output wave function data.
        Python version
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

            intermediate = numpy.tensordot(h2ecomp, intermediate, axes=1)

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

            intermediate = numpy.tensordot(h2e, intermediate, axes=2)

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

            intermediate = numpy.tensordot(h2ecomp, intermediate, axes=1)

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
        if fqe.settings.use_accelerated_code:
            return self._apply_array_spin12_lowfilling_fast(h1e, h2e)
        else:
            return self._apply_array_spin12_lowfilling_python(h1e, h2e)

    def _apply_array_spin12_lowfilling_fast(self, h1e: 'Nparray',
                                            h2e: 'Nparray') -> 'Nparray':
        """
        Low-filling specialization of the code to calculate application of
        1- and 2-body spin-orbital operators to the wavefunction self. It
        returns numpy.ndarray that corresponds to the output wave function data.
        Accelerated C version
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
            alpha_array = self._to_array1(alpha_map, norb)
            intermediate = numpy.zeros(
                (nlt, int(binom(norb, nalpha - 2)), lenb), dtype=self._dtype)
            _apply_array12_lowfillingaa(self.coeff, alpha_array, intermediate)

            intermediate = numpy.tensordot(h2ecompa, intermediate, axes=1)
            _apply_array12_lowfillingaa2(intermediate, alpha_array, out)

        if self.nalpha() - 1 >= 0 and self.nbeta() - 1 >= 0:
            alpha_map, beta_map = self._core.find_mapping(-1, -1)
            intermediate = numpy.zeros((norb, norb, int(binom(
                norb, nalpha - 1)), int(binom(norb, nbeta - 1))),
                                       dtype=self._dtype)

            alpha_array = self._to_array2(alpha_map, norb)
            beta_array = self._to_array2(beta_map, norb)
            _apply_array12_lowfillingab(self.coeff, alpha_array, beta_array,
                                        nalpha, nbeta, intermediate)
            intermediate = numpy.tensordot(h2e[:norb, norb:, :norb, norb:],
                                           intermediate,
                                           axes=2)
            _apply_array12_lowfillingab2(alpha_array, beta_array, nalpha, nbeta,
                                         intermediate, out)

        if self.nbeta() - 2 >= 0:
            _, beta_map = self._core.find_mapping(0, -2)
            beta_array = self._to_array1(beta_map, norb)
            intermediate = numpy.zeros((nlt, lena, int(binom(norb, nbeta - 2))),
                                       dtype=self._dtype)
            _apply_array12_lowfillingaa(self.coeff,
                                        beta_array,
                                        intermediate,
                                        alpha=False)

            intermediate = numpy.tensordot(h2ecompb, intermediate, axes=1)
            _apply_array12_lowfillingaa2(intermediate,
                                         beta_array,
                                         out,
                                         alpha=False)
        return out

    def _apply_array_spin12_lowfilling_python(self, h1e: 'Nparray',
                                              h2e: 'Nparray') -> 'Nparray':
        """
        Low-filling specialization of the code to calculate application of
        1- and 2-body spin-orbital operators to the wavefunction self. It
        returns numpy.ndarray that corresponds to the output wave function data.
        Python version
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

            intermediate = numpy.tensordot(h2ecompa, intermediate, axes=1)

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

            intermediate = numpy.tensordot(h2e[:norb, norb:, :norb, norb:],
                                           intermediate,
                                           axes=2)

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

            intermediate = numpy.tensordot(h2ecompb, intermediate, axes=1)

            for i in range(norb):
                for j in range(i + 1, norb):
                    ijn = i + j * (j + 1) // 2
                    for source, target, sign in beta_map[(min(i, j), max(i,
                                                                         j))]:
                        out[:, source] -= intermediate[ijn, :, target] * sign
        return out

    def _apply_array_spatial123(self,
                                h1e: Optional['Nparray'],
                                h2e: Optional['Nparray'],
                                h3e: 'Nparray',
                                dvec: Optional['Nparray'] = None,
                                evec: Optional['Nparray'] = None) -> 'Nparray':
        """
        Code to calculate application of 1- through 3-body spatial operators to
        the wavefunction self. It returns numpy.ndarray that corresponds to the
        output wave function data.
        """
        norb = self.norb()
        assert h3e.shape == (norb, norb, norb, norb, norb, norb)

        out = None
        if h1e is not None and h2e is not None:
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
            odvec = self.calculate_dvec_spatial()
        else:
            odvec = dvec

        if evec is None:
            dvec = numpy.zeros_like(odvec)
            for i in range(norb):
                for j in range(norb):
                    tmp = odvec[i, j, :, :]
                    tmp2 = self._calculate_dvec_spatial_with_coeff(tmp)
                    dvec += numpy.tensordot(h3e[:, :, i, :, :, j],
                                            tmp2,
                                            axes=((1, 3), (0, 1)))
        else:
            dvec = numpy.tensordot(h3e, evec, axes=((1, 4, 2, 5), (0, 1, 2, 3)))

        if out is not None:
            out -= self._calculate_coeff_spatial_with_dvec(dvec)
        else:
            out = -self._calculate_coeff_spatial_with_dvec(dvec)
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
        assert h3e.shape == (norb * 2,) * 6
        assert not (dvec is None) ^ (evec is None)

        from1234 = (dvec is not None) and (evec is not None)

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

        n = norb  # This is just shorter
        if not from1234:
            symfac = 2.0
            axes = ((1, 3), (0, 1))
            (odveca, odvecb) = self.calculate_dvec_spin()
            dveca = numpy.zeros_like(odveca)
            dvecb = numpy.zeros_like(odvecb)

            for i in range(norb):
                for j in range(norb):
                    evecaa, _ = self._calculate_dvec_spin_with_coeff(
                        odveca[i, j, :, :])
                    evecab, evecbb = self._calculate_dvec_spin_with_coeff(
                        odvecb[i, j, :, :])

                    dveca += numpy.tensordot(h3e[:n, :n, i, :n, :n, j],
                                             evecaa,
                                             axes=axes)
                    dveca += numpy.tensordot(h3e[:n, :n, n + i, :n, :n, n + j],
                                             evecab,
                                             axes=axes) * symfac
                    dveca += numpy.tensordot(h3e[:n, n:, n + i, :n, n:, n + j],
                                             evecbb,
                                             axes=axes)

                    dvecb += numpy.tensordot(h3e[n:, :n, i, n:, :n, j],
                                             evecaa,
                                             axes=axes)
                    dvecb += numpy.tensordot(h3e[n:, :n, n + i, n:, :n, n + j],
                                             evecab,
                                             axes=axes) * symfac
                    dvecb += numpy.tensordot(h3e[:n, n:, n + i, :n, n:, n + j],
                                             evecbb,
                                             axes=axes)
        else:
            symfac = 1.0
            axes = ((1, 4, 2, 5), (0, 1, 2, 3))  # type: ignore
            dveca, dvecb = dvec  # type: ignore
            evecaa, evecab, evecba, evecbb = evec  # type: ignore

            dveca = numpy.tensordot(h3e[:n, :n, :n, :n, :n, :n],
                                    evecaa, axes=axes) \
                + numpy.tensordot(h3e[:n, :n, n:, :n, :n, n:],
                                  evecab, axes=axes) * symfac \
                + numpy.tensordot(h3e[:n, n:, n:, :n, n:, n:],
                                  evecbb, axes=axes) + \
                + numpy.tensordot(h3e[:n, n:, :n, :n, n:, :n],
                                  evecba, axes=axes)

            dvecb = numpy.tensordot(h3e[n:, :n, :n, n:, :n, :n],
                                    evecaa, axes=axes) \
                + numpy.tensordot(h3e[n:, :n, n:, n:, :n, n:],
                                  evecab, axes=axes) * symfac \
                + numpy.tensordot(h3e[n:, n:, n:, n:, n:, n:],
                                  evecbb, axes=axes) + \
                + numpy.tensordot(h3e[n:, n:, :n, n:, n:, :n],
                                  evecba, axes=axes)

        out -= self._calculate_coeff_spin_with_dvec((dveca, dvecb))
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

        evec = numpy.transpose(numpy.tensordot(h4e,
                                               evec,
                                               axes=((2, 6, 3, 7), (0, 1, 2,
                                                                    3))),
                               axes=[0, 2, 1, 3, 4, 5])

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

        if fqe.settings.use_accelerated_code:
            _make_nh123(norb, h4e, nh1e, nh2e, nh3e)
        else:
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
                            nh3e[i, j, k, :, :, :] += (
                                h4e[k, i, j, l, l, :, :, :] +
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

        def ncon(A, B):
            """Tensor contraction and transposition corresponding with
            einsum 'ikmojlnp,mnopxy->ijklxy'
            """
            return numpy.transpose(numpy.tensordot(A,
                                                   B,
                                                   axes=((2, 6, 3, 7), (0, 1, 2,
                                                                        3))),
                                   axes=(0, 2, 1, 3, 4, 5))

        n = norb  # shorter
        nevecaa = ncon(h4e[:n, :n, :n, :n, :n, :n, :n, :n], evecaa) \
            + 2.0 * ncon(h4e[:n, :n, :n, n:, :n, :n, :n, n:], evecab) \
            + ncon(h4e[:n, :n, n:, n:, :n, :n, n:, n:], evecbb)

        nevecab = ncon(h4e[:n, n:, :n, :n, :n, n:, :n, :n], evecaa) \
            + 2.0 * ncon(h4e[:n, n:, :n, n:, :n, n:, :n, n:], evecab) \
            + ncon(h4e[:n, n:, n:, n:, :n, n:, n:, n:], evecbb)

        nevecbb = ncon(h4e[n:, n:, :n, :n, n:, n:, :n, :n], evecaa) \
            + 2.0 * ncon(h4e[n:, n:, :n, n:, n:, n:, :n, n:], evecab) \
            + ncon(h4e[n:, n:, n:, n:, n:, n:, n:, n:], evecbb)

        dveca2 = numpy.zeros(dveca.shape, dtype=self._dtype)
        dvecb2 = numpy.zeros(dvecb.shape, dtype=self._dtype)
        for i in range(norb):
            for j in range(norb):
                dveca[:, :, :, :] = nevecaa[i, j, :, :, :, :]
                dvecb[:, :, :, :] = nevecab[i, j, :, :, :, :]
                cvec = self._calculate_coeff_spin_with_dvec((dveca, dvecb))
                dveca2[i, j, :, :] += cvec[:, :]

                dveca[:, :, :, :] = nevecab[:, :, i, j, :, :]
                dvecb[:, :, :, :] = nevecbb[i, j, :, :, :, :]
                cvec = self._calculate_coeff_spin_with_dvec((dveca, dvecb))
                dvecb2[i, j, :, :] += cvec[:, :]

        out += self._calculate_coeff_spin_with_dvec((dveca2, dvecb2))
        return out

    def _apply_columns_recursive_alpha(self, mat: 'Nparray', buf: 'Nparray'):
        """
        Apply only alpha-alpha operator that is represented by mat, whose
        dimension is norb times norb
        """
        norb = self.norb()
        matT = mat.T.copy()
        index, exc, diag = self._core._map_to_deexc_alpha_icol()

        if fqe.settings.use_accelerated_code:
            for icol in range(norb):
                _lm_apply_array1_alpha_column(self.coeff, matT[icol, :],
                                              index[icol], exc[icol],
                                              diag[icol], self.lena(),
                                              self.lenb(), icol)
        else:
            na, ne = exc.shape[1:3]
            na2 = diag.shape[1]
            for icol in range(norb):
                for a in range(na):
                    target = index[icol, a]
                    for e in range(ne):
                        source, ishift, parity = exc[icol, a, e]
                        self.coeff[target, :] += parity * matT[
                            icol, ishift] * self.coeff[source, :]
                for a2 in range(na2):
                    target = diag[icol, a2]
                    self.coeff[target, :] *= (1 + matT[icol, icol])

    def apply_columns_recursive_inplace(self, mat1: 'Nparray',
                                        mat2: 'Nparray') -> None:
        """
        Apply column operators recursively to perform wave function transformation
        under the unitary transfomration of the orbitals. Only called from
        Wavefunction.transform. The results are stored in-place.

        Args:
            mat1 (Nparray): the alpha-alpha part of the transformation matrix

            mat2 (Nparray): the beta-beta part of the transformation matrix
        """
        trans = FqeData(self.nbeta(),
                        self.nalpha(),
                        self.norb(),
                        self._core.alpha_beta_transpose(),
                        dtype=self.coeff.dtype)
        buf = trans.coeff.reshape(self.lena(), self.lenb())
        self._apply_columns_recursive_alpha(mat1, buf)

        if fqe.settings.use_accelerated_code:
            _transpose(trans.coeff, self.coeff)
        else:
            trans.coeff[:, :] = self.coeff.T[:, :]
        buf = self.coeff.reshape(self.lenb(), self.lena())
        trans._apply_columns_recursive_alpha(mat2, buf)

        if fqe.settings.use_accelerated_code:
            _transpose(self.coeff, trans.coeff)
        else:
            self.coeff[:, :] = trans.coeff.T[:, :]

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
        It is assumed that the operator is spin conserving.

        Args:
            coeff (complex): scalar coefficient to be multiplied to the result

            daga (List[int]): indices corresponding to the alpha creation \
                operators in the Hamiltonian

            undaga (List[int]): indices corresponding to the alpha annihilation \
                operators in the Hamiltonian

            dagb (List[int]): indices corresponding to the beta creation \
                operators in the Hamiltonian

            undagb (List[int]): indices corresponding to the beta annihilation \
                operators in the Hamiltonian

        Returns:
            FqeData: FqeData object that stores the result of application
        """

        out = copy.deepcopy(self)
        out.coeff.fill(0.0)
        out.apply_individual_nbody_accumulate(coeff, self, daga, undaga, dagb,
                                              undagb)
        return out

    def apply_individual_nbody_accumulate(self, coeff: complex,
                                          idata: 'FqeData', daga: List[int],
                                          undaga: List[int], dagb: List[int],
                                          undagb: List[int]) -> None:
        """
        Apply function with an individual operator represented in arrays.
        It is assumed that the operator is spin conserving. The result will
        be accumulated to self

        Args:
            coeff (complex): scalar coefficient to be multiplied to the result

            idata (FqeData): input FqeData to which the operators are applied

            daga (List[int]): indices corresponding to the alpha creation \
                operators in the Hamiltonian

            undaga (List[int]): indices corresponding to the alpha annihilation \
                operators in the Hamiltonian

            dagb (List[int]): indices corresponding to the beta creation \
                operators in the Hamiltonian

            undagb (List[int]): indices corresponding to the beta annihilation \
                operators in the Hamiltonian
        """
        assert len(daga) == len(undaga) and len(dagb) == len(undagb)

        ualphamap = numpy.zeros((self.lena(), 3), dtype=numpy.uint64)
        ubetamap = numpy.zeros((self.lenb(), 3), dtype=numpy.uint64)

        acount = self._core.make_mapping_each(ualphamap, True, daga, undaga)
        if acount == 0:
            return
        bcount = self._core.make_mapping_each(ubetamap, False, dagb, undagb)
        if bcount == 0:
            return

        ualphamap = ualphamap[:acount, :]
        ubetamap = ubetamap[:bcount, :]

        alphamap = numpy.zeros((acount, 3), dtype=numpy.int64)
        sourceb_vec = numpy.zeros((bcount,), dtype=numpy.int64)
        targetb_vec = numpy.zeros((bcount,), dtype=numpy.int64)
        parityb_vec = numpy.zeros((bcount,), dtype=numpy.int64)

        alphamap[:, 0] = ualphamap[:, 0]
        for i in range(acount):
            alphamap[i, 1] = self._core.index_alpha(ualphamap[i, 1])
        alphamap[:, 2] = 1 - 2 * ualphamap[:, 2]

        sourceb_vec[:] = ubetamap[:, 0]
        for i in range(bcount):
            targetb_vec[i] = self._core.index_beta(ubetamap[i, 1])
        parityb_vec[:] = 1 - 2 * ubetamap[:, 2]

        if fqe.settings.use_accelerated_code:
            _apply_individual_nbody1_accumulate(coeff, self.coeff, idata.coeff,
                                                alphamap, targetb_vec,
                                                sourceb_vec, parityb_vec)
        else:
            FqeData._apply_individual_nbody1_accumulate_python(
                coeff, self.coeff, idata.coeff, alphamap, targetb_vec,
                sourceb_vec, parityb_vec)

    @staticmethod
    def _apply_individual_nbody1_accumulate_python(
            coeff: 'Nparray', ocoeff: 'Nparray', icoeff: 'Nparray',
            amap: 'Nparray', btarget: 'Nparray', bsource: 'Nparray',
            bparity: 'Nparray') -> None:
        """
        Python version of _apply_individual_nbody1_accumulate
        ported from C from fqe_data.c for compatibility
        """
        for sourcea, targeta, paritya in amap:
            ocoeff[targeta, btarget] += coeff * paritya * numpy.multiply(
                icoeff[sourcea, bsource], bparity)

    def rdm1(self, bradata: Optional['FqeData'] = None) -> Tuple['Nparray']:
        """
        API for calculating 1-particle RDMs given a wave function. When bradata
        is given, it calculates transition RDMs. Depending on the filling, the
        code selects an optimal algorithm.

        Args:
            bradata (optional, FqeData): FqeData for the bra wavefunction. When \
                not given, the ket function is also used for the bra wavefunction

        Returns:
            Tuple[Nparray]: tuple of length 1 that contains numpy array for 1RDM
        """
        return self._rdm1_blocked(bradata)

    def _rdm1_blocked(self,
                      bradata: Optional['FqeData'] = None,
                      max_states: int = 100) -> Tuple['Nparray']:
        """
        API for calculating 1-particle RDMs given a wave function. When bradata
        is given, it calculates transition RDMs. Depending on the filling, the
        code selects an optimal algorithm.
        """
        bradata = self if bradata is None else bradata

        if fqe.settings.use_accelerated_code:
            mappings = bradata._core._get_block_mappings(max_states=max_states)
            norb = bradata.norb()
            coeff_a = bradata.coeff
            coeff_b = bradata.coeff.T.copy()

            coeffconj = self.coeff.conj()
            rdm = numpy.zeros((norb, norb), dtype=bradata._dtype)
            for alpha_range, beta_range, alpha_maps, beta_maps in mappings:
                dvec = _make_dvec_part(coeff_a, alpha_maps, alpha_range,
                                       beta_range, norb, self.lena(),
                                       self.lenb(), True)
                dvec = _make_dvec_part(coeff_b,
                                       beta_maps,
                                       alpha_range,
                                       beta_range,
                                       norb,
                                       self.lena(),
                                       self.lenb(),
                                       False,
                                       out=dvec)

                rdm[:, :] += numpy.tensordot(
                    dvec, coeffconj[alpha_range.start:alpha_range.
                                    stop, beta_range.start:beta_range.stop])

            return (numpy.transpose(rdm.conj()),)
        else:
            dvec2 = self.calculate_dvec_spatial()
            return (numpy.transpose(
                numpy.tensordot(dvec2.conj(), self.coeff,
                                axes=((2, 3), (0, 1)))),)

    def rdm12(self, bradata: Optional['FqeData'] = None
             ) -> Tuple['Nparray', 'Nparray']:
        """
        API for calculating 1- and 2-particle RDMs given a wave function.
        When bradata is given, it calculates transition RDMs. Depending on the
        filling, the code selects an optimal algorithm.

        Args:
            bradata (optional, FqeData): FqeData for the bra wavefunction. When \
                not given, the ket function is also used for the bra wavefunction

        Returns:
            Tuple[Nparray]: tuple of length 2 that contains numpy array for 1 \
                and 2RDM
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
            if fqe.settings.use_accelerated_code:
                return self._rdm12_lowfilling(bradata)
            else:
                return self._rdm12_lowfilling_python(bradata)

        return self._rdm12_halffilling(bradata)

    def _rdm12_halffilling(self, bradata: Optional['FqeData'] = None
                          ) -> Tuple['Nparray', 'Nparray']:
        """
        Standard code for calculating 1- and 2-particle RDMs given a
        wavefunction. When bradata is given, it calculates transition RDMs.
        """
        if fqe.settings.use_accelerated_code:
            return self._rdm12_halffilling_blocked(bradata)
        else:
            dvec = self.calculate_dvec_spatial()
            dvec2 = dvec if bradata is None \
                    else bradata.calculate_dvec_spatial()
            out1 = numpy.transpose(numpy.tensordot(dvec2.conj(), self.coeff))
            out2 = numpy.transpose(numpy.tensordot(
                dvec2.conj(), dvec, axes=((2, 3), (2, 3))),
                                   axes=(1, 2, 0, 3)) * (-1.0)

            for i in range(self.norb()):
                out2[:, i, i, :] += out1[:, :]
            return out1, out2

    def _rdm12_halffilling_blocked(self,
                                   bradata: Optional['FqeData'] = None,
                                   max_states: int = 100
                                  ) -> Tuple['Nparray', 'Nparray']:
        """
        Standard code for calculating 1- and 2-particle RDMs given a
        wavefunction. When bradata is given, it calculates transition RDMs.
        """
        bradata = self if bradata is None else bradata

        mappings = self._core._get_block_mappings(max_states=max_states)
        norb = bradata.norb()
        coeff_a = self.coeff
        coeff_b = self.coeff.T.copy()
        bcoeff_a = bradata.coeff
        bcoeff_b = bradata.coeff.T.copy()

        rdm1 = numpy.zeros((norb,) * 2, dtype=bradata._dtype)
        rdm2 = numpy.zeros((norb,) * 4, dtype=bradata._dtype)
        for alpha_range, beta_range, alpha_maps, beta_maps in mappings:
            dvec = _make_dvec_part(coeff_a, alpha_maps, alpha_range, beta_range,
                                   norb, self.lena(), self.lenb(), True)
            dvec = _make_dvec_part(coeff_b,
                                   beta_maps,
                                   alpha_range,
                                   beta_range,
                                   norb,
                                   self.lena(),
                                   self.lenb(),
                                   False,
                                   out=dvec)

            dvec2 = _make_dvec_part(bcoeff_a, alpha_maps,
                                    alpha_range, beta_range, norb, self.lena(),
                                    self.lenb(), True)
            dvec2 = _make_dvec_part(bcoeff_b,
                                    beta_maps,
                                    alpha_range,
                                    beta_range,
                                    norb,
                                    self.lena(),
                                    self.lenb(),
                                    False,
                                    out=dvec2)

            dvec2conj = dvec2.conj()
            rdm1[:, :] += numpy.tensordot(
                dvec2conj, self.coeff[alpha_range.start:alpha_range.
                                      stop, beta_range.start:beta_range.stop])
            rdm2[:, :, :, :] += \
                numpy.tensordot(dvec2conj, dvec, axes=((2, 3), (2, 3)))

        rdm2 = -rdm2.transpose(1, 2, 0, 3)
        for i in range(self.norb()):
            rdm2[:, i, i, :] += rdm1[:, :]
        return (numpy.transpose(rdm1), rdm2)

    def _rdm12_lowfilling_python(self, bradata: Optional['FqeData'] = None
                                ) -> Tuple['Nparray', 'Nparray']:
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
            outpack += numpy.tensordot(inter2.conj(),
                                       inter,
                                       axes=((1, 2), (1, 2)))

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
            outunpack += numpy.tensordot(inter2.conj(),
                                         inter,
                                         axes=((2, 3), (2, 3)))

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
            outpack += numpy.tensordot(inter2.conj(),
                                       inter,
                                       axes=((1, 2), (1, 2)))

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

    def _rdm12_lowfilling(self, bradata: Optional['FqeData'] = None
                         ) -> Tuple['Nparray', 'Nparray']:
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
            alpha_array = self._to_array1(alpha_map, norb)

            def compute_intermediate0(coeff):
                tmp = numpy.zeros((nlt, int(binom(norb, nalpha - 2)), lenb),
                                  dtype=self.coeff.dtype)
                _apply_array12_lowfillingaa(self.coeff, alpha_array, tmp)
                return tmp

            inter = compute_intermediate0(self.coeff)
            inter2 = inter if bradata is None else compute_intermediate0(
                bradata.coeff)
            outpack += numpy.tensordot(inter2.conj(),
                                       inter,
                                       axes=((1, 2), (1, 2)))

        if self.nalpha() - 1 >= 0 and self.nbeta() - 1 >= 0:
            alpha_map, beta_map = self._core.find_mapping(-1, -1)
            inter = numpy.zeros((norb, norb, int(binom(
                norb, nalpha - 1)), int(binom(norb, nbeta - 1))),
                                dtype=self._dtype)

            alpha_array = self._to_array2(alpha_map, norb)
            beta_array = self._to_array2(beta_map, norb)

            alpha_map, beta_map = self._core.find_mapping(-1, -1)
            _apply_array12_lowfillingab(self.coeff, alpha_array, beta_array,
                                        nalpha, nbeta, inter)

            if bradata is None:
                inter2 = inter
            else:
                inter2 = numpy.zeros((norb, norb, int(binom(
                    norb, nalpha - 1)), int(binom(norb, nbeta - 1))),
                                     dtype=self._dtype)
                _apply_array12_lowfillingab(bradata.coeff, alpha_array, beta_array, \
                                            nalpha, nbeta, inter2)

            # 0.25 needed since _apply_array12_lowfillingab adds a factor 2
            outunpack += numpy.tensordot(
                inter2.conj(), inter, axes=((2, 3), (2, 3))) * 0.25

        if self.nbeta() - 2 >= 0:
            _, beta_map = self._core.find_mapping(0, -2)
            beta_array = self._to_array1(beta_map, norb)

            def compute_intermediate2(coeff):
                tmp = numpy.zeros((nlt, lena, int(binom(norb, nbeta - 2))),
                                  dtype=self.coeff.dtype)
                _apply_array12_lowfillingaa(self.coeff,
                                            beta_array,
                                            tmp,
                                            alpha=False)

                return tmp

            inter = compute_intermediate2(self.coeff)
            inter2 = inter if bradata is None else compute_intermediate2(
                bradata.coeff)
            outpack += numpy.tensordot(inter2.conj(),
                                       inter,
                                       axes=((1, 2), (1, 2)))

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
               evec2: 'Nparray' = None
              ) -> Tuple['Nparray', 'Nparray', 'Nparray']:
        """
        Calculates 1- through 3-particle RDMs given a wave function. When
        bradata is given, it calculates transition RDMs.

        Args:
            bradata (optional, FqeData): FqeData for the bra wavefunction. When \
                not given, the ket function is also used for the bra wavefunction

        Returns:
            Tuple[Nparray]: tuple of length 3 that contains numpy array for 1, \
                2, and 3RDM
        """
        norb = self.norb()
        if dvec is None:
            dvec = self.calculate_dvec_spatial()
        if dvec2 is None:
            dvec2 = dvec if bradata is None else bradata.calculate_dvec_spatial(
            )

        out1 = numpy.transpose(numpy.tensordot(dvec2.conj(), self.coeff))
        out2 = numpy.transpose(numpy.tensordot(
            dvec2.conj(), dvec, axes=((2, 3), (2, 3))),
                               axes=(1, 2, 0, 3)) * (-1.0)
        for i in range(norb):
            out2[:, i, i, :] += out1[:, :]

        if evec2 is not None:
            out3 = -numpy.transpose(numpy.tensordot(
                evec2.conj(), dvec, axes=((4, 5), (2, 3))),
                                    axes=(3, 1, 4, 2, 0, 5))
        else:
            out3 = numpy.empty((norb,) * 6, dtype=dvec.dtype)
            dvec_conj = dvec.conj()
            for i in range(norb):
                for j in range(norb):
                    tmp = dvec2[i, j, :, :]
                    evec_ij = self._calculate_dvec_spatial_with_coeff(tmp)
                    out3[j, i, :, :, :, :] = -numpy.tensordot(
                        evec_ij, dvec_conj, axes=((2, 3), (2, 3)))
            out3 = numpy.transpose(out3.conj(), axes=(0, 3, 4, 1, 2, 5))

        for i in range(norb):
            out3[:, i, :, i, :, :] -= out2[:, :, :, :]
            out3[:, :, i, :, i, :] -= out2[:, :, :, :]
            for j in range(norb):
                out3[:, i, j, i, j, :] += out1[:, :]
                for k in range(norb):
                    out3[j, k, i, i, :, :] -= out2[k, j, :, :]
        return (out1, out2, out3)

    def rdm1234(self, bradata: Optional['FqeData'] = None
               ) -> Tuple['Nparray', 'Nparray', 'Nparray', 'Nparray']:
        """
        Calculates 1- through 4-particle RDMs given a wave function. When
        bradata is given, it calculates transition RDMs.

        Args:
            bradata (optional, FqeData): FqeData for the bra wavefunction. When \
                not given, the ket function is also used for the bra wavefunction

        Returns:
            Tuple[Nparray]: tuple of length 4 that contains numpy array for 1, \
                2, 3, and 4RDM
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

        out4 = numpy.transpose(numpy.tensordot(evec2.conj(),
                                               evec,
                                               axes=((4, 5), (4, 5))),
                               axes=(3, 1, 4, 6, 2, 0, 5, 7))
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

        Returns:
            Nparray: result of computation
        """
        return self._calculate_dvec_spatial_with_coeff(self.coeff)

    def calculate_dvec_spin(self) -> Tuple['Nparray', 'Nparray']:
        """Generate a pair of

        .. math::
            D^J_{ij} = \\sum_I \\langle J|a^\\dagger_i a_j|I\\rangle C_I

        using self.coeff as an input.

        Returns:
            Tuple[Nparray, Nparray]: result of computation. Alpha and beta \
                are seperately packed in the tuple to be returned
        """
        return self._calculate_dvec_spin_with_coeff(self.coeff)

    def calculate_dvec_spatial_fixed_j(self, jorb: int) -> 'Nparray':
        """Generate, for a fixed j,

        .. math::
            D^J_{ij} = \\sum_I \\langle J|a^\\dagger_i a_j|I\\rangle C_I

        using self.coeff as an input

        Args:
            jorb (int): specify j in the above expression

        Returns:
            Nparray: result of computation.
        """
        return self._calculate_dvec_spatial_with_coeff_fixed_j(self.coeff, jorb)

    def calculate_dvec_spin_fixed_j(self, jorb: int) -> 'Nparray':
        """Generate a pair of the following, for a fixed j

        .. math::
            D^J_{ij} = \\sum_I \\langle J|a^\\dagger_i a_j|I\\rangle C_I

        using self.coeff as an input. Alpha and beta are seperately packed in
        the tuple to be returned

        Args:
            jorb (int): specify j in the above expression

        Returns:
            Nparray: result of computation.
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
        if fqe.settings.use_accelerated_code:
            _make_dvec(dvec, coeff, [
                self.alpha_map(i, j) for i in range(norb) for j in range(norb)
            ], self.lena(), self.lenb(), True)
            _make_dvec(
                dvec, coeff,
                [self.beta_map(i, j) for i in range(norb) for j in range(norb)],
                self.lena(), self.lenb(), False)
        else:
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
        if fqe.settings.use_accelerated_code:
            alpha_maps = [
                self.alpha_map(i, j) for i in range(norb) for j in range(norb)
            ]
            beta_maps = [
                self.beta_map(i, j) for i in range(norb) for j in range(norb)
            ]
            _make_dvec(dveca, coeff, alpha_maps, self.lena(), self.lenb(), True)
            _make_dvec(dvecb, coeff, beta_maps, self.lena(), self.lenb(), False)
        else:
            for i in range(norb):
                for j in range(norb):
                    for source, target, parity in self.alpha_map(i, j):
                        dveca[i, j, target, :] += coeff[source, :] * parity
                    for source, target, parity in self.beta_map(i, j):
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
        norb = self.norb()
        out = numpy.zeros(self.coeff.shape, dtype=self._dtype)
        if fqe.settings.use_accelerated_code:
            alpha_maps = [
                self.alpha_map(j, i) for i in range(norb) for j in range(norb)
            ]
            beta_maps = [
                self.beta_map(j, i) for i in range(norb) for j in range(norb)
            ]
            _make_coeff(dvec, out, alpha_maps, self.lena(), self.lenb(), True)
            _make_coeff(dvec, out, beta_maps, self.lena(), self.lenb(), False)
        else:
            for i in range(self.norb()):
                for j in range(self.norb()):
                    for source, target, parity in self.alpha_map(j, i):
                        out[source, :] += dvec[i, j, target, :] * parity
                    for source, target, parity in self.beta_map(j, i):
                        out[:, source] += dvec[i, j, :, target] * parity
        return out

    def _calculate_dvec_spatial_compressed(self) -> 'Nparray':
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

    def _calculate_coeff_spin_with_dvec(self, dvec: Tuple['Nparray', 'Nparray']
                                       ) -> 'Nparray':
        """Generate

        .. math::

            C_I = \\sum_J \\langle I|a^\\dagger_i a_j|J\\rangle D^J_{ij}
        """
        norb = self.norb()
        out = numpy.zeros(self.coeff.shape, dtype=self._dtype)
        if fqe.settings.use_accelerated_code:
            alpha_maps = [
                self.alpha_map(j, i) for i in range(norb) for j in range(norb)
            ]
            beta_maps = [
                self.beta_map(j, i) for i in range(norb) for j in range(norb)
            ]
            _make_coeff(dvec[0], out, alpha_maps, self.lena(), self.lenb(),
                        True)
            _make_coeff(dvec[1], out, beta_maps, self.lena(), self.lenb(),
                        False)
        else:
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
        parity due to sorting. opa and opb are integer arrays. The result will be
        stored in-place.

        Args:
            time (float): time to be used for time propagation

            coeff (complex): scalar coefficient

            opa (List[int]): index list for alpha

            opb (List[int]): index list for beta
        """
        n_a = len(opa)
        n_b = len(opb)
        coeff *= (-1)**(n_a * (n_a - 1) // 2 + n_b * (n_b - 1) // 2)

        amap = numpy.zeros((self.lena(),), dtype=numpy.int64)
        bmap = numpy.zeros((self.lenb(),), dtype=numpy.int64)
        amask = reverse_integer_index(opa)
        bmask = reverse_integer_index(opb)
        count = 0
        for index in range(self.lena()):
            current = int(self._core.string_alpha(index))
            if (~current) & amask == 0:
                amap[count] = index
                count += 1
        amap = amap[:count]
        count = 0
        for index in range(self.lenb()):
            current = int(self._core.string_beta(index))
            if (~current) & bmask == 0:
                bmap[count] = index
                count += 1
        bmap = bmap[:count]

        factor = numpy.exp(-time * numpy.real(coeff) * 2.j)
        if amap.size != 0 and bmap.size != 0:
            xi, yi = numpy.meshgrid(amap, bmap, indexing='ij')
            if fqe.settings.use_accelerated_code:
                _sparse_scale(xi, yi, factor, self.coeff)
            else:
                self.coeff[xi, yi] *= factor

    def evolve_individual_nbody_nontrivial(self, time: float, coeff: complex,
                                           daga: List[int], undaga: List[int],
                                           dagb: List[int],
                                           undagb: List[int]) -> 'FqeData':
        """
        This code time-evolves a wave function with an individual n-body
        generator which is spin-conserving. It is assumed that :math:`\\hat{T}^2 = 0`.
        Using :math:`TT = 0` and :math:`TT^\\dagger` is diagonal in the determinant
        space, one could evaluate as

        .. math::
            \\exp(-i(T+T^\\dagger)t)
                &= 1 + i(T+T^\\dagger)t - \\frac{1}{2}(TT^\\dagger + T^\\dagger T)t^2
                 - i\\frac{1}{6}(TT^\\dagger T + T^\\dagger TT^\\dagger)t^3 + \\cdots \\\\
                &= -1 + \\cos(t\\sqrt{TT^\\dagger}) + \\cos(t\\sqrt{T^\\dagger T})
                 - iT\\frac{\\sin(t\\sqrt{T^\\dagger T})}{\\sqrt{T^\\dagger T}}
                 - iT^\\dagger\\frac{\\sin(t\\sqrt{TT^\\dagger})}{\\sqrt{TT^\\dagger}}

        Args:
            time (float): time to be used for time propagation

            coeff (complex): scalar coefficient

            daga (List[int]): index list for alpha creation operators

            undaga (List[int]): index list for alpha annihilation operators

            dagb (List[int]): index list for beta creation operators

            undagb (List[int]): index list for beta annihilation operators

        Returns:
            FqeData: result is returned as an FqeData object
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
        absol = numpy.absolute(ncoeff)
        sinfactor = numpy.sin(time * absol) / absol

        out = copy.deepcopy(self)

        out.apply_cos_inplace(time, ncoeff, numbera + dagworka, undagworka,
                              numberb + dagworkb, undagworkb)

        out.apply_cos_inplace(time, ncoeff, numbera + undagworka, dagworka,
                              numberb + undagworkb, dagworkb)
        phase = (-1)**((len(daga) + len(undaga)) * (len(dagb) + len(undagb)))
        work_cof = numpy.conj(coeff) * phase * (-1.0j)

        out.apply_individual_nbody_accumulate(work_cof * sinfactor, self,
                                              undaga, daga, undagb, dagb)

        out.apply_individual_nbody_accumulate(coeff * (-1.0j) * sinfactor, self,
                                              daga, undaga, dagb, undagb)
        return out

    def _evaluate_map(self, opa: List[int], oha: List[int], opb: List[int],
                      ohb: List[int]):
        """
        Utility internal function that performs part of the operations in
        evolve_inplace_individual_nbody_nontrivial, in which alpha and beta
        mappings are created.
        """
        amap = numpy.zeros((self.lena(),), dtype=numpy.int64)
        bmap = numpy.zeros((self.lenb(),), dtype=numpy.int64)
        apmask = reverse_integer_index(opa)
        ahmask = reverse_integer_index(oha)
        bpmask = reverse_integer_index(opb)
        bhmask = reverse_integer_index(ohb)
        if fqe.settings.use_accelerated_code:
            count = _evaluate_map_each(amap, self._core._astr, self.lena(),
                                       apmask, ahmask)
            amap = amap[:count]
            count = _evaluate_map_each(bmap, self._core._bstr, self.lenb(),
                                       bpmask, bhmask)
            bmap = bmap[:count]
        else:
            counter = 0
            for index in range(self.lena()):
                current = int(self._core.string_alpha(index))
                if ((~current) & apmask) == 0 and (current & ahmask) == 0:
                    amap[counter] = index
                    counter += 1
            amap = amap[:counter]
            counter = 0
            for index in range(self.lenb()):
                current = int(self._core.string_beta(index))
                if ((~current) & bpmask) == 0 and (current & bhmask) == 0:
                    bmap[counter] = index
                    counter += 1
            bmap = bmap[:counter]
        return amap, bmap

    def apply_cos_inplace(self, time: float, ncoeff: complex, opa: List[int],
                          oha: List[int], opb: List[int],
                          ohb: List[int]) -> None:
        """
        Utility internal function that performs part of the operations in
        evolve_inplace_individual_nbody_nontrivial. This function processes
        both TTd and TdT cases simultaneously and in-place without allocating
        memory.

        Args:
            time (float): time to be used for time propagation

            ncoeff (complex): scalar coefficient

            opa (List[int]): index list for alpha creation operators

            oha (List[int]): index list for alpha annihilation operators

            opb (List[int]): index list for beta creation operators

            ohb (List[int]): index list for beta annihilation operators
        """
        absol = numpy.absolute(ncoeff)
        factor = numpy.cos(time * absol)

        amap, bmap = self._evaluate_map(opa, oha, opb, ohb)

        if amap.size != 0 and bmap.size != 0:
            xi, yi = numpy.meshgrid(amap, bmap, indexing='ij')
            if fqe.settings.use_accelerated_code:
                _sparse_scale(xi, yi, factor, self.coeff)
            else:
                self.coeff[xi, yi] *= factor

    def alpha_map(self, iorb: int, jorb: int) -> List[Tuple[int, int, int]]:
        """Access the mapping for a singlet excitation from the current
        sector for alpha orbitals

        Args:
            iorb: index for creation

            jorb: index for annihilation

        Returns:
            List[Tuple[int, int, int]]: mapping informatin for the index pairs
        """
        return self._core.alpha_map(iorb, jorb)

    def beta_map(self, iorb: int, jorb: int) -> List[Tuple[int, int, int]]:
        """Access the mapping for a singlet excitation from the current
        sector for beta orbitals

        Args:
            iorb: index for creation

            jorb: index for annihilation

        Returns:
            List[Tuple[int, int, int]]: mapping informatin for the index pairs
        """
        return self._core.beta_map(iorb, jorb)

    def ax_plus_y(self, sval: complex, other: 'FqeData') -> 'FqeData':
        """Scale and add the data in the fqedata structure

            self.coeff += sval * other.coeff

        Args:
            sval (complex): scalar coefficient

            other (FqeData): FqeData object to be added to self
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

        Returns:
            int: length of the alpha configuration space
        """
        return self._core.lena()

    def lenb(self) -> int:
        """Length of the beta configuration space

        Returns:
            int: length of the beta configuration space
        """
        return self._core.lenb()

    def nalpha(self) -> int:
        """Number of alpha electrons

        Returns:
            int: the number of alpha electrons
        """
        return self._core.nalpha()

    def nbeta(self) -> int:
        """Number of beta electrons

        Returns:
            int: the number of beta electrons
        """
        return self._core.nbeta()

    def n_electrons(self) -> int:
        """Particle number getter

        Returns:
            int: the number of electrons in total
        """
        return self._nele

    def generator(self) -> Iterator[Tuple[int, int, complex]]:
        """Generator for configurations in the FqeData object
        """
        for inda in range(self._core.lena()):
            alpha_str = self._core.string_alpha(inda)
            for indb in range(self._core.lenb()):
                beta_str = self._core.string_beta(indb)
                yield alpha_str, beta_str, self.coeff[inda, indb]

    def norb(self) -> int:
        """Number of beta electrons

        Returns:
            int: the number of orbitals
        """
        return self._core.norb()

    def norm(self) -> float:
        """Return the norm of the the sector wavefunction

        Returns:
            float: norm
        """
        return numpy.linalg.norm(self.coeff)

    def print_sector(self,
                     pformat: Optional[Callable[[int, int], str]] = None,
                     threshold: Optional[float] = 0.0001):
        """Iterate over the strings and coefficients and print then
        using the print format

        Args:
            pformat (Callable[[int, int], str]): custom format

            threshold (float): threshold above which the elements are \
                printed
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

    def beta_inversion(self) -> 'Nparray':
        """Return the coefficients with an inversion of the beta strings.

        Returns:
            int: resulting coefficient in numpy array
        """
        return numpy.flip(self.coeff, 1)

    def scale(self, sval: complex) -> None:
        """ Scale the wavefunction by the value sval

        Args:
            sval (complex): value to scale by
        """
        self.coeff = self.coeff.astype(numpy.complex128) * sval

    def fill(self, value: complex) -> None:
        """ Fills the wavefunction with the value specified

        Args:
            value (complex): value to be filled
        """
        self.coeff.fill(value)

    def set_wfn(self,
                strategy: Optional[str] = None,
                raw_data: 'Nparray' = numpy.empty(0)) -> None:
        """Set the values of the fqedata wavefunction based on a strategy

        Args:
            strategy (str): the procedure to follow to set the coeffs

            raw_data (numpy.array(dim(self.lena(), self.lenb()), \
                dtype=numpy.complex128)): the values to use \
                if setting from data.  If vrange is supplied, the first column \
                in data will correspond to the first index in vrange
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

    def empty_copy(self) -> 'FqeData':
        """create a copy of the self with zero coefficients

        Returns:
            FqeData: a new object with zero coefficients
        """
        new_data = FqeData(nalpha=self.nalpha(),
                           nbeta=self.nbeta(),
                           norb=self._core.norb(),
                           fcigraph=self._core,
                           dtype=self._dtype)
        new_data._low_thresh = self._low_thresh
        new_data.coeff = numpy.zeros_like(self.coeff)
        return new_data

    def get_spin_opdm(self) -> Tuple['Nparray', 'Nparray']:
        """calculate the alpha-alpha and beta-beta block of the 1-RDM

        Returns:
            Tuple[Nparray, Nparray]: alpha and beta 1RDM
        """
        dveca, dvecb = self.calculate_dvec_spin()
        alpha_opdm = numpy.tensordot(dveca, self.coeff.conj(), axes=2)
        beta_opdm = numpy.tensordot(dvecb, self.coeff.conj(), axes=2)
        return alpha_opdm, beta_opdm

    def get_ab_tpdm(self) -> 'Nparray':
        """Get the alpha-beta block of the 2-RDM

        tensor[i, j, k, l] = <ia^ jb^ kb la>

        Returns:
            Nparray: the above quantity in numpy array
        """
        dveca, dvecb = self.calculate_dvec_spin()
        tpdm_ab = numpy.transpose(numpy.tensordot(dveca.conj(),
                                                  dvecb,
                                                  axes=((2, 3), (2, 3))),
                                  axes=(1, 2, 3, 0))
        return tpdm_ab

    def get_aa_tpdm(self) -> Tuple['Nparray', 'Nparray']:
        """Get the alpha-alpha block of the 1- and 2-RDMs

        tensor[i, j, k, l] = <ia^ ja^ ka la>

        Returns:
            Tuple[Nparray, Nparray]: alpha-alpha block of the 1- and 2-RDMs
        """
        dveca, _ = self.calculate_dvec_spin()
        alpha_opdm = numpy.tensordot(dveca, self.coeff.conj(), axes=2)
        nik_njl_aa = numpy.transpose(numpy.tensordot(dveca.conj(),
                                                     dveca,
                                                     axes=((2, 3), (2, 3))),
                                     axes=(1, 2, 0, 3))
        for ii in range(nik_njl_aa.shape[1]):
            nik_njl_aa[:, ii, ii, :] -= alpha_opdm
        return alpha_opdm, -nik_njl_aa

    def get_bb_tpdm(self):
        """Get the beta-beta block of the 1- and 2-RDMs

        tensor[i, j, k, l] = <ib^ jb^ kb lb>

        Returns:
            Tuple[Nparray, Nparray]: beta-beta block of the 1- and 2-RDMs
        """
        _, dvecb = self.calculate_dvec_spin()
        beta_opdm = numpy.tensordot(dvecb, self.coeff.conj(), axes=2)
        nik_njl_bb = numpy.transpose(numpy.tensordot(dvecb.conj(),
                                                     dvecb,
                                                     axes=((2, 3), (2, 3))),
                                     axes=(1, 2, 0, 3))
        for ii in range(nik_njl_bb.shape[1]):
            nik_njl_bb[:, ii, ii, :] -= beta_opdm
        return beta_opdm, -nik_njl_bb

    def get_openfermion_rdms(self) -> Tuple['Nparray', 'Nparray']:
        """
        Generate spin-rdms and return in openfermion format

        Returns:
            Tuple[Nparray, Nparray]: 1 and 2 RDMs in the OpenFermion \
                format in numpy arrays
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
        tpdm[::2, 1::2, ::2, 1::2] = -tpdm_ab.transpose(0, 1, 3, 2)
        tpdm[1::2, ::2, ::2, 1::2] = tpdm_ab.transpose(1, 0, 3, 2)
        tpdm[1::2, ::2, 1::2, ::2] = \
            -tpdm[1::2, ::2, ::2, 1::2].transpose(0, 1, 3, 2)

        return opdm, tpdm

    def get_three_spin_blocks_rdm(self) -> 'Nparray':
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

        Returns:
            Nparray: above RDM in numpy array
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
        # alpha-alpha-alpha
        for t, u in itertools.product(range(self.norb()), repeat=2):
            tdveca_a, _ = self._calculate_dvec_spin_with_coeff(
                dveca[t, u, :, :])
            tdveca_b, tdvecb_b = self._calculate_dvec_spin_with_coeff(
                dvecb[t, u, :, :])
            for r, s in itertools.product(range(self.norb()), repeat=2):
                # p(:)^ q(:) r^ s t^ u
                # a-a-a
                pq_rdm = numpy.tensordot(dveca_conj, tdveca_a[r, s, :, :]).T
                ckckck_aaa[:, :, r, s, t, u] = pq_rdm
                # a-a-b
                pq_rdm = numpy.tensordot(dveca_conj, tdveca_b[r, s, :, :]).T
                ckckck_aab[:, :, r, s, t, u] = pq_rdm
                # a-b-b
                pq_rdm = numpy.tensordot(dveca_conj, tdvecb_b[r, s, :, :]).T
                ckckck_abb[:, :, r, s, t, u] = pq_rdm
                # b-b-b
                pq_rdm = numpy.tensordot(dvecb_conj, tdvecb_b[r, s, :, :]).T
                ckckck_bbb[:, :, r, s, t, u] = pq_rdm

        # p^ r^ t^ u s q = p^ q r^ s t^ u + d(q, r) p^ t^ s u - d(q, t)p^ r^ s u
        #                 + d(s, t)p^ r^ q u - d(q,r)d(s,t)p^ u
        tpdm_swapped = tpdm.transpose(0, 2, 1, 3).copy()

        for ii in range(ckckck_aaa.shape[0]):
            ckckck_aaa[:, ii, ii, :, :, :] += tpdm_swapped[::2, ::2, ::2, ::2]
            ckckck_aaa[:, ii, :, :, ii, :] -= tpdm[::2, ::2, ::2, ::2]
            ckckck_aaa[:, :, :, ii, ii, :] += tpdm_swapped[::2, ::2, ::2, ::2]
            for jj in range(ckckck_aaa.shape[0]):
                ckckck_aaa[:, ii, ii, jj, jj, :] -= opdm[::2, ::2]
        ccckkk_aaa = ckckck_aaa.transpose(0, 2, 4, 5, 3, 1).copy()

        for ii in range(ckckck_aab.shape[0]):
            ckckck_aab[:, ii, ii, :, :, :] += tpdm_swapped[::2, ::2, 1::2, 1::2]
        ccckkk_aab = ckckck_aab.transpose(0, 2, 4, 5, 3, 1).copy()

        for ii in range(ckckck_abb.shape[0]):
            ckckck_abb[:, :, :, ii, ii, :] += tpdm_swapped[::2, ::2, 1::2, 1::2]
        ccckkk_abb = ckckck_abb.transpose(0, 2, 4, 5, 3, 1).copy()

        for ii in range(ckckck_bbb.shape[0]):
            ckckck_bbb[:, ii, ii, :, :, :] += tpdm_swapped[1::2, 1::2, 1::2, 1::
                                                           2]
            ckckck_bbb[:, ii, :, :, ii, :] -= tpdm[1::2, 1::2, 1::2, 1::2]
            ckckck_bbb[:, :, :, ii, ii, :] += tpdm_swapped[1::2, 1::2, 1::2, 1::
                                                           2]
            for jj in range(ckckck_bbb.shape[0]):
                ckckck_bbb[:, ii, ii, jj, jj, :] -= opdm[1::2, 1::2]
        ccckkk_bbb = ckckck_bbb.transpose(0, 2, 4, 5, 3, 1).copy()

        return ccckkk_aaa, ccckkk_aab, ccckkk_abb, ccckkk_bbb

    def get_three_pdm(self):
        norbs = self.norb()
        ccckkk = numpy.zeros((2 * norbs,) * 6, dtype=self._dtype)
        ccckkk_aaa, ccckkk_aab, ccckkk_abb, ccckkk_bbb = \
            self.get_three_spin_blocks_rdm()

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
