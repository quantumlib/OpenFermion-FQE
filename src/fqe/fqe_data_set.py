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
"""Fundamental Fqe data structure
"""
#pylint: disable=protected-access
import copy
from typing import Tuple, Dict, List, Optional, TYPE_CHECKING

import numpy

from fqe.fci_graph_set import FciGraphSet
from fqe.lib.fqe_data import _calculate_dvec1, _calculate_dvec2
from fqe.lib.fqe_data import _calculate_dvec1_j, _calculate_dvec2_j
from fqe.lib.fqe_data import _calculate_coeff1, _calculate_coeff2
from fqe.lib.fqe_data import _apply_individual_nbody1_accumulate
from fqe.fqe_data import FqeData
import fqe.settings

if TYPE_CHECKING:
    from fqe.fci_graph import FciGraph
    from numpy import ndarray as Nparray


class FqeDataSet:
    """One of the fundamental data structures in the FQE. FqeDataSet
    is essentially a view of the list of FqeData's that belong to the same subspace
    characterized by a given particle number. The keys for self._data are (n_alpha, n_beta)
    where n_alpha and n_beta are the numbers of alpha and beta electrons.
    For the two-electron case, FqeDataSet consists of the sectors for
    (alpha, beta) = (2, 0), (1, 1), and (0, 2).
    """

    def __init__(self, nele: int, norb: int,
                 data: Dict[Tuple[int, int], 'FqeData']) -> None:
        """
        Args:
            nele (int): the number of electrons

            norb (int): the number of spatial orbitals

            data (Dict[Tuple[int, int], FqeData]): FqeData to be included
        """
        self._nele = nele
        self._norb = norb
        self._data: Dict[Tuple[int, int], 'FqeData'] = data

        graphset = FciGraphSet(0, 1)
        for work in self._data.values():
            graphset.append(work.get_fcigraph())

    def empty_copy(self) -> 'FqeDataSet':
        """
        Copy the object but leave all data arrays uninitialized

        Returns:
            FqeDataSet: copied FqeDataSet object
        """
        newdata = dict()
        for key, value in self._data.items():
            newdata[key] = FqeData(value._core.nalpha(), value._core.nbeta(),
                                   value.norb(), value._core, value._dtype)

        return FqeDataSet(self._nele, self._norb, newdata)

    def sectors(self) -> Dict[Tuple[int, int], 'FqeData']:
        """
        Returns:
            Dict[Tuple[int, int], FqeData]: the CI vectors stored in self._data
        """
        return self._data

    def ax_plus_y(self, factor: complex, other: 'FqeDataSet') -> None:
        """
        Performs :math:`y = ax + y` with :math:`y` being self.
        The result will be kept inplace.

        Args:
            factor (complex): scalar factor :math:`a`

            other (FqeDataSet): FqeDataSet to be added (:math:`y` above)
        """
        if self._data.keys() != other._data.keys():
            raise ValueError('keys are inconsistent in FqeDataSet.ax_plus_y')
        for key, sector in self._data.items():
            sector.ax_plus_y(factor, other._data[key])

    def scale(self, factor: complex) -> None:
        """
        Scales all of the data by the factor specified

        Args:
            factor (complex): factor using which self is scaled
        """
        for _, sector in self._data.items():
            sector.scale(factor)

    def fill(self, value: complex) -> None:
        """
        Fills all of the data to the value specified

        Args:
            value (complex): value to be filled
        """
        for _, sector in self._data.items():
            sector.fill(value)

    def apply_inplace(self, array: Tuple['Nparray', ...]) -> None:
        """
        Applies an operator specified by the tuple of numpy arrays.
        The result will be kept in-place.

        Args:
            array (Tuple[Nparray, ...]): array that represents the Hamiltonian
        """
        other = self.apply(array)
        for key, sector in self._data.items():
            sector.coeff[:, :] = other._data[key].coeff[:, :]

    def apply(self, array: Tuple['Nparray', ...]) -> 'FqeDataSet':
        """
        Applies an operator specified by the tuple of numpy arrays.
        The result will be returned as a FqeDataSet object. self is unchanged.

        Args:
            array (Tuple[Nparray, ...]): array that represents the Hamiltonian

        Returns:
            FqeDataSet: result of the computation in FqeDataSet
        """
        if len(array) == 1:
            out = self._apply1(array[0])
        elif len(array) == 2:
            out = self._apply12(array[0], array[1])
        elif len(array) == 3:
            out = self._apply123(array[0], array[1], array[2])
        elif len(array) == 4:
            out = self._apply1234(array[0], array[1], array[2], array[3])
        else:
            raise ValueError('unexpected array passed in FqeData apply_inplace')
        return out

    def _apply1(self, h1e: 'Nparray') -> 'FqeDataSet':
        """
        Applies a one-body operator specified by the tuple of numpy arrays.
        The result will be returned as a FqeDataSet object. self is unchanged.
        """
        norb = self._norb
        assert h1e.shape == (norb * 2, norb * 2)

        ncol = 0
        jorb = 0
        for j in range(norb * 2):
            if numpy.any(h1e[:, j]):
                ncol += 1
                jorb = j
            if ncol > 1:
                break

        if ncol > 1:
            dvec = self._calculate_dvec()
            out = self.empty_copy()
            for key, sector in out._data.items():
                sector.coeff = numpy.tensordot(h1e, dvec[key])
        else:
            dvec = self._calculate_dvec_fixed_j(jorb)
            out = self.empty_copy()
            for key, sector in out._data.items():
                sector.coeff = numpy.tensordot(h1e[:, jorb], dvec[key], axes=1)

        return out

    def _apply12(self, h1e: 'Nparray', h2e: 'Nparray') -> 'FqeDataSet':
        """
        Applies a one- and two-body operator specified by the tuple of numpy arrays.
        The result will be returned as a FqeDataSet object. self is unchanged.
        """
        norb = self._norb
        assert h1e.shape == (norb * 2,
                             norb * 2) and h2e.shape == (norb * 2, norb * 2,
                                                         norb * 2, norb * 2)

        h1e = copy.deepcopy(h1e)
        h2e = numpy.moveaxis(copy.deepcopy(h2e), 1, 2) * (-1.0)
        for k in range(norb * 2):
            h1e[:, :] -= h2e[:, k, k, :]

        dvec = self._calculate_dvec()
        out = self.empty_copy()
        for key, fsector in out._data.items():
            fsector.coeff = numpy.tensordot(h1e, dvec[key])

        for key, sector in dvec.items():
            dvec[key] = numpy.tensordot(h2e, sector)

        result = self._calculate_coeff_with_dvec(dvec)
        for key, fsector in out._data.items():
            fsector.coeff += result[key]
        return out

    def _apply123(self, h1e: 'Nparray', h2e: 'Nparray',
                  h3e: 'Nparray') -> 'FqeDataSet':
        """
        Applies a one-, two-, and three-body operator specified by the tuple of numpy arrays.
        The result will be returned as a FqeDataSet object. self is unchanged.
        """
        norb = self._norb
        assert h3e.shape == (norb * 2, norb * 2, norb * 2, norb * 2, norb * 2,
                             norb * 2)

        nh1e = numpy.copy(h1e)
        nh2e = numpy.copy(h2e)

        for i in range(norb * 2):
            for j in range(norb * 2):
                for k in range(norb * 2):
                    nh2e[j, k, :, :] += (- h3e[k, j, i, i, :, :] \
                                         - h3e[j, i, k, i, :, :] \
                                         - h3e[j, k, i, :, i, :])
                nh1e[:, :] += h3e[:, i, j, i, j, :]

        out = self._apply12(nh1e, nh2e)

        dvec = self._calculate_dvec()
        evec = self._calculate_evec(dvec)

        for key, sector in evec.items():
            dvec[key] = numpy.tensordot(h3e,
                                        sector,
                                        axes=((1, 4, 2, 5), (0, 1, 2, 3)))

        result = self._calculate_coeff_with_dvec(dvec)
        for key, fsector in out._data.items():
            fsector.coeff -= result[key]
        return out

    def _apply1234(self, h1e: 'Nparray', h2e: 'Nparray', h3e: 'Nparray',
                   h4e: 'Nparray') -> 'FqeDataSet':
        """
        Applies a one-, two-, three-, and four-body operator specified by the tuple of numpy arrays.
        The result will be returned as a FqeDataSet object. self is unchanged.
        """
        norb = self._norb
        assert h4e.shape == (norb * 2, norb * 2, norb * 2, norb * 2, norb * 2,
                             norb * 2, norb * 2, norb * 2)

        nh1e = numpy.copy(h1e)
        nh2e = numpy.copy(h2e)
        nh3e = numpy.copy(h3e)

        for i in range(norb * 2):
            for j in range(norb * 2):
                for k in range(norb * 2):
                    nh1e[:, :] -= h4e[:, j, i, k, j, i, k, :]
                    for l in range(norb * 2):
                        nh2e[i, j, :, :] += (h4e[j, l, i, k, l, k, :, :] \
                                             + h4e[i, j, l, k, l, k, :, :] \
                                             + h4e[i, l, k, j, l, k, :, :] \
                                             + h4e[j, i, k, l, l, k, :, :] \
                                             + h4e[i, k, j, l, k, :, l, :] \
                                             + h4e[j, i, k, l, k, :, l, :] \
                                             + h4e[i, j, k, l, :, k, l, :])
                        nh3e[i, j, k, :, :, :] += \
                            (h4e[k, i, j, l, l, :, :, :] \
                             + h4e[j, i, l, k, l, :, :, :] \
                             + h4e[i, l, j, k, l, :, :, :] \
                             + h4e[i, k, j, l, :, l, :, :] \
                             + h4e[i, j, l, k, :, l, :, :] \
                             + h4e[i, j, k, l, :, :, l, :])

        out = self._apply123(nh1e, nh2e, nh3e)

        dvec = self._calculate_dvec()
        evec = self._calculate_evec(dvec)

        for key, sector in evec.items():
            evec[key] = numpy.transpose(numpy.tensordot(h4e,
                                                        sector,
                                                        axes=((2, 6, 3, 7),
                                                              (0, 1, 2, 3))),
                                        axes=(0, 2, 1, 3, 4, 5))

        dvec2 = copy.deepcopy(dvec)
        for i in range(norb * 2):
            for j in range(norb * 2):
                for key in evec:
                    dvec2[key][:, :, :, :] = evec[key][i, j, :, :, :, :]

                cvec = self._calculate_coeff_with_dvec(dvec2)

                for key in dvec:
                    dvec[key][i, j, :, :] = cvec[key][:, :]

        result = self._calculate_coeff_with_dvec(dvec)
        for key, fsector in out._data.items():
            fsector.coeff += result[key]
        return out

    def apply_individual_nbody(self, coeff: complex, daga: List[int],
                               undaga: List[int], dagb: List[int],
                               undagb: List[int]) -> 'FqeDataSet':
        """
        Apply function with an individual operator represented in arrays,
        which can handle spin-nonconserving operators and returns the result

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
            FqeDataSet: FqeData object that stores the result of application
        """
        out = self.empty_copy()
        out.apply_individual_nbody_accumulate(coeff, self, daga, undaga, dagb,
                                              undagb)
        return out

    def apply_individual_nbody_accumulate(self, coeff: complex,
                                          idata: 'FqeDataSet', daga: List[int],
                                          undaga: List[int], dagb: List[int],
                                          undagb: List[int]) -> None:
        """
        Apply function with an individual operator represented in arrays,
        which can handle spin-nonconserving operators

        Args:
            coeff (complex): scalar coefficient to be multiplied to the result

            idata (FqeDataSet): input FqeDataSet to which the operators are applied

            daga (List[int]): indices corresponding to the alpha creation \
                operators in the Hamiltonian

            undaga (List[int]): indices corresponding to the alpha annihilation \
                operators in the Hamiltonian

            dagb (List[int]): indices corresponding to the beta creation \
                operators in the Hamiltonian

            undagb (List[int]): indices corresponding to the beta annihilation \
                operators in the Hamiltonian
        """
        assert len(daga) + len(dagb) == len(undaga) + len(undagb)
        nda = len(daga) - len(undaga)

        for (nalpha, nbeta), source in idata._data.items():
            if (nalpha + nda, nbeta - nda) in self._data.keys():
                target = self._data[(nalpha + nda, nbeta - nda)]

                ualphamap = numpy.zeros((source.lena(), 3), dtype=numpy.uint64)
                ubetamap = numpy.zeros((source.lenb(), 3), dtype=numpy.uint64)

                acount = source._core.make_mapping_each(ualphamap, True, daga,
                                                        undaga)
                bcount = source._core.make_mapping_each(ubetamap, False, dagb,
                                                        undagb)
                ualphamap = ualphamap[:acount, :]
                ubetamap = ubetamap[:bcount, :]

                alphamap = numpy.zeros((acount, 3), dtype=numpy.int64)
                betamap = numpy.zeros((bcount, 3), dtype=numpy.int64)

                alphamap[:, 0] = ualphamap[:, 0]
                for i in range(acount):
                    alphamap[i, 1] = target._core.index_alpha(ualphamap[i, 1])
                alphamap[:, 2] = 1 - 2 * ualphamap[:, 2]

                betamap[:, 0] = ubetamap[:, 0]
                for i in range(bcount):
                    betamap[i, 1] = target._core.index_beta(ubetamap[i, 1])
                betamap[:, 2] = 1 - 2 * ubetamap[:, 2]

                if fqe.settings.use_accelerated_code:
                    if alphamap.size != 0 and betamap.size != 0:
                        pfac = (-1)**((len(dagb) + len(undagb)) * nalpha)
                        sourceb_vec = numpy.array(betamap[:, 0])
                        targetb_vec = numpy.array(betamap[:, 1])
                        parityb_vec = numpy.array(betamap[:, 2]) * pfac
                        _apply_individual_nbody1_accumulate(
                            coeff, target.coeff, source.coeff, alphamap,
                            targetb_vec, sourceb_vec, parityb_vec)
                else:
                    for sourcea, targeta, paritya in alphamap:
                        paritya *= (-1)**((len(dagb) + len(undagb)) * nalpha)
                        for sourceb, targetb, parityb in betamap:
                            work = coeff * source.coeff[sourcea, sourceb]
                            target.coeff[targeta,
                                         targetb] += work * paritya * parityb

    def evolve_individual_nbody(self, time: float, coeff: complex,
                                daga: List[int], undaga: List[int],
                                dagb: List[int],
                                undagb: List[int]) -> 'FqeDataSet':
        """
        This code time-evolves a wave function with an individual n-body generator
        which is spin-nonconserving. It is assumed that :math:`T^2 = 0`.
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
            FqeDataSet: result is returned as an FqeDataSet object
        """

        def isolate_number_operators(dag: List[int], undag: List[int],
                                     dagwork: List[int], undagwork: List[int],
                                     number: List[int]):
            """
            This code isolate operators that are paired between creation and
            annhilation operators, since they are to be treated seperately.
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
        for key in self._data.keys():
            out._data[key].apply_cos_inplace(time, ncoeff, numbera + dagworka,
                                             undagworka, numberb + dagworkb,
                                             undagworkb)
            out._data[key].apply_cos_inplace(time, ncoeff, numbera + undagworka,
                                             dagworka, numberb + undagworkb,
                                             dagworkb)

        phase = (-1)**((len(daga)+len(undaga))*(len(dagb)+len(undagb)) \
                       + len(daga)*(len(daga)-1)//2 \
                       + len(dagb)*(len(dagb)-1)//2 \
                       + len(undaga)*(len(undaga)-1)//2 \
                       + len(undagb)*(len(undagb)-1)//2)

        out.apply_individual_nbody_accumulate(
            -1.0j * numpy.conj(coeff) * phase * sinfactor, self, undaga, daga,
            undagb, dagb)
        out.apply_individual_nbody_accumulate(-1.0j * coeff * sinfactor, self,
                                              daga, undaga, dagb, undagb)
        return out

    def rdm1(self, bra: Optional['FqeDataSet'] = None) -> Tuple['Nparray']:
        """
        Computes 1-particle RDMs. When bra is specified, it computes a transition RDM

        Args:
            bradata (optional, FqeDataSet): FqeDataSet for the bra wavefunction. When \
                not given, the ket function is also used for the bra wavefunction

        Returns:
            Tuple[Nparray]: tuple of length 1 that contains numpy array for 1RDM
        """
        assert bra is None or self._data.keys() == bra._data.keys()

        dvec2 = self._calculate_dvec() if bra is None else bra._calculate_dvec()
        out = None
        for key in self._data.keys():
            tmp = numpy.einsum('jikl, kl->ij', dvec2[key].conj(),
                               self._data[key].coeff)
            out = tmp if out is None else (out + tmp)
        return (out,)

    def rdm12(self, bra: Optional['FqeDataSet'] = None
             ) -> Tuple['Nparray', 'Nparray']:
        """
        Computes 1- and 2-particle RDMs. When bra is specified, it computes a transition RDMs

        Args:
            bradata (optional, FqeDataSet): FqeDataSet for the bra wavefunction. When \
                not given, the ket function is also used for the bra wavefunction

        Returns:
            Tuple[Nparray]: tuple of length 2 that contains numpy array for 1 \
                and 2RDM
        """
        assert bra is None or self._data.keys() == bra._data.keys()

        dvec = self._calculate_dvec()
        dvec2 = dvec if bra is None else bra._calculate_dvec()

        out1 = numpy.empty(0)
        out2 = numpy.empty(0)
        for key in self._data.keys():
            tmp1 = numpy.tensordot(dvec2[key].conj(), self._data[key].coeff).T
            tmp2 = numpy.transpose(numpy.tensordot(
                dvec2[key].conj(), dvec[key], axes=((2, 3), (2, 3))),
                                   axes=(1, 2, 0, 3)) * (-1.0)
            if out1.shape == (0,):
                out1 = tmp1
            else:
                out1 = out1 + tmp1
            if out2.shape == (0,):
                out2 = tmp2
            else:
                out2 = out2 + tmp2

        for i in range(out1.shape[0]):
            out2[:, i, i, :] += out1[:, :]
        return (out1, out2)

    def rdm123(self,
               bra: Optional['FqeDataSet'] = None,
               dvec: Optional[Dict[Tuple[int, int], 'Nparray']] = None,
               dvec2: Optional[Dict[Tuple[int, int], 'Nparray']] = None,
               evec2: Optional[Dict[Tuple[int, int], 'Nparray']] = None
              ) -> Tuple['Nparray', 'Nparray', 'Nparray']:
        """
        Computes 1-, 2-, and 3-particle RDMs. When bra is specified, it computes a transition RDMs

        Args:
            bradata (optional, FqeDataSet): FqeDataSet for the bra wavefunction. When \
                not given, the ket function is also used for the bra wavefunction

        Returns:
            Tuple[Nparray]: tuple of length 3 that contains numpy array for 1, \
                2, and 3RDM
        """
        assert bra is None or self._data.keys() == bra._data.keys()

        if dvec is None:
            dvec = self._calculate_dvec()
        if dvec2 is None:
            dvec2 = dvec if bra is None else bra._calculate_dvec()
        if evec2 is None:
            evec2 = self._calculate_evec(dvec2)

        out1 = numpy.empty(0)
        out2 = numpy.empty(0)
        out3 = numpy.empty(0)
        for key in self._data.keys():
            tmp1 = numpy.tensordot(dvec2[key].conj(), self._data[key].coeff).T
            tmp2 = numpy.transpose(numpy.tensordot(
                dvec2[key].conj(), dvec[key], axes=((2, 3), (2, 3))),
                                   axes=(1, 2, 0, 3)) * (-1.0)
            tmp3 = numpy.transpose(numpy.tensordot(
                evec2[key].conj(), dvec[key], axes=((4, 5), (2, 3))),
                                   axes=(3, 1, 4, 2, 0, 5)) * (-1.0)
            if out1.shape == (0,):
                out1 = tmp1
            else:
                out1 = out1 + tmp1
            if out2.shape == (0,):
                out2 = tmp2
            else:
                out2 = out2 + tmp2
            if out3.shape == (0,):
                out3 = tmp3
            else:
                out3 = out3 + tmp3

        nsize = out1.shape[0]
        for i in range(nsize):
            out2[:, i, i, :] += out1[:, :]
        for i in range(nsize):
            out3[:, i, :, i, :, :] -= out2[:, :, :, :]
            out3[:, :, i, :, i, :] -= out2[:, :, :, :]
            for j in range(nsize):
                out3[:, i, j, i, j, :] += out1[:, :]
                for k in range(nsize):
                    out3[j, k, i, i, :, :] -= out2[k, j, :, :]
        return (out1, out2, out3)

    def rdm1234(self, bra: Optional['FqeDataSet'] = None
               ) -> Tuple['Nparray', 'Nparray', 'Nparray', 'Nparray']:
        """
        Computes 1-, 2-, 3-, and 4-particle RDMs. When bra is specified, it
        computes a transition RDMs

        Args:
            bradata (optional, FqeDataSet): FqeDataSet for the bra wavefunction. When \
                not given, the ket function is also used for the bra wavefunction

        Returns:
            Tuple[Nparray]: tuple of length 4 that contains numpy array for 1, \
                2, 3, and 4RDM
        """
        assert bra is None or self._data.keys() == bra._data.keys()

        dvec = self._calculate_dvec()
        dvec2 = dvec if bra is None else bra._calculate_dvec()
        evec = self._calculate_evec(dvec)
        evec2 = evec if bra is None else bra._calculate_evec(dvec2)

        (out1, out2, out3) = self.rdm123(bra, dvec, dvec2, evec2)

        out4 = numpy.empty(0)
        for key in self._data.keys():
            tmp4 = numpy.transpose(numpy.tensordot(evec2[key].conj(),
                                                   evec[key],
                                                   axes=((4, 5), (4, 5))),
                                   axes=(3, 1, 4, 6, 2, 0, 5, 7))
            if out4.shape == (0,):
                out4 = tmp4
            else:
                out4 = out4 + tmp4

        nsize = out1.shape[0]
        for i in range(nsize):
            for j in range(nsize):
                for k in range(nsize):
                    out4[:, j, i, k, j, i, k, :] -= out1[:, :]
                    for l in range(nsize):
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

    def _calculate_dvec(self) -> Dict[Tuple[int, int], 'Nparray']:
        """Generate, using self.coeff as C_I,

        .. math::
            D^{J}_{ij} = \\sum_I \\langle J|a^\\dagger_i a_j|I \\rangle C_I

        """
        return self._calculate_dvec_with_coeff(self._data)

    def _calculate_dvec_fixed_j(self,
                                jorb: int) -> Dict[Tuple[int, int], 'Nparray']:
        """Generate, using self.coeff as C_I, for fixed j

        .. math::
            D^{J}_{ij} = \\sum_I \\langle J|a^\\dagger_i a_j|I \\rangle C_I

        """
        return self._calculate_dvec_with_coeff_fixed_j(self._data, jorb)

    def _calculate_evec(self, dvec: Dict[Tuple[int, int], 'Nparray']
                       ) -> Dict[Tuple[int, int], 'Nparray']:
        """Generate

        .. math::
            E^{J}_{klij} = \\sum_I \\langle J|a^\\dagger_k a_l|I \\rangle D^I_{ij}

        """
        norb = self._norb
        evec = {}
        for key, sector in dvec.items():
            evec[key] = numpy.zeros((norb * 2, norb * 2, norb * 2, norb * 2,
                                     sector.shape[2], sector.shape[3]),
                                    dtype=sector.dtype)

        #civec = copy.deepcopy(self._data)
        civec = dict()
        for key, value in self._data.items():
            civec[key] = FqeData(value._core.nalpha(), value._core.nbeta(),
                                 value.norb(), value._core, value._dtype)
        for i in range(norb * 2):
            for j in range(norb * 2):
                for key in dvec.keys():
                    civec[key].coeff[:, :] = dvec[key][i, j, :, :]

                dvec2 = self._calculate_dvec_with_coeff(civec)

                for key in evec:
                    evec[key][:, :, i, j, :, :] = dvec2[key][:, :, :, :]
        return evec

    def _calculate_dvec_with_coeff(self, data: Dict[Tuple[int, int], 'FqeData']
                                  ) -> Dict[Tuple[int, int], 'Nparray']:
        """Generate

        .. math::
            D^{J}_{ij} = \\sum_I \\langle J|a^\\dagger_i a_j|I \\rangle C_I

        """

        def to_array(maps, norb):
            nstate = len(maps[(0,)])
            arrays = numpy.zeros((norb, nstate, 3), dtype=numpy.int32)
            for key, value in maps.items():
                i = key[0]
                for k, data in enumerate(value):
                    arrays[i, k, 0] = data[0]
                    arrays[i, k, 1] = data[1]
                    arrays[i, k, 2] = data[2]
            return arrays

        norb = self._norb

        dvec = {}
        for key, sector in data.items():
            dvec[key] = numpy.zeros(
                (norb * 2, norb * 2, sector.lena(), sector.lenb()),
                dtype=sector.coeff.dtype)

        for (nalpha, nbeta), sector in data.items():
            dveca, dvecb = sector.calculate_dvec_spin()
            dvec[(nalpha, nbeta)][:norb, :norb, :, :] += dveca
            dvec[(nalpha, nbeta)][norb:, norb:, :, :] += dvecb

            # a^+ b |0>
            if nalpha + 1 <= norb and nbeta - 1 >= 0:
                dvec1 = dvec[(nalpha + 1, nbeta - 1)]
                (alphamap, betamap) = sector.get_fcigraph().find_mapping(1, -1)
                if fqe.settings.use_accelerated_code:
                    alpha_array = to_array(alphamap, norb)
                    beta_array = to_array(betamap, norb)
                    _calculate_dvec1(alpha_array, beta_array, norb, nalpha,
                                     nbeta, sector.coeff, dvec1)
                else:
                    for i in range(norb):
                        for j in range(norb):
                            for sourcea, targeta, paritya in alphamap[(i,)]:
                                paritya *= (-1)**nalpha
                                for sourceb, targetb, parityb in betamap[(j,)]:
                                    work = sector.coeff[sourcea, sourceb]
                                    dvec1[i, j + norb, targeta,
                                          targetb] += work * paritya * parityb
            # b^+ a |0>
            if nalpha - 1 >= 0 and nbeta + 1 <= norb:
                dvec1 = dvec[(nalpha - 1, nbeta + 1)]
                (alphamap, betamap) = sector.get_fcigraph().find_mapping(-1, 1)
                if fqe.settings.use_accelerated_code:
                    alpha_array = to_array(alphamap, norb)
                    beta_array = to_array(betamap, norb)
                    _calculate_dvec2(alpha_array, beta_array, norb, nalpha,
                                     nbeta, sector.coeff, dvec1)
                else:
                    for i in range(norb):
                        for j in range(norb):
                            for sourcea, targeta, paritya in alphamap[(j,)]:
                                paritya *= (-1)**(nalpha - 1)
                                for sourceb, targetb, parityb in betamap[(i,)]:
                                    work = sector.coeff[sourcea, sourceb]
                                    dvec1[i + norb, j, targeta,
                                          targetb] += work * paritya * parityb
        return dvec

    def _calculate_dvec_with_coeff_fixed_j(
            self, data: Dict[Tuple[int, int], 'FqeData'],
            jorb: int) -> Dict[Tuple[int, int], 'Nparray']:
        """Generate, for fixed j,

        .. math::
            D^{J}_{ij} = \\sum_I \\langle J|a^\\dagger_i a_j|I \\rangle C_I

        """

        def to_array(maps, norb):
            nstate = len(maps[(0,)])
            arrays = numpy.zeros((norb, nstate, 3), dtype=numpy.int32)
            for key, value in maps.items():
                i = key[0]
                for k, data in enumerate(value):
                    arrays[i, k, 0] = data[0]
                    arrays[i, k, 1] = data[1]
                    arrays[i, k, 2] = data[2]
            return arrays

        norb = self._norb
        norb = self._norb

        dvec = {}
        for key, sector in data.items():
            dvec[key] = numpy.zeros((norb * 2, sector.lena(), sector.lenb()),
                                    dtype=sector.coeff.dtype)

        for (nalpha, nbeta), sector in data.items():
            dvec0 = sector.calculate_dvec_spin_fixed_j(jorb)
            if jorb < norb:
                # a^+ a |0>
                dvec[(nalpha, nbeta)][:norb, :, :] += dvec0
            else:
                # b^+ b |0>
                dvec[(nalpha, nbeta)][norb:, :, :] += dvec0

            if jorb < norb:
                # b^+ a |0>
                if nalpha - 1 >= 0 and nbeta + 1 <= norb:
                    dvec1 = dvec[(nalpha - 1, nbeta + 1)]
                    (alphamap,
                     betamap) = sector.get_fcigraph().find_mapping(-1, 1)
                    if fqe.settings.use_accelerated_code:
                        alpha_array = to_array(alphamap, norb)
                        beta_array = to_array(betamap, norb)
                        _calculate_dvec1_j(alpha_array, beta_array, norb, jorb,
                                           nalpha, nbeta, sector.coeff, dvec1)
                    else:
                        for i in range(norb):
                            for sourcea, targeta, paritya in alphamap[(jorb,)]:
                                paritya *= (-1)**(nalpha - 1)
                                for sourceb, targetb, parityb in betamap[(i,)]:
                                    work = sector.coeff[sourcea, sourceb]
                                    dvec1[i + norb, targeta,
                                          targetb] += work * paritya * parityb
            else:
                # a^+ b |0>
                if nalpha + 1 <= norb and nbeta - 1 >= 0:
                    dvec1 = dvec[(nalpha + 1, nbeta - 1)]
                    (alphamap,
                     betamap) = sector.get_fcigraph().find_mapping(1, -1)
                    if fqe.settings.use_accelerated_code:
                        alpha_array = to_array(alphamap, norb)
                        beta_array = to_array(betamap, norb)
                        _calculate_dvec2_j(alpha_array, beta_array, norb, jorb,
                                           nalpha, nbeta, sector.coeff, dvec1)
                    else:
                        for i in range(norb):
                            for sourcea, targeta, paritya in alphamap[(i,)]:
                                paritya *= (-1)**nalpha
                                for sourceb, targetb, parityb in betamap[(
                                        jorb - norb,)]:
                                    work = sector.coeff[sourcea, sourceb]
                                    dvec1[i, targeta,
                                          targetb] += work * paritya * parityb
        return dvec

    def _calculate_coeff_with_dvec(self, dvec: Dict[Tuple[int, int], 'Nparray']
                                  ) -> Dict[Tuple[int, int], 'Nparray']:
        """Generate

        .. math::
            C_I = \\sum_J \\langle I|a^\\dagger_i a_j|J \\rangle D^J_{ij}

        """

        def to_array(maps, norb):
            nstate = len(maps[(0,)])
            arrays = numpy.zeros((norb, nstate, 3), dtype=numpy.int32)
            for key, value in maps.items():
                i = key[0]
                for k, data in enumerate(value):
                    arrays[i, k, 0] = data[0]
                    arrays[i, k, 1] = data[1]
                    arrays[i, k, 2] = data[2]
            return arrays

        norb = self._norb

        out = {}
        for key, sector in self._data.items():
            out[key] = numpy.zeros((sector.lena(), sector.lenb()),
                                   dtype=sector.coeff.dtype)

        for (nalpha, nbeta), sector in self._data.items():
            assert (nalpha, nbeta) in out.keys()
            dvec0 = dvec[(nalpha, nbeta)]
            out0 = out[(nalpha, nbeta)]
            # <0| a^+ a |dvec>
            if nalpha > 0:
                for i in range(norb):
                    for j in range(norb):
                        for source, target, parity in sector.alpha_map(j, i):
                            out0[source, :] += dvec0[i, j, target, :] * parity
            # <0| b^+ b |dvec>
            if nbeta > 0:
                for i in range(norb):
                    for j in range(norb):
                        for source, target, parity in sector.beta_map(j, i):
                            out0[:, source] += dvec0[i + norb, j +
                                                     norb, :, target] * parity
            # <0| b^+ a |dvec>
            if nalpha + 1 <= norb and nbeta - 1 >= 0:
                dvec1 = dvec[(nalpha + 1, nbeta - 1)]
                (alphamap, betamap) = sector.get_fcigraph().find_mapping(1, -1)
                if fqe.settings.use_accelerated_code:
                    alpha_array = to_array(alphamap, norb)
                    beta_array = to_array(betamap, norb)
                    for i in range(norb):
                        for j in range(norb):
                            _calculate_coeff1(alpha_array, beta_array, norb, i,
                                              j, nalpha, nbeta, dvec1, out0)
                else:
                    for i in range(norb):
                        for j in range(norb):
                            for sourcea, targeta, paritya in alphamap[(j,)]:
                                paritya *= (-1)**nalpha
                                for sourceb, targetb, parityb in betamap[(i,)]:
                                    work = dvec1[i + norb, j, targeta, targetb]
                                    out0[sourcea,
                                         sourceb] += work * paritya * parityb
            # <0| a^+ b | dvec>
            if nalpha - 1 >= 0 and nbeta + 1 <= norb:
                dvec1 = dvec[(nalpha - 1, nbeta + 1)]
                (alphamap, betamap) = sector.get_fcigraph().find_mapping(-1, 1)
                if fqe.settings.use_accelerated_code:
                    alpha_array = to_array(alphamap, norb)
                    beta_array = to_array(betamap, norb)
                    for i in range(norb):
                        for j in range(norb):
                            _calculate_coeff2(alpha_array, beta_array, norb, i,
                                              j, nalpha, nbeta, dvec1, out0)
                else:
                    for i in range(norb):
                        for j in range(norb):
                            for sourcea, targeta, paritya in alphamap[(i,)]:
                                paritya *= (-1)**(nalpha - 1)
                                for sourceb, targetb, parityb in betamap[(j,)]:
                                    work = dvec1[i, j + norb, targeta, targetb]
                                    out0[sourcea,
                                         sourceb] += work * paritya * parityb
        return out
