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
"""FciGraphSet manages global bitstring mapping between different sectors.
   The data itself are stored in each FciGraph
"""

from itertools import combinations

from typing import Dict, List, Set, Tuple, TYPE_CHECKING

import numpy
import scipy
from scipy import special

import fqe.settings
from fqe.fci_graph import FciGraph
from fqe.util import alpha_beta_electrons
from fqe.bitstring import integer_index, count_bits_between
from fqe.bitstring import count_bits_above, unset_bit, reverse_integer_index
from fqe.bitstring import lexicographic_bitstring_generator

from fqe.lib.fci_graph import _make_mapping_each_set
from fqe.lib.bitstring import _lexicographic_bitstring_generator

if TYPE_CHECKING:
    from numpy import ndarray as Nparray

Spinmap = Dict[Tuple[int, ...], 'Nparray']


class FciGraphSet:
    """
    FciGraghSet is the global manager for string mapping. It generalizes
    alphamapping and betamapping in FciGraph and maps between different sectors.
    """

    def __init__(self,
                 maxparticle: int,
                 maxspin: int,
                 params: List[List[int]] = None) -> None:
        """
        params are the same as Wavefunction.

        Args:
            maxparticle (int): the maximum particle number difference up to which \
                the sectors are to be linked

            maxspin (int): the maximum spin number difference up to which the \
                sectors are to be linked

            params (List[List[int]]): a list of parameter lists.  The parameter \
                lists are comprised of

                p[0] (int): number of particles;

                p[1] (int): z component of spin angular momentum;

                p[2] (int): number of spatial orbitals

        """
        self._dataset: Dict[Tuple[int, int], 'FciGraph'] = {}
        self._maxparticle = maxparticle
        self._maxspin = maxspin
        self._linked: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()
        if params is not None:
            for param in params:
                assert len(param) == 3
                nalpha, nbeta = alpha_beta_electrons(param[0], param[1])
                self._dataset[(nalpha,
                               nbeta)] = FciGraph(nalpha, nbeta, param[2])
            self._link()

    def _link(self) -> None:
        """
        Links between all of the sectors in the self._dataset
        """
        for ikey, isec in self._dataset.items():
            for jkey, jsec in self._dataset.items():
                if ikey < jkey and not (ikey, jkey) in self._linked:
                    delta_na = jkey[0] - ikey[0]
                    delta_nb = jkey[1] - ikey[1]
                    if abs(delta_na + delta_nb) <= self._maxparticle and \
                        max(abs(delta_na), abs(delta_nb)) <= self._maxspin:
                        self._sectors_link(isec, jsec)
                        self._linked.add((ikey, jkey))

    def append(self, graph: 'FciGraph') -> None:
        """
        Add an FciGraph object to self._dataset and links it against all the exisiting sectors.

        Args:
            graph (FciGraph): a FciGraph object to be appended
        """
        self._dataset[(graph.nalpha(), graph.nbeta())] = graph
        self._link()

    def _sectors_link(self, isec: 'FciGraph', jsec: 'FciGraph') -> None:
        """
        Links two sectors and forms alpha and beta mapping. The mapping will be
        stored in the FciGraph objects.

        Args:
            isec (FciGraph): one of the FciGraph objects to be linked

            jsec (FciGraph): the other FciGraph objects to be linked
        """
        norb = isec.norb()
        assert isec.norb() == jsec.norb()
        dna = jsec.nalpha() - isec.nalpha()
        dnb = jsec.nbeta() - isec.nbeta()

        def make_mapping_each_set(istrings, dnv, norb, nele):
            nsize = int(special.binom(norb - dnv, nele - dnv))
            msize = int(special.binom(norb, dnv))

            mapping_down = numpy.zeros((msize, nsize, 3), dtype=numpy.uint64)
            mapping_up = numpy.zeros((msize, nsize, 3), dtype=numpy.uint64)

            combmap = lexicographic_bitstring_generator(dnv, norb)
            assert combmap.size == msize
            for anni in range(msize):
                mask = int(combmap[anni])
                ops = integer_index(mask)

                count = 0
                for isource in istrings:
                    source = int(isource)
                    if ((source & mask) ^ mask) != 0:
                        continue
                    parity = (count_bits_above(source, ops[-1]) * len(ops))
                    target = unset_bit(source, ops[-1])
                    for iop in reversed(range(len(ops) - 1)):
                        parity += (
                            (iop + 1) *
                            count_bits_between(source, ops[iop], ops[iop + 1]))
                        target = unset_bit(target, ops[iop])

                    mapping_down[anni, count, :] = source, target, parity
                    mapping_up[anni, count, :] = target, source, parity
                    count += 1
                assert count == nsize
            return mapping_down, mapping_up

        def _postprocess(spinmap, dnv, index0, index1):
            transformed: Spinmap = {}

            assert spinmap.shape[0] == int(special.binom(norb, dnv))
            combmap = lexicographic_bitstring_generator(dnv, norb)

            for index in range(spinmap.shape[0]):
                out = numpy.zeros(spinmap.shape[1:], dtype=numpy.int32)
                for i in range(out.shape[0]):
                    out[i, 0] = index0[spinmap[index, i, 0]]
                    out[i, 1] = index1[spinmap[index, i, 1]]
                    out[i, 2] = -1 if spinmap[index, i, 2] % 2 else 1

                key = tuple(integer_index(int(combmap[index])))
                transformed[key] = out
            return transformed

        if dna != 0:
            (iasec, jasec) = (isec, jsec) if dna < 0 else (jsec, isec)
            if fqe.settings.use_accelerated_code:
                ndowna, nupa = _make_mapping_each_set(iasec.string_alpha_all(),
                                                      abs(dna), norb,
                                                      iasec.nalpha())
            else:
                ndowna, nupa = make_mapping_each_set(iasec.string_alpha_all(),
                                                     abs(dna), norb,
                                                     iasec.nalpha())

            downa = _postprocess(ndowna, abs(dna), iasec.index_alpha_all(),
                                 jasec.index_alpha_all())
            upa = _postprocess(nupa, abs(dna), jasec.index_alpha_all(),
                               iasec.index_alpha_all())

            if dna > 0:
                downa, upa = upa, downa
        else:
            upa, downa = {}, {}

        if dnb != 0:
            (ibsec, jbsec) = (isec, jsec) if dnb < 0 else (jsec, isec)
            if fqe.settings.use_accelerated_code:
                ndownb, nupb = _make_mapping_each_set(ibsec.string_beta_all(),
                                                      abs(dnb), norb,
                                                      ibsec.nbeta())
            else:
                ndownb, nupb = make_mapping_each_set(ibsec.string_beta_all(),
                                                     abs(dnb), norb,
                                                     ibsec.nbeta())

            downb = _postprocess(ndownb, abs(dnb), ibsec.index_beta_all(),
                                 jbsec.index_beta_all())
            upb = _postprocess(nupb, abs(dnb), jbsec.index_beta_all(),
                               ibsec.index_beta_all())

            if dnb > 0:
                downb, upb = upb, downb
        else:
            upb, downb = {}, {}

        assert upa != {} or upb != {}
        isec.insert_mapping(dna, dnb, (downa, downb))
        jsec.insert_mapping(-dna, -dnb, (upa, upb))
