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

from typing import Dict, List, Set, Tuple
from fqe.fci_graph import FciGraph
from fqe.util import alpha_beta_electrons
from fqe.bitstring import integer_index, count_bits_between
from fqe.bitstring import count_bits_above, unset_bit

Spinmap = Dict[Tuple[int, ...], List[Tuple[int, int, int]]]


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
            maxparticle (int) - the maximum particle number difference up to which \
                the sectors are to be linked

            maxspin (int) - the maximum spin number difference up to which the \
                sectors are to be linked

            params (List[List[int]]) - a list of parameter lists.  The parameter \
                lists are comprised of

                p[0] (int) - number of particles;
                p[1] (int) - z component of spin angular momentum;
                p[2] (int) - number of spatial orbitals

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
            self.link()

    def link(self) -> None:
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
            graph (FciGraph) - a FciGraph object to be appended
        """
        self._dataset[(graph.nalpha(), graph.nbeta())] = graph
        self.link()

    def _sectors_link(self, isec: 'FciGraph', jsec: 'FciGraph') -> None:
        """
        Links two sectors and forms alpha and beta mapping. The mapping will be
        stored in the FciGraph objects.

        Args:
            isec (FciGraph) - one of the FciGraph objects to be linked

            jsec (FciGraph) - the other FciGraph objects to be linked
        """
        assert isec.norb() == jsec.norb()
        dna = jsec.nalpha() - isec.nalpha()
        dnb = jsec.nbeta() - isec.nbeta()

        def make_mapping_each(istrings, iindex, jindex, dnv):
            mapping_down = {}
            mapping_up = {}
            for source in istrings:
                # print()
                # print("Source ", source, np.binary_repr(source, width=isec.norb()))
                # print("1s-pos in Source ", integer_index(source))

                # lower the source to the target and keep track of parity
                # this allows us to get the <Target|a_{i}...|source> as +- 1
                comb = combinations(integer_index(source), dnv)
                for ops in comb:
                    # print("bits to be removed ", ops)
                    parity = (count_bits_above(source, ops[-1]) * len(ops)) % 2
                    target = unset_bit(source, ops[-1])
                    for iop in reversed(range(len(ops) - 1)):
                        # print("inside removing bit", ops[iop])
                        parity += ((iop + 1) * count_bits_between(
                            source, ops[iop], ops[iop + 1])) % 2
                        target = unset_bit(target, ops[iop])

                    # print("Target ", target, np.binary_repr(target, width=isec.norb()), "parity", parity)
                    source_index = iindex[source]
                    target_index = jindex[target]
                    factor = (-1)**parity

                    key = tuple(ops)
                    if key not in mapping_down:
                        mapping_down[key] = []
                        mapping_up[key] = []
                    mapping_down[key].append(
                        (source_index, target_index, factor))
                    mapping_up[key].append((target_index, source_index, factor))
            return mapping_down, mapping_up

        upa: Spinmap = {}
        upb: Spinmap = {}
        downa: Spinmap = {}
        downb: Spinmap = {}

        if dna != 0:
            # what is happening here?
            (iasec, jasec) = (isec, jsec) if dna < 0 else (jsec, isec)
            # print("iasec")
            # for ii in iasec.string_alpha_all():
            #     print(ii, np.binary_repr(ii, width=iasec.norb()))
            # print()
            # print("jasec")
            # for ii in jasec.string_alpha_all():
            #     print(ii, np.binary_repr(ii, width=jasec.norb()))

            downa, upa = make_mapping_each(iasec.string_alpha_all(),
                                           iasec.index_alpha_all(),
                                           jasec.index_alpha_all(), abs(dna))
            if dna > 0:
                downa, upa = upa, downa

        if dnb != 0:
            (ibsec, jbsec) = (isec, jsec) if dnb < 0 else (jsec, isec)
            downb, upb = make_mapping_each(ibsec.string_beta_all(),
                                           ibsec.index_beta_all(),
                                           jbsec.index_beta_all(), abs(dnb))
            if dnb > 0:
                downb, upb = upb, downb

        assert upa != {} or upb != {}
        isec.insert_mapping(dna, dnb, (downa, downb))
        jsec.insert_mapping(-dna, -dnb, (upa, upb))


if __name__ == "__main__":
    import numpy as np
    params = [[3, 3, 6], [3, 1, 6], [3, -1, 6], [2, 0, 6]]
    test = FciGraphSet(3, 3, params)
