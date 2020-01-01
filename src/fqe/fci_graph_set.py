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

"""FciGraphSet manages global bitstring mapping between different sectors.
   The data itself are stored in each FciGraph 
"""

import copy
from itertools import combinations

from typing import List
import fqe
from fqe.fci_graph import FciGraph
from fqe.util import alpha_beta_electrons
from fqe.bitstring import integer_index, count_bits, count_bits_between, count_bits_above, unset_bit


class FciGraphSet:
    """
    FciGraghSet is the global manager for string mapping. It generalizes alphamapping and betamapping in FciGraph and maps between different sectors.
    """
    def __init__(self, maxparticle: int, maxspin: int, params: List[List[int]] = None) -> None:
        """
        params are the same as Wavefunction
        maxparticle and maxspin are the number up to which the sectors are to be linked
        """
        self._dataset = {}
        self._maxparticle = maxparticle
        self._maxspin = maxspin
        self._linked = set()
        if params is not None:
            for param in params:
                assert len(param) == 3
                nalpha, nbeta = alpha_beta_electrons(param[0], param[1])
                self._dataset[(nalpha, nbeta)] = FciGraph(nalpha, nbeta, param[2])
            self.link()

    
    def link(self):
        """
        Links between all of the sectors in the self._dataset
        """
        for ikey,isec in self._dataset.items():
            for jkey,jsec in self._dataset.items():
                if ikey < jkey and not (ikey, jkey) in self._linked:
                    delta_na = jkey[0] - ikey[0]
                    delta_nb = jkey[1] - ikey[1]
                    if abs(delta_na + delta_nb) <= self._maxparticle and max(abs(delta_na), abs(delta_nb)) <= self._maxspin:
                        self.sectors_link(isec, jsec)
                        self._linked.add((ikey, jkey))


    def append(self, graph: 'FciGraph'):
        """
        Add an FciGraph object to self._dataset and links it against all the exisiting sectors.
        """
        self._dataset[(graph.nalpha(), graph.nbeta())] = graph
        self.link()


    def sectors_link(self, isec: 'FciGraph', jsec: 'FciGraph'):
        """
        Links two sectors and forms alpha and beta mapping. The mapping will be stored in the FciGraph objects.
        """
        assert isec.norb() == jsec.norb()
        norb = isec.norb()
        dna = jsec.nalpha() - isec.nalpha()
        dnb = jsec.nbeta() - isec.nbeta()

        def make_mapping_each(istrings, iindex, jindex, dnv):
            mapping_down = {}
            mapping_up = {}
            for source in istrings:
                comb = combinations(integer_index(source), dnv)
                for ops in comb:
                    parity = (count_bits_above(source, ops[-1]) * len(ops)) % 2
                    target = unset_bit(source, ops[-1])
                    for iop in reversed(range(len(ops)-1)):
                        parity += ((iop+1) * count_bits_between(source, ops[iop], ops[iop+1])) % 2
                        target = unset_bit(target, ops[iop])
                    source_index = iindex[source]
                    target_index = jindex[target]
                    factor = (-1)**parity

                    key = tuple(ops)
                    if not key in mapping_down:
                        mapping_down[key] = []
                        mapping_up[key] = []
                    mapping_down[key].append((source_index, target_index, factor))
                    mapping_up[key].append((target_index, source_index, factor))
            return mapping_down, mapping_up

        upa = {}
        upb = {}
        downa = {}
        downb = {}

        if dna != 0:
            (iasec, jasec) = (isec, jsec) if dna < 0 else (jsec, isec)
            downa, upa = make_mapping_each(iasec.string_alpha_all(), iasec.index_alpha_all(), jasec.index_alpha_all(), abs(dna))
            if dna > 0:
                downa, upa = upa, downa

        if dnb != 0:
            (ibsec, jbsec) = (isec, jsec) if dnb < 0 else (jsec, isec)
            downb, upb = make_mapping_each(ibsec.string_beta_all(), ibsec.index_beta_all(), jbsec.index_beta_all(), abs(dnb))
            if dnb > 0:
                downb, upb = upb, downb

        assert upa != {} or upb != {}
        isec.insert_mapping(dna, dnb, (downa, downb))  
        jsec.insert_mapping(-dna, -dnb, (upa, upb))  
