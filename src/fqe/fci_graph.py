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
"""FciGraph stores the addressing scheme for the fqe_data structures.
"""

from typing import Dict, List, Tuple
from functools import lru_cache

from scipy.special import binom
import numpy
from numpy import ndarray as Nparray

from fqe.bitstring import integer_index, lexicographic_bitstring_generator
from fqe.util import init_bitstring_groundstate
from fqe.lib.fci_graph import _build_mapping_strings

Spinmap = Dict[Tuple[int, ...], Nparray]


def map_to_deexc(mappings, states, norbs):
    dexc = [[] for state in range(states)]
    for (i, j), values in mappings.items():
        for state, target, parity in values:
            dexc[target].append([state, i * norbs + j, parity])
    return numpy.asarray(dexc, dtype=numpy.int32)


@lru_cache()
def _get_Z_matrix(norb: int, nele: int) -> Nparray:
    """Builds the Z-matrix as given in eq.11 in 'A new determinant-based full
    configuration interaction method' (Knowles, Handy). Uses lru_cache for
    caching already calculated z matrices.

    Args:
        norb (int) - the number of spatial orbitals

        nele (int) - the number of electrons for a single spin case

        Returns:
            Z (Nparray) - The Z matrix for building addresses.
    """
    Z = numpy.zeros((nele, norb), dtype=numpy.int32)
    # If nele or norb == 0
    if Z.size == 0:
        return Z

    for k in range(1, nele):
        for ll in range(k, norb - nele + k + 1):
            Z[k - 1, ll - 1] = sum(
                binom(m, nele-k) - binom(m-1, nele-k-1)
                for m in range(norb - ll + 1, norb - k + 1))
    k = nele
    for ll in range(nele, norb + 1):
        Z[k - 1, ll - 1] = ll - nele
    return Z


class FciGraph:
    """ FciGraph contains the addressing system for the wavefunction.  Each
    determinant is considered as a product of alpha creation operators and
    beta operators acting on the vacuum in the manner {alpha ops}{beta ops}|>.
    To find any determinant in the model one needs the occupation index of the
    alpha orbitals and the beta orbitals.  From this information, any pointer
    into the wavefunction can be generated.
    """

    def __init__(self, nalpha: int, nbeta: int, norb: int) -> None:
        """
        Args:
            nalpha (int) - The number of alpha electrons

            nbeta (int) - The number of beta electrons

            norb (int) - The number of spatial orbitals such that the total number
                of orbitals is ntot = 2*norb.

            _alpha_map and _beta_map are  Dict[Tuple[int,int], List[Tuple[int,int,int]]]
        """
        self._norb = norb
        self._nalpha = nalpha
        self._nbeta = nbeta
        self._lena = int(binom(norb, nalpha))  # size of alpha-Hilbert space
        self._lenb = int(binom(norb, nbeta))  # size of beta-Hilbert space
        self._astr: List[int] = []  # string labels for alpha-Hilbert space
        self._bstr: List[int] = []  # string labels for beta-Hilbert space
        self._aind: Dict[int, int] = {}  # map string-binary to matrix index
        self._bind: Dict[int, int] = {}  # map string-binary to matrix index
        self._astr, self._aind = self._build_strings(self._nalpha, self._lena)
        self._bstr, self._bind = self._build_strings(self._nbeta, self._lenb)
        self._alpha_map: Spinmap = self._build_mapping(self._astr, self._nalpha)
        self._beta_map: Spinmap = self._build_mapping(self._bstr, self._nbeta)
        self._dexca = map_to_deexc(self._alpha_map, self._lena, self._norb)
        self._dexcb = map_to_deexc(self._beta_map, self._lenb, self._norb)

        self._fci_map: Dict[Tuple[int, ...], Tuple[Spinmap, Spinmap]] = {}

    def insert_mapping(self, dna: int, dnb: int,
                       mapping_pair: Tuple[Spinmap, Spinmap]) -> None:
        """
        Insert a new pair of alpha and beta mappings with a key that are the
        differences for the number of alpha and beta electrons.

        Args:
            dna (int) - the difference in the number of alpha electrons

            dnb (int) - the difference in the number of beta electrons

            mapping_pair (Tuple[Spinmap, Spinmap]) - mapping for alpha and \
                beta electrons
        """
        self._fci_map[(dna, dnb)] = mapping_pair

    def find_mapping(self, dna: int, dnb: int) -> Tuple[Spinmap, Spinmap]:
        """
        Returns the pair of mappings that corresponds to dna and dnb
        (difference in the number of electrons for alpha and beta)

        Args:
            dna (int) - the difference in the number of alpha electrons

            dnb (int) - the difference in the number of beta electrons

        Returns:
            (Tuple[Spinmap, Spinmap]) - mapping for alpha and beta electrons
        """
        assert (dna, dnb) in self._fci_map
        return self._fci_map[(dna, dnb)]

    def _build_mapping(self, strings: List[int], nele: int) -> Spinmap:
        """Construct the mapping of alpha string and beta string excitations
        for :math:`a^\\dagger_i a_j` from the bitstrings contained in the fci_graph.

        Args:
            strings (List[int]) - list of the determinant bitstrings

            index (Dict[int,int])) - list of the indices corresponding to the \
                determinant bitstrings
        """
        norb = self._norb
        return _build_mapping_strings(
            strings,
            _get_Z_matrix(norb, nele),
            nele,
            norb
        )

    def alpha_map(self, iorb: int, jorb: int) -> List[Tuple[int, int, int]]:
        """
        Returns the Knowles-Handy mapping (within this FciGraph) for alpha electrons for
        :math:`a^\\dagger_i a_j`

        Args:
            iorb (int) - orbital index for the creation operator

            jorb (int) - orbital index for the annhilation operator

        Returns:
            (List[Tuple[int, int, int]]) - array of string mapping with phases
        """
        assert (iorb, jorb) in self._alpha_map.keys()
        return self._alpha_map[(iorb, jorb)]

    def beta_map(self, iorb: int, jorb: int) -> List[Tuple[int, int, int]]:
        """
        Returns the Knowles-Handy mapping (within this FciGraph) for beta electrons for
        :math:`a^\\dagger_i a_j`

        Args:
            iorb (int) - orbital index for the creation operator

            jorb (int) - orbital index for the annhilation operator

        Returns:
            (List[Tuple[int, int, int]]) - array of string mapping with phases
        """
        assert (iorb, jorb) in self._beta_map.keys()
        return self._beta_map[(iorb, jorb)]

    def lena(self) -> int:
        """Return the number of alpha electrons
        """
        return self._lena

    def lenb(self) -> int:
        """Return the number of beta electrons
        """
        return self._lenb

    def nalpha(self) -> int:
        """Return the number of alpha electrons
        """
        return self._nalpha

    def nbeta(self) -> int:
        """Return the number of beta electrons
        """
        return self._nbeta

    def norb(self) -> int:
        """Return the number of beta electrons
        """
        return self._norb

    def _build_strings(self, nele: int,
                       length: int) -> Tuple[List[int], Dict[int, int]]:
        """Build all bitstrings for index the FCI and their lexicographic index
           for a single spin case.

        Args:
            nele (int) - number of electrons in this graph

            length (int) - the largest dimension of the graph

        Returns:
            An initialized string array for accessing configurations in the FCI
        """
        grs = init_bitstring_groundstate(nele)
        blist = lexicographic_bitstring_generator(grs, self._norb)
        string_list = [0 for _ in range(length)
                      ]  # strings in lexicographic order
        index_list = {}  # map bitsting to its lexicographic address
        for i in range(length):
            wbit = blist[i]  # integer that is the spin-bitstring
            occ = integer_index(wbit)
            # get the lexicographic address of the bitstring
            address = self._build_string_address(nele, self._norb, occ)
            string_list[address] = wbit
            index_list[wbit] = address

        return string_list, index_list

    def string_alpha(self, address: int) -> int:
        """Retrieve the alpha bitstring reprsentation stored at the address

        Args:
            address (int) - an integer pointing into the fcigraph

        Returns:
            (bitstring) - an occupation representation of the configuration
        """
        return self._astr[address]

    def string_beta(self, address: int) -> int:
        """Retrieve the beta bitstring reprsentation stored at the address

        Args:
            address (int) - an integer pointing into the fcigraph

        Returns:
            (bitstring) - an occupation representation of the configuration
        """
        return self._bstr[address]

    def string_alpha_all(self) -> List[int]:
        """Return all bitstrings for alpha occupied orbitals
        """
        return self._astr

    def string_beta_all(self) -> List[int]:
        """Return all bitstrings for beta occupied orbitals
        """
        return self._bstr

    def index_alpha(self, bit_string: int) -> int:
        """Retrieve the alpha index stored by it's bitstring

        Args:
            bit_string (bitstring) - an occupation representation of the configuration

        Returns:
            address (int) - an integer pointing into the fcigraph
        """
        return self._aind[bit_string]

    def index_beta(self, bit_string: int) -> int:
        """Retrieve the beta bitstring reprsentation stored at the address

        Args:
            bit_string (bitstring) - an occupation representation of the configuration

        Returns:
            address (int) - an integer pointing into the fcigraph
        """
        return self._bind[bit_string]

    def index_alpha_all(self) -> Dict[int, int]:
        """Return the index and the corresponding occupation string for all
        alpha strings
        """
        return self._aind

    def index_beta_all(self) -> Dict[int, int]:
        """Return the index and the corresponding occupation string for all
        beta strings
        """
        return self._bind

    def _build_string_address(self, nele: int, norb: int,
                              occupation: List[int]) -> int:
        """Given a list of occupied orbitals in ascending order generate the
        index into the CI matrix.

        Args:
            nele (int) - the number of electrons for a single spin case

            norb (int) - the number of spatial orbitals

            occupation (list[int]) - a list with integers indicating the index
                of the occupied orbitals starting from 0

        Returns:
            address (int) - A pointer into a spin a block of the CI addressing
                system
        """
        Z = _get_Z_matrix(norb, nele)
        return sum(Z[i, occupation[i]] for i in range(nele))

    def _get_block_mappings(self, max_states=100, jorb=None):
        from itertools import product

        def split(maps, totstates, max_states):
            totmaps = []
            for (io, jo), mp in maps.items():
                if jorb is not None and jo != jorb:
                    continue

                index = io * self.norb() + jo if jorb is None else io
                totmaps.extend([[index, t, s, p] for s, t, p in mp])

            totmaps = numpy.asarray(
                sorted(totmaps, key=lambda x: (x[1], x[0])),
                dtype=numpy.int32
            )
            rangelist = list(range(0, totstates, max_states)) + [totstates]
            dat = []
            for begin, end in zip(rangelist, rangelist[1:]):
                indexes1 = numpy.logical_and(end > totmaps[:, 1],
                                             totmaps[:, 1] >= begin)
                indexes2 = numpy.logical_and(end > totmaps[:, 2],
                                             totmaps[:, 2] >= begin)

                map1 = numpy.sort(totmaps[indexes1].view('i4,i4,i4,i4'),
                                  order=['f0', 'f1']).view(numpy.int32)
                map2 = numpy.sort(totmaps[indexes2].view('i4,i4,i4,i4'),
                                  order=['f0', 'f1']).view(numpy.int32)
                dat.append([range(begin, end), map1, map2])
            return dat

        adat = split(self._alpha_map, self.lena(), max_states)
        bdat = split(self._beta_map, self.lenb(), max_states)
        return [(ar, br, (am1, am2), (bm1, bm2))
                for (ar, am1, am2), (br, bm1, bm2) in product(adat, bdat)]


if __name__ == "__main__":
    fcig = FciGraph(4, 4, 8)
    print(fcig._alpha_map[(3, 7)])
