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

import copy
from typing import Dict, List, Tuple
from functools import lru_cache

from scipy.special import binom
import numpy
from numpy import ndarray as Nparray

from fqe.bitstring import integer_index, lexicographic_bitstring_generator, \
    get_bit, set_bit, unset_bit, count_bits_above, count_bits_between
from fqe.lib.fci_graph import _build_mapping_strings, _map_deexc, \
                              _calculate_string_address, \
                              _c_map_to_deexc_alpha_icol, \
                              _make_mapping_each, _calculate_Z_matrix
import fqe.settings

Spinmap = Dict[Tuple[int, ...], Nparray]


def map_to_deexc(mappings: Spinmap, states: int, norbs: int,
                 nele: int) -> Nparray:
    """Build map to de-excitations from excitations.

    Args:
        mappings (Spinmap): Map of excitations

        states (int): number of states

        norb (int): number of orbitals

        nele (int): number of electrons

    Returns:
        (Nparray): de-excitations
    """
    lk = nele * (norbs - nele + 1)
    dexc = numpy.zeros((states, lk, 3), dtype=numpy.int32)
    index = numpy.zeros((states,), dtype=numpy.uint32)
    for (i, j), values in mappings.items():
        idx = i * norbs + j
        if fqe.settings.use_accelerated_code:
            _map_deexc(dexc, values, index, idx)
        else:
            for state, target, parity in values:
                dexc[target, index[target], :] = state, idx, parity
                index[target] += 1
    return dexc


@lru_cache()
def _get_Z_matrix(norb: int, nele: int) -> Nparray:
    """Builds the Z-matrix as given in eq.11 in 'A new determinant-based full
    configuration interaction method' (Knowles, Handy). Uses lru_cache for
    caching already calculated z matrices.

    Args:
        norb (int): the number of spatial orbitals

        nele (int): the number of electrons for a single spin case

    Returns:
        Z (Nparray): The Z matrix for building addresses.
    """
    Z = numpy.zeros((nele, norb), dtype=numpy.int32)

    if Z.size == 0:
        return Z

    if fqe.settings.use_accelerated_code:
        _calculate_Z_matrix(Z, norb, nele)
    else:
        for k in range(1, nele):
            for ll in range(k, norb - nele + k + 1):
                Z[k - 1, ll - 1] = sum(
                    binom(m, nele - k) - binom(m - 1, nele - k - 1)
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
            nalpha (int): The number of alpha electrons

            nbeta (int): The number of beta electrons

            norb (int): The number of spatial orbitals such that the total number
                of orbitals is ntot = 2*norb.

            _alpha_map and _beta_map are  Dict[Tuple[int,int], Nparray]
        """
        if norb < 0:
            raise ValueError(f'norb needs to be >= 0, passed value is {norb}')
        if nalpha < 0:
            raise ValueError(
                f'nalpha needs to be >= 0, passed value is {nalpha}')
        if nbeta < 0:
            raise ValueError(f'nbeta needs to be >= 0, passed value is {nbeta}')
        if nalpha > norb:
            raise ValueError(
                f'nalpha needs to be <= norb, passed value is {nbeta}')
        if nbeta > norb:
            raise ValueError(
                f'nbeta needs to be <= norb, passed value is {nbeta}')

        self._norb = norb
        self._nalpha = nalpha
        self._nbeta = nbeta
        self._lena = int(binom(norb, nalpha))  # size of alpha-Hilbert space
        self._lenb = int(binom(norb, nbeta))  # size of beta-Hilbert space
        self._astr: Nparray = None  # string labels for alpha-Hilbert space
        self._bstr: Nparray = None  # string labels for beta-Hilbert space
        self._aind: Dict[int, int] = {}  # map string-binary to matrix index
        self._bind: Dict[int, int] = {}  # map string-binary to matrix index
        self._astr, self._aind = self._build_strings(self._nalpha, self._lena)
        self._bstr, self._bind = self._build_strings(self._nbeta, self._lenb)
        self._alpha_map: Spinmap = self._build_mapping(self._astr, self._nalpha,
                                                       self._aind)
        self._beta_map: Spinmap = self._build_mapping(self._bstr, self._nbeta,
                                                      self._bind)
        self._dexca = map_to_deexc(self._alpha_map, self._lena, self._norb,
                                   self._nalpha)
        self._dexcb = map_to_deexc(self._beta_map, self._lenb, self._norb,
                                   self._nbeta)

        self._fci_map: Dict[Tuple[int, ...], Tuple[Spinmap, Spinmap]] = {}

    def alpha_beta_transpose(self):
        """
        Creates a new FciGraph object where the alpha-electrons and
        beta-electrons and their corresponding determinants are switched.

        Returns:
            (FciGraph): The transposed instance
        """
        out = copy.deepcopy(self)
        out._nalpha, out._nbeta = out._nbeta, out._nalpha
        out._lena, out._lenb = out._lenb, out._lena
        out._astr, out._bstr = out._bstr, out._astr
        out._aind, out._bind = out._bind, out._aind
        out._alpha_map, out._beta_map = out._beta_map, out._alpha_map
        out._dexca, out._dexcb = out._dexcb, out._dexca
        return out

    def insert_mapping(self, dna: int, dnb: int,
                       mapping_pair: Tuple[Spinmap, Spinmap]) -> None:
        """
        Insert a new pair of alpha and beta mappings with a key that are the
        differences for the number of alpha and beta electrons.

        Args:
            dna (int): the difference in the number of alpha electrons

            dnb (int): the difference in the number of beta electrons

            mapping_pair (Tuple[Spinmap, Spinmap]): mapping for alpha and
                beta electrons
        """
        self._fci_map[(dna, dnb)] = mapping_pair

    def find_mapping(self, dna: int, dnb: int) -> Tuple[Spinmap, Spinmap]:
        """
        Returns the pair of mappings that corresponds to dna and dnb
        (difference in the number of electrons for alpha and beta)

        Args:
            dna (int): the difference in the number of alpha electrons

            dnb (int): the difference in the number of beta electrons

        Returns:
            (Tuple[Spinmap, Spinmap]): mapping for alpha and beta electrons
        """
        return self._fci_map[(dna, dnb)]

    def _build_mapping(self, strings: Nparray, nele: int,
                       index: Dict[int, int]) -> Spinmap:
        """Construct the mapping of alpha string and beta string excitations
        for :math:`a^\\dagger_i a_j` from the bitstrings contained in the
        fci_graph.

        Args:
            strings (Nparray): list of the determinant bitstrings

            nele (int): number of electrons in the the determinants

            index (Dict[int,int])): list of the indices corresponding to the
                determinant bitstrings
        """
        norb = self._norb

        if fqe.settings.use_accelerated_code:
            return _build_mapping_strings(strings, _get_Z_matrix(norb, nele),
                                          nele, norb)
        else:
            out = {}
            for iorb in range(norb):  # excitation
                for jorb in range(norb):  # deexcitation
                    value = []
                    for string in strings:
                        if get_bit(string, jorb) and not get_bit(string, iorb):
                            parity = count_bits_between(string, iorb, jorb)
                            sign = 1 if parity % 2 == 0 else -1
                            value.append(
                                (index[string],
                                 index[unset_bit(set_bit(string, iorb),
                                                 jorb)], sign))
                        elif iorb == jorb and get_bit(string, iorb):
                            value.append((index[string], index[string], 1))
                    out[(iorb, jorb)] = value

            # cast to numpy arrays
            return {
                k: numpy.asarray(v, dtype=numpy.int32).reshape(-1, 3)
                for k, v in out.items()
            }

    def alpha_map(self, iorb: int, jorb: int) -> List[Tuple[int, int, int]]:
        """
        Returns the Knowles-Handy mapping (within this FciGraph) for alpha electrons for
        :math:`a^\\dagger_i a_j`

        Args:
            iorb (int): orbital index for the creation operator

            jorb (int): orbital index for the annhilation operator

        Returns:
            (List[Tuple[int, int, int]]) - array of string mapping with phases
        """
        return self._alpha_map[(iorb, jorb)]

    def beta_map(self, iorb: int, jorb: int) -> List[Tuple[int, int, int]]:
        """
        Returns the Knowles-Handy mapping (within this FciGraph) for beta electrons for
        :math:`a^\\dagger_i a_j`

        Args:
            iorb (int): orbital index for the creation operator

            jorb (int): orbital index for the annhilation operator

        Returns:
            (List[Tuple[int, int, int]]): array of string mapping with phases
        """
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
                       length: int) -> Tuple[Nparray, Dict[int, int]]:
        """Build all bitstrings for index the FCI and their lexicographic index
           for a single spin case.

        Args:
            nele (int): number of electrons in this graph

            length (int): the largest dimension of the graph

        Returns:
            An initialized string array for accessing configurations in the FCI
        """

        norb = self._norb
        blist = lexicographic_bitstring_generator(nele, norb)

        if fqe.settings.use_accelerated_code:
            Z = _get_Z_matrix(norb, nele)
            string_list = _calculate_string_address(Z, nele, norb, blist)
        else:
            string_list = numpy.zeros((length,), dtype=numpy.uint64)
            for i in range(length):
                wbit = blist[i]
                occ = integer_index(int(wbit))
                address = self._build_string_address(nele, norb, occ)
                string_list[address] = wbit

        index_list = {}
        for address, wbit in enumerate(string_list):
            index_list[wbit] = address

        return string_list, index_list

    def string_alpha(self, address: int) -> int:
        """Retrieve the alpha bitstring representation stored at the address

        Args:
            address (int): an integer pointing into the fcigraph

        Returns:
            (bitstring): an occupation representation of the configuration
        """
        return self._astr[address]

    def string_beta(self, address: int) -> int:
        """Retrieve the beta bitstring representation stored at the address

        Args:
            address (int): an integer pointing into the fcigraph

        Returns:
            (bitstring): an occupation representation of the configuration
        """
        return self._bstr[address]

    def string_alpha_all(self) -> Nparray:
        """Return all bitstrings for alpha occupied orbitals
        """
        return self._astr

    def string_beta_all(self) -> Nparray:
        """Return all bitstrings for beta occupied orbitals
        """
        return self._bstr

    def index_alpha(self, bit_string: int) -> int:
        """Retrieve the alpha index stored by it's bitstring

        Args:
            bit_string (bitstring): an occupation representation of the configuration

        Returns:
            address (int): an integer pointing into the fcigraph
        """
        return self._aind[bit_string]

    def index_beta(self, bit_string: int) -> int:
        """Retrieve the beta bitstring reprsentation stored at the address

        Args:
            bit_string (bitstring): an occupation representation of the configuration

        Returns:
            address (int): an integer pointing into the fcigraph
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
            nele (int): the number of electrons for a single spin case

            norb (int): the number of spatial orbitals

            occupation (list[int]): a list with integers indicating the index
                of the occupied orbitals starting from 0

        Returns:
            address (int): A pointer into a spin a block of the CI addressing
                system
        """
        Z = _get_Z_matrix(norb, nele)
        return sum(Z[i, occupation[i]] for i in range(nele))

    def _get_block_mappings(self, max_states=100, jorb=None):
        """Internal function that blocks the mappings in pieces of max_states
        width for both alpha and beta determinants. When jorb is not None, get
        mappings are restricted to the ones with (i, j) == (i, jorb).
        """
        from itertools import product

        def split(maps, totstates, max_states):
            totmaps = []
            for (io, jo), mp in maps.items():
                if jorb is not None and jo != jorb:
                    continue

                index = io * self.norb() + jo if jorb is None else io
                totmaps.extend([[index, t, s, p] for s, t, p in mp])

            totmaps = numpy.asarray(sorted(totmaps, key=lambda x: (x[1], x[0])),
                                    dtype=numpy.int32).reshape(-1, 4)
            rangelist = list(range(0, totstates, max_states)) + [totstates]
            dat = []
            for begin, end in zip(rangelist, rangelist[1:]):
                indexes1 = numpy.logical_and(end > totmaps[:, 1],
                                             totmaps[:, 1] >= begin)

                mp = numpy.sort(totmaps[indexes1].view('i4,i4,i4,i4'),
                                order=['f0', 'f1']).view(numpy.int32)
                dat.append([range(begin, end), mp])
            return dat

        adat = split(self._alpha_map, self.lena(), max_states)
        bdat = split(self._beta_map, self.lenb(), max_states)
        return [(ar, br, am, bm) for (ar, am), (br, bm) in product(adat, bdat)]

    def _map_to_deexc_alpha_icol(self):
        """Internal function for generating mapping for column-wise application
        of one-body hamiltonian.
        """
        norb = self.norb()
        nele = self.nalpha()
        length = int(binom(norb - 1, nele - 1))
        length2 = int(binom(norb - 1, nele))

        exc = numpy.zeros((norb, length2, nele, 3), dtype=numpy.int32)
        diag = numpy.zeros((
            norb,
            length,
        ), dtype=numpy.int32)
        index = numpy.zeros((
            norb,
            length2,
        ), dtype=numpy.int32)
        astrings = self.string_alpha_all()
        if fqe.settings.use_accelerated_code:
            _c_map_to_deexc_alpha_icol(exc, diag, index, astrings, norb,
                                       self._alpha_map)
        else:
            alpha = numpy.ones((norb, self.lena()), dtype=int) * -1
            count = numpy.zeros(norb, dtype=int)
            for i, astring in enumerate(astrings):
                # Loop unoccupied orbitals
                for icol in set(range(norb)).difference(integer_index(astring)):
                    alpha[icol, i] = count[icol]
                    index[icol, count[icol]] = i
                    count[icol] += 1

            assert numpy.all(numpy.equal(count, length2))
            icounter = numpy.zeros(norb, dtype=int)

            counter = numpy.zeros((
                norb,
                length2,
            ), dtype=int)
            for (i, j), values in self._alpha_map.items():
                icol = j
                if i != j:
                    for source, target, parity in values:
                        pos = alpha[icol, target]
                        assert pos != -1
                        exc[icol, pos, counter[icol, pos]] = source, i, parity
                        counter[icol, pos] += 1
                else:
                    for source, target, parity in values:
                        assert source == target
                        assert parity == 1
                        diag[icol, icounter[icol]] = target
                        icounter[icol] += 1
            assert numpy.all(numpy.equal(counter, exc.shape[2]))

        return index, exc, diag

    def make_mapping_each(self, result: 'Nparray', alpha: bool, dag: List[int],
                          undag: List[int]) -> int:
        """Generates the mapping for an the alpha or beta part of an individual
        operator onto the given FciGraph. The operator should be particle
        number conserving.

        Args:
            result ('Nparray'): The filled in mapping
                [origin index, target determinant, parity]

            alpha (bool): True or false if the alpha or beta part of the
                individual operator is being treated, respectively.

            dag (List[int]): List of orbitals where the creation operators are
                applied.

            undag (List[int]): List of orbitals where the annihilation
                operators are applied.

        Returns:
            (int): The total number of mappings filled in `result`.

        """
        if alpha:
            strings = self.string_alpha_all()
            length = self.lena()
        else:
            strings = self.string_beta_all()
            length = self.lenb()

        if fqe.settings.use_accelerated_code:
            count = _make_mapping_each(result, strings, length,
                                       numpy.array(dag, dtype=numpy.int32),
                                       numpy.array(undag, dtype=numpy.int32))
        else:
            dag_mask = 0
            for i in dag:
                if i not in undag:
                    dag_mask = set_bit(dag_mask, i)
            undag_mask = 0
            for i in undag:
                undag_mask = set_bit(undag_mask, i)

            count = 0
            for index in range(length):
                current = int(strings[index])

                check = (current & dag_mask) == 0 and \
                    (current & undag_mask ^ undag_mask) == 0
                if check:
                    parity = 0
                    for i in reversed(undag):
                        parity += count_bits_above(current, i)
                        current = unset_bit(current, i)
                    for i in reversed(dag):
                        parity += count_bits_above(current, i)
                        current = set_bit(current, i)
                    result[count, :] = index, current, parity % 2
                    count += 1
        return count
