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
"""Defines the FciGraph class which stores the addressing scheme for FQE data
structures.
"""

# pylint: disable=too-many-instance-attributes

from typing import Dict, List, Tuple

from scipy.special import binom

from fqe.bitstring import integer_index, lexicographic_bitstring_generator
from fqe.bitstring import count_bits_between, get_bit, set_bit, unset_bit
from fqe.util import init_bitstring_groundstate

Spinmap = Dict[Tuple[int, ...], List[Tuple[int, int, int]]]


class FciGraph:
    """FciGraph contains the addressing system for the wavefunction. Each
    determinant is considered as a product of alpha creation operators and
    beta operators acting on the vacuum in the manner {alpha ops}{beta ops}|>.
    To find any determinant in the model one needs the occupation index of the
    alpha orbitals and the beta orbitals. From this information, any pointer
    into the wavefunction can be generated.
    """

    def __init__(self, nalpha: int, nbeta: int, norb: int) -> None:
        """Initializes an FciGraph.

        Args:
            nalpha: The number of alpha electrons.
            nbeta: The number of beta electrons.
            norb: The number of spatial orbitals such that the total number
            of orbitals is ntot = 2 * norb.
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
        self._alpha_map: Spinmap = self._build_mapping(self._astr, self._aind)
        self._beta_map: Spinmap = self._build_mapping(self._bstr, self._bind)

        self._fci_map: Dict[Tuple[int, ...], Tuple[Spinmap, Spinmap]] = {}

    def insert_mapping(
        self, dna: int, dnb: int, mapping_pair: Tuple[Spinmap, Spinmap]
    ) -> None:
        """Inserts a new pair of alpha and beta mappings with key that is the
        tuple of the differences for the number of alpha and beta electrons.

        Args:
            dna: Difference in the number of alpha electrons.
            dnb: Difference in the number of beta electrons.
            mapping_pair: Mapping for alpha and beta electrons.
        """
        self._fci_map[(dna, dnb)] = mapping_pair

    def find_mapping(self, dna: int, dnb: int) -> Tuple[Spinmap, Spinmap]:
        """Returns the pair of mappings that corresponds to dna and dnb
        (difference in the number of electrons for alpha and beta).

        Args:
            dna: Difference in the number of alpha electrons.
            dnb: Difference in the number of beta electrons.

        Returns:
            Mapping for alpha and beta electrons.
        """
        # TODO: Raise an error instead of assert clause.
        assert (dna, dnb) in self._fci_map
        return self._fci_map[(dna, dnb)]

    def _build_mapping(
        self, strings: List[int], index: Dict[int, int]
    ) -> Spinmap:
        """Constructs the mapping of alpha string and beta string excitations
        for :math:`a^\\dagger_i a_j` from the bitstrings contained in the
        fci_graph.

        Args:
            strings: List of the determinant bitstrings.
            index: List of the indices corresponding to the determinant
                bitstrings
        """
        out: Dict[Tuple[int, ...], List[Tuple[int, int, int]]] = {}
        norb = self._norb
        for iorb in range(norb):  # excitation
            for jorb in range(norb):  # de-excitation
                value = []
                for string in strings:
                    if get_bit(string, jorb) and not get_bit(string, iorb):
                        parity = count_bits_between(string, iorb, jorb)
                        value.append(
                            (
                                index[string],
                                index[unset_bit(set_bit(string, iorb), jorb)],
                                int((-1) ** parity),
                            )
                        )
                    elif iorb == jorb and get_bit(string, iorb):
                        value.append((index[string], index[string], 1))
                out[(iorb, jorb)] = value

        return out

    def alpha_map(self, iorb: int, jorb: int) -> List[Tuple[int, int, int]]:
        """Returns the Knowles-Handy mapping (within this FciGraph) for alpha
        electrons for :math:`a^\\dagger_i a_j`.

        Args:
            iorb: Orbital index for the creation operator.
            jorb: Orbital index for the annihilation operator.

        Returns:
            Array of string mapping with phases.
        """
        # TODO: Raise error instead of assert clause.
        assert (iorb, jorb) in self._alpha_map.keys()
        return self._alpha_map[(iorb, jorb)]

    def beta_map(self, iorb: int, jorb: int) -> List[Tuple[int, int, int]]:
        """Returns the Knowles-Handy mapping (within this FciGraph) for beta
        electrons for :math:`a^\\dagger_i a_j`.

        Args:
            iorb: Orbital index for the creation operator.
            jorb: Orbital index for the annihilation operator.

        Returns:
            Array of string mapping with phases.
        """
        # TODO: Raise error instead of assert clause.
        assert (iorb, jorb) in self._beta_map.keys()
        return self._beta_map[(iorb, jorb)]

    def lena(self) -> int:
        """Returns the number of alpha electrons."""
        return self._lena

    def lenb(self) -> int:
        """Returns the number of beta electrons."""
        return self._lenb

    def nalpha(self) -> int:
        """Returns the number of alpha electrons."""
        return self._nalpha

    def nbeta(self) -> int:
        """Returns the number of beta electrons."""
        return self._nbeta

    def norb(self) -> int:
        """Returns the number of beta electrons."""
        return self._norb

    def _build_strings(
        self, nele: int, length: int
    ) -> Tuple[List[int], Dict[int, int]]:
        """Builds all bitstrings for index the FCI and their lexicographic
        index for a single spin case.

        Args:
            nele: Number of electrons in this graph.
            length: The largest dimension of the graph.

        Returns:
            Initialized string array for accessing configurations in the FCI.
        """
        grs = init_bitstring_groundstate(nele)
        blist = lexicographic_bitstring_generator(grs, self._norb)
        string_list = [
            0 for _ in range(length)
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
        """Retrieves the alpha bitstring representation stored at the address.

        Args:
            address: An integer pointing into the FciGraph.

        Returns:
            An occupation representation of the configuration
        """
        return self._astr[address]

    def string_beta(self, address: int) -> int:
        """Retrieves the beta bitstring representation stored at the address.

        Args:
            address: An integer pointing into the FciGraph.

        Returns:
            An occupation representation of the configuration
        """
        return self._bstr[address]

    def string_alpha_all(self) -> List[int]:
        """Returns all bitstrings for alpha occupied orbitals."""
        return self._astr

    def string_beta_all(self) -> List[int]:
        """Returns all bitstrings for beta occupied orbitals."""
        return self._bstr

    def index_alpha(self, bit_string: int) -> int:
        """Retrieves the alpha index stored by its bitstring.

        Args:
            bit_string: An occupation representation of the configuration.

        Returns:
            An integer pointing into the FciGraph.
        """
        return self._aind[bit_string]

    def index_beta(self, bit_string: int) -> int:
        """Retrieve the beta bitstring representation stored at the address.

        Args:
            bit_string: An occupation representation of the configuration.

        Returns:
            An integer pointing into the FciGraph.
        """
        return self._bind[bit_string]

    def index_alpha_all(self) -> Dict[int, int]:
        """Returns the index and the corresponding occupation string for all
        alpha strings.
        """
        return self._aind

    def index_beta_all(self) -> Dict[int, int]:
        """Returns the index and the corresponding occupation string for all
        beta strings.
        """
        return self._bind

    @staticmethod
    def _addressing_array_element(
            norb: int, nele: int, el_i: int, or_i: int
    ) -> int:
        """Calculate an addressing array element zar( el_i, or_i)

        Args:
            norb: Number of orbitals.
            nele: Number of electrons in the state.
            el_i: Index of the current electron.
            or_i: Index of the current orbital.

        Returns:
            Weight associated with step on string graph.
        """
        if el_i == nele:
            zar = or_i - nele
        else:
            zar = 0

            def _addressing_array_summand(oc_i: int, nmk: int) -> int:
                """Calculates a summand in the addressing array."""
                assert nmk > 0, "-1 meaningless in binomial address"
                return binom(oc_i, nmk) - binom(oc_i - 1, nmk - 1)

            for i in range(norb - or_i + 1, norb - el_i + 1):
                zar += _addressing_array_summand(i, nele - el_i)

        return int(zar)

    @staticmethod
    def _build_string_address(
        nele: int, norb: int, occupation: List[int]
    ) -> int:
        """Given a list of occupied orbitals in ascending order, generates the
        index into the CI matrix.

        Args:
            nele: Number of electrons for a single spin case.
            norb: Number of spatial orbitals.
            occupation: List with integers indicating the index of the occupied
                orbitals starting from 0.

        Returns:
            A pointer into a spin a block of the CI addressing system.
        """
        det_addr = 0

        for i in range(1, nele + 1):
            det_addr += FciGraph._addressing_array_element(
                norb, nele, i, occupation[i - 1] + 1
            )

        return int(det_addr)
