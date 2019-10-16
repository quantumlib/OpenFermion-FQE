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

"""FciGraph hold the strings and lexical ordering for a set of strings
"""

from scipy.special import binom

from fqe.bitstring import integer_index, lexicographic_bitstring_generator
from fqe.string_addressing import build_string_address
from fqe.util import init_bitstring_groundstate, validate_config

class FciGraph():
    """ FciGraph contains the addressing system for the wavefunction.  Each
    determinant is considered as a product of alpha creation operators and
    beta operators acting on the vacuum in the manner {alpha ops}{beta ops}|>.
    To find any determinant in the model one needs the occupation index of the
    alpha orbitals and the beta orbitals.  From this information, any pointer
    into the wavefunction can be generated.

    This is an internal class that should not be exposed to the user
    """

    def __init__(self, nalpha: int, nbeta: int, norb: int) -> None:
        """

        Args:
            nalpha (int) - The number of alpha electrons
            nbeta (int) - The number of beta electrons
            norb (int) - The number of spatial orbitals such that the total number
                of orbitals is ntot = 2*norb.
        """

        validate_config(nalpha, nbeta, norb)

        self._norb = norb
        self._nalpha = nalpha
        self._nbeta = nbeta
        self._lena = int(binom(norb, nalpha))
        self._lenb = int(binom(norb, nbeta))
        self._astr = [0 for _ in range(self._lena)]
        self._bstr = [0 for _ in range(self._lenb)]
        self._build_fci_strings()


    def _build_strings(self, nele: int, length: int, string_list: List[int]) -> None:
        """Build all bitstrings for index the FCI and their lexicographic index
           for a single spin case.

        Args:
            nele (int) - number of electrons in this graph
            length (int) - the largest dimension of the graph
            string_list list[bitstring] - an array holding the bitstrings
                that we want to access

        Returns:
            An initialized string array for accessing configurations in the FCI
        """
        grs = init_bitstring_groundstate(nele)
        blist = lexicographic_bitstring_generator(grs, self._norb)
        for i in range(length):
            wbit = blist[i]
            occ = integer_index(wbit)
            string_list[build_string_address(nele, self._norb, occ)] = wbit


    def _build_fci_strings(self) -> None:
        """Build the Fcigraph for each spin case in the configuration.  This is
        just a convenience wrapper to accomplish initialization of each spin
        case.

        Args:
            None

        Returns:
            None
        """
        self._build_strings(self._nalpha, self._lena, self._astr)
        self._build_strings(self._nbeta, self._lenb, self._bstr)


    def get_alpha(self, address: int) -> int:
        """Retrieve the alpha bitstring reprsentation stored at the address

        Args:
            address (int) - an integer pointing into the fcigraph

        Returns:
            (bistring) - an occupation representation of the configuration
        """
        return self._astr[address]


    def get_beta(self, address: int) -> int:
        """Retrieve the beta bitstring reprsentation stored at the address

        Args:
            address (int) - an integer pointing into the fcigraph

        Returns:
            (bistring) - an occupation representation of the configuration
        """
        return self._bstr[address]
