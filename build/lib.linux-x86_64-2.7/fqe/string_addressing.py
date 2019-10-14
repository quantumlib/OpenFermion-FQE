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

"""This module implements the basic operations for determinant strings
"""

from scipy import special

from fqe.bitstring import gbit_index


def build_string_address(nele, norb, occupation):
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
    nocc = len(occupation)
    if nele != nocc:
        raise ValueError("The number of electrons is not equal to occupation")
    det_addr = 0

    def _addressing_array_element(norb, nele, el_i, or_i):
        """Calculate an addressing array element zar( el_i, or_i)

        Args:
            norb : Number of orbitals
            nele : number of electrons in the state
            el_i : index of the current electron
            or_i : index of the current orbital

        Returns:
            int: weight associated with step on string graph
        """
        if el_i == nele:
            zar = or_i - nele
        else:
            zar = 0

            def _addressing_array_summand(oc_i, nmk):
                """Calculate a summand in the addressing array
                """
                assert nmk > 0, '-1 meaningless in binomial address'
                return special.binom(oc_i, nmk) - special.binom(oc_i-1, nmk-1)

            for i in range(norb-or_i+1, norb-el_i+1):
                zar += _addressing_array_summand(i, nele-el_i)

        return int(zar)

    for i in range(1, nele+1):
        det_addr += _addressing_array_element(norb, nele, i, occupation[i-1]+1)

    return int(det_addr)


def count_bits(string, bitval='1'):
    """Count the bit value in a bistring

    Args:
        string (bitstring) - a bitstring to count the bits of
        bitval (string) - include the option to count unset bits

    Returns:
        int - the number of bits equal to bitval
    """
    return bin(string).count(bitval)


def generate_excitations(det, norb):
    """Given an occupation representation return all single excitations out of
       that bitstring. -> sum_{kl} E_{kl}|psi>

    Args:
        det (int) - the birstring to build excitations out of
        norb (int) - the highest molecular orbital to excite to

    Returns:
        list[(k, l, bitstrings)]

    e.g. |0001011>, ->

         |0010011>, |0100011>, |1000011>
         |0001101>, |0011001>, |0101001>
         |1001001>, ...
    """
    out = []
    _gbi = gbit_index(det)
    for occ in _gbi:
        _wbits = det
        _wbits = _wbits ^ 2**occ
        for j in range(occ+1, norb):
            if not _wbits & 2**j:
                out.append((j, occ, _wbits | 2**j))
    return out


def string_parity(string1, string0):
    """Given two strings of equal particle number, determine the parity of
    going the process string1 = (p^q) string0

    Args:
        string1 (bitsring) - a bitstring representation of the occupation
        string2 (bitsring) - a bitstring representation of another occupation

    Returns:
        (int) - the parity associated with changing from one string to another
    """
    dif = string1 - string0
    if dif > 0:
        return (-1) ** (count_bits(string1 & dif))
    if dif == 0:
        return 1
    return (-1) ** (count_bits(string0 & (-dif)))
