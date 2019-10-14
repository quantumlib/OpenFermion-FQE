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

"""Bitsrting manipulation routines are wrapped up here to give context to 
frequently used operations.
"""

from itertools import permutations


def check_conserved_bits(str0, conserved):
    """Are all the bits set in conserved also set in str0

    Args:
        str0 (bitstring) - a bitstring representing an occupation state of a
            onfiguration
        conserved (bitstring) - a bitstring with bits set that should also be
            set in str0

    Returns:
        (bool) - if all the conserved bits are set in str0 then return True
    """
    return (str0 & conserved) == conserved


def gbit_index(str0):
    """Generator for returning integers that associate each bit in sequence with
    a corresponding orbital index

    Args:
        str0 (bitstring) - string to process

    Returns:
        (int) - integer corresponding to the index of a set bit in str0
            starting at 0

            bitstring ->  100101
            index ->      5  2 0
            output ->   0
                        2
                        5
    """
    work_bits = str0
    bit_index = 0
    while work_bits:
        if work_bits & 1:
            yield bit_index
        work_bits = work_bits >> 1
        bit_index += 1


def integer_index(str0):
    """Generate integers indicating the position of occupied orbitals in a
    bitstring starting from 0.  This is a convience wrapper for the gbit_index
    generator

    Args:
         str0 (bistring) - orbital occupation representation

    Returns:
        list[int] - a list of integers indicating the index of each occupied
            orbital
    """
    return list(gbit_index(str0))


def lexicographic_bitstring_generator(str0, norb):
    """
    Generate all bitstrings with a definite bit count starting from an initial
    state

    Args:
         str0 (bitstring) - integer representing bitstring ground state
         norb (bitstring) - number of spatial orbitals to distribute the 
         particles into

    Returns:
        list[bitstrings] - a list of bitstrings representing the occupation
            states
    """
    out = []
    gs_bs = '{0:b}'.format(str0).zfill(norb)
    bits_set = set(permutations(gs_bs))
    for string in bits_set:
        out.append(int(''.join(string), 2))

    return sorted(out)
