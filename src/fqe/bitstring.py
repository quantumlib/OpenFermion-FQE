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
"""Bitsrting manipulation routines are wrapped up here to give context to
frequently used operations.
"""

from typing import Generator, List

from itertools import combinations


def check_conserved_bits(str0: int, conserved: int) -> bool:
    """Check that str0 has bits set in the same place that conserved has bits
    set.

    Args:
        str0 (bitstring) - a bitstring representing an occupation state of a \
            configuration

        conserved (bitstring) - a bitstring with bits set that should also be \
            set in str0

    Returns:
        (bool) - if all the conserved bits are set in str0 then return True
    """
    return (str0 & conserved) == conserved


def gbit_index(str0: int) -> Generator[int, None, None]:
    """Generator for returning integers that associate each bit in sequence with
    a corresponding orbital index

    Args:
        str0 (bitstring) - string to process

    Returns:
        (int) - integer corresponding to the index of a set bit in str0

        .. code-block:: text

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


def integer_index(str0: int) -> List[int]:
    """Generate integers indicating the position of occupied orbitals in a
    bitstring starting from 0.  This is a convience wrapper for the gbit_index
    generator

    Args:
         str0 (bistring) - orbital occupation representation

    Returns:
        list[int] - a list of integers indicating the index of each occupied \
            orbital
    """
    return list(gbit_index(str0))


def reverse_integer_index(occ: List[int]) -> int:
    """Reverse of the integer index function above. This function generates an
    bitstring that correspoinds to the list passed as an argument.

    Args:
        occ (List[int]) - list of occupied orbitals

    Returns:
        int bitstring - orbital occupation representation
    """
    out = 0
    for i in occ:
        out = set_bit(out, i)
    return out


def lexicographic_bitstring_generator(str0: int, norb: int) -> List[int]:
    """
    Generate all bitstrings with a definite bit count starting from an initial
    state

    Args:
         str0 (bitstring) - integer representing bitstring ground state

         norb (int) - number of spatial orbitals to distribute the \
         particles into

    Returns:
        list[bitstrings] - a list of bitstrings representing the occupation \
            states
    """
    out = []
    gs_bs = [int(x) for x in '{0:b}'.format(str0).zfill(norb)]
    n_elec = sum(gs_bs)
    n_orbs = len(gs_bs)
    for ones_positions in combinations(range(n_orbs), n_elec):
        out.append(sum([2**z for z in ones_positions
                       ]))  # convert directly to int
    return sorted(out)


def count_bits(string: int, bitval: str = '1') -> int:
    """Count the bit value in a bistring

    Args:
        string (bitstring) - a bitstring to count the bits of

        bitval (string) - include the option to count unset bits

    Returns:
        int - the number of bits equal to bitval
    """
    return bin(string).count(bitval)


def get_bit(string: int, pos: int) -> int:
    """Return a bit located at the position

    Args:
        string (int) - bit string

        pos (int) - position in the bit string
    """
    return string & (2**pos)


def set_bit(string: int, pos: int) -> int:
    """Return bitstring with the bit at the position set

    Args:
        string (int) - bit string

        pos (int) - position in the bit string
    """
    return string | (2**pos)


def unset_bit(string: int, pos: int) -> int:
    """Return bitstring with the bit at the position unset

    Args:
        string (int) - bit string

        pos (int) - position in the bit string
    """
    return string & ~(2**pos)


def count_bits_above(string: int, pos: int) -> int:
    """Return the number of set bits higher than the position

    Args:
        string (int) - bit string

        pos (int) - position in the bit string
    """
    return count_bits(string & ~(2**(pos + 1) - 1))


def count_bits_below(string: int, pos: int) -> int:
    """Return the number of set bits lower than the position

    Args:
        string (int) - bit string

        pos (int) - position in the bit string
    """
    return count_bits(string & (2**pos - 1))


def count_bits_between(string: int, pos1: int, pos2: int) -> int:
    """Count the number of bits between position1 and position2

    Args:
        string (int) - bit string

        pos1 (int) - one of the positions in the bit string

        pos2 (int) - the other position in the bit string
    """
    mask = (2**max(pos1, pos2) - 1) ^ (2**(min(pos1, pos2) + 1) - 1)
    return count_bits(string & mask)


def show_bits(string: int, nbits: int = 16) -> str:
    """Return a string showing the occupations of the bitstring

    Args:
        string (int) - bit string

        nbits (int) - the number of bits to show
    """
    return str(bin(string)[2:].zfill(nbits))
