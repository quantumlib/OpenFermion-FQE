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
"""Bitstring manipulation routines are wrapped up here to give context to
frequently used operations.
"""

from typing import Generator, List

from itertools import combinations


def check_conserved_bits(str0: int, conserved: int) -> bool:
    """Check that str0 has bits set in the same place that conserved has bits
    set.

    Args:
        str0: A bitstring representing an occupation state of a configuration.
        conserved: A bitstring with bits set that should also be set in str0.

    Returns:
        True if all the conserved bits are set in str0, else False.
    """
    return (str0 & conserved) == conserved


def gbit_index(str0: int) -> Generator[int, None, None]:
    """Generator for returning integers that associate each bit in sequence
    with a corresponding orbital index.

    Args:
        str0: String to process.

    Returns:
        Integer corresponding to the index of a set bit in str0.

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
         str0: Orbital occupation representation.

    Returns:
        A list of integers indicating the index of each occupied orbital.
    """
    return list(gbit_index(str0))


def reverse_integer_index(occ: List[int]) -> int:
    """Reverse of `integer_index` function. This function generates an
    bitstring that corresponds to the list passed as an argument.

    Args:
        occ: List of occupied orbitals.

    Returns:
        Orbital occupation representation.
    """
    out = 0
    for i in occ:
        out = set_bit(out, i)
    return out


def lexicographic_bitstring_generator(str0: int, norb: int) -> List[int]:
    """Generate all bitstrings with a definite bit count starting from an
    initial state.

    Args:
         str0: Integer representing bitstring ground state.
         norb: Number of spatial orbitals to distribute the particles into.

    Returns:
        A list of bitstrings representing the occupation states.
    """
    out = []
    gs_bs = "{0:b}".format(str0).zfill(norb)
    gs_bs = [int(x) for x in gs_bs]
    n_elec = sum(gs_bs)
    n_orbs = len(gs_bs)
    for ones_positions in combinations(range(n_orbs), n_elec):
        out.append(
            sum([2 ** z for z in ones_positions])
        )  # convert directly to int
    return sorted(out)


def count_bits(string: int, bitval: str = "1") -> int:
    """Returns the number of bits equal to bitval in the input string.

    Args:
        string: A bitstring to count the bits of.
        bitval: Count number of this value in string.

    Returns:
        The number of bits equal to bitval in string.
    """
    return bin(string).count(bitval)


def get_bit(string: int, pos: int) -> int:
    """Returns a bit located at the position.

    Args:
        string: Bitstring.
        pos: Position in the bitstring.
    """
    return string & (2 ** pos)


def set_bit(string: int, pos: int) -> int:
    """Returns bitstring with the bit at the position set.

    Args:
        string: Bitstring.
        pos: Position in the bitstring.
    """
    return string | (2 ** pos)


def unset_bit(string: int, pos: int) -> int:
    """Returns bitstring with the bit at the position unset.

    Args:
        string: Bitstring.
        pos: Position in the bitstring.
    """
    return string & ~(2 ** pos)


def count_bits_above(string: int, pos: int) -> int:
    """Returns the number of set bits higher than the position.

    Args:
        string: Bitstring.
        pos: Position in the bitstring.
    """
    return count_bits(string & ~(2 ** (pos + 1) - 1))


def count_bits_below(string: int, pos: int) -> int:
    """Returns the number of set bits lower than the position.

    Args:
        string: Bitstring.
        pos: Position in the bitstring.
    """
    return count_bits(string & (2 ** pos - 1))


def count_bits_between(string: int, pos1: int, pos2: int) -> int:
    """Counts the number of bits between position1 and position2.

    Args:
        string: Bitstring.
        pos1: One position in the bitstring.
        pos2: Another position in the bitstring.
    """
    mask = (2 ** max(pos1, pos2) - 1) ^ (2 ** (min(pos1, pos2) + 1) - 1)
    return count_bits(string & mask)


def show_bits(string: int, nbits: int = 16) -> str:
    """Returns a string showing the occupations of the bitstring

    Args:
        string: Bitstring.
        nbits: The number of bits to show.
    """
    return str(bin(string)[2:].zfill(nbits))
