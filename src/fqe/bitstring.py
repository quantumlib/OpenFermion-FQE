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

from itertools import combinations
from typing import Generator, List

import numpy
from numpy import ndarray as Nparray
from scipy import special

from fqe.lib.bitstring import _lexicographic_bitstring_generator, _count_bits, \
                              _get_occupation
import fqe.settings


def gbit_index(str0: int) -> Generator[int, None, None]:
    """Generator for returning integers that associate each bit in sequence with
    a corresponding orbital index

    Args:
        str0 (bitstring): string to process

    Returns:
        (int): integer corresponding to the index of a set bit in str0

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


def integer_index(string: int) -> 'Nparray':
    """Generate integers indicating the position of occupied orbitals in a
    bitstring starting from 0.  This is a convience wrapper for the gbit_index
    generator

    Args:
        string (int): orbital occupation representation

    Returns:
        Nparray: a list of integers indicating the index of each occupied \
            orbital
    """
    if fqe.settings.use_accelerated_code:
        return _get_occupation(string)
    else:
        return numpy.array(list(gbit_index(int(string)))).astype(numpy.int32)


def reverse_integer_index(occ: List[int]) -> int:
    """Reverse of the integer index function above. This function generates an
    bitstring that correspoinds to the list passed as an argument.

    Args:
        occ (List[int]): list of occupied orbitals

    Returns:
        int: orbital occupation representation
    """
    out = 0
    for i in occ:
        out = set_bit(out, i)
    return out


def lexicographic_bitstring_generator(nele: int, norb: int) -> 'Nparray':
    """
    Generate all bitstrings with a definite bit count starting from an initial
    state

    Args:
        nele (int): number of electrons

        norb (int): number of spatial orbitals

    Returns:
        Nparray: a list of bitstrings representing the occupation \
            states
    """
    if nele > norb:
        raise ValueError("nele cannot be larger than norb")

    if fqe.settings.use_accelerated_code:
        out = numpy.zeros((int(special.comb(norb, nele)),), dtype=numpy.uint64)
        _lexicographic_bitstring_generator(out, norb, nele)
        return out
    else:
        out = []
        for comb in combinations(range(norb), nele):
            out.append(reverse_integer_index(list(comb)))
        return numpy.array(sorted(out), dtype=numpy.uint64)


def count_bits(string: int) -> int:
    """Count the bit value in a bistring

    Args:
        string (int): a bitstring to count the bits of

    Returns:
        int: the number of bits equal to 1
    """
    if fqe.settings.use_accelerated_code:
        return _count_bits(string)
    else:
        return bin(int(string)).count('1')


def get_bit(string: int, pos: int) -> int:
    """Return a bit located at the position

    Args:
        string (int): bit string

        pos (int): position in the bit string

    Returns:
        int: 0 if the bit is 0, 2**pos if the bit is 1
    """
    return int(string) & (1 << pos)


def set_bit(string: int, pos: int) -> int:
    """Return bitstring with the bit at the position set

    Args:
        string (int): bit string

        pos (int): position in the bit string

    Returns:
        int: string with the pos bit set to 1
    """
    return int(string) | (1 << pos)


def unset_bit(string: int, pos: int) -> int:
    """Return bitstring with the bit at the position unset

    Args:
        string (int): bit string

        pos (int): position in the bit string

    Returns:
        int: string with the pos bit set to 0
    """
    return int(string) & ~(1 << pos)


def count_bits_above(string: int, pos: int) -> int:
    """Return the number of set bits higher than the position

    Args:
        string (int): bit string

        pos (int): position in the bit string

    Returns:
        int: the number of 1 bits above pos
    """
    return count_bits(int(string) & ~((1 << (pos + 1)) - 1))


def count_bits_below(string: int, pos: int) -> int:
    """Return the number of set bits lower than the position

    Args:
        string (int): bit string

        pos (int): position in the bit string

    Returns:
        int: the number of 1 bits below pos
    """
    return count_bits(int(string) & ((1 << pos) - 1))


def count_bits_between(string: int, pos1: int, pos2: int) -> int:
    """Count the number of bits between position1 and position2

    Args:
        string (int): bit string

        pos1 (int): one of the positions in the bit string

        pos2 (int): the other position in the bit string

    Returns:
        int: the number of 1 bits between pos1 and pos2
    """
    mask = (((1 << pos1) - 1) ^ ((1 << (pos2 + 1)) - 1)) \
         & (((1 << pos2) - 1) ^ ((1 << (pos1 + 1)) - 1))
    return count_bits(int(string) & mask)


def show_bits(string: int, nbits: int = 16) -> str:
    """Return a string showing the occupations of the bitstring

    Args:
        string (int): bit string

        nbits (int): the number of bits to show

    Returns:
        str: string representation of the bit string
    """
    return str(bin(int(string))[2:].zfill(nbits))
