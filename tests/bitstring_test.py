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
"""Test cases for bitstring.py
"""

import numpy
import pytest

from fqe import bitstring
import fqe.settings


def test_bit_integer_index_val():
    """The index of bits should start at 0.
    """
    _gbiti = bitstring.gbit_index(1)
    assert next(_gbiti) == 0
    _gbiti = bitstring.gbit_index(11)
    assert next(_gbiti) == 0
    assert next(_gbiti) == 1
    assert next(_gbiti) == 3


def test_bit_integer_index_list(c_or_python):
    """Make sure sequential integers are returned for a full bitstring
    """
    fqe.settings.use_accelerated_code = c_or_python
    test_list = list(range(8))
    start = (1 << 8) - 1
    biti_list = bitstring.integer_index(start)
    assert (biti_list == test_list).all()


def test_lexicographic_bitstring_generator_init(c_or_python):
    """Check that the returned array is empty when arguments are wrong
    """
    fqe.settings.use_accelerated_code = c_or_python
    with pytest.raises(ValueError):
        bitstring.lexicographic_bitstring_generator(4, 1)


def test_lexicographic_bitstring_generator_list(c_or_python):
    """lexicographic bitstrings for a single bit should be the set of binary
    numbers.
    """
    fqe.settings.use_accelerated_code = c_or_python
    test_list = numpy.array([2**i for i in range(10)], dtype=numpy.int32)
    _gbitl = bitstring.lexicographic_bitstring_generator(1, 10)
    assert numpy.array_equal(_gbitl, test_list)


def test_lexicographic_bitstring_generator_order(c_or_python):
    """Here is a use case of the lexicographic bitstring routine.
    """
    fqe.settings.use_accelerated_code = c_or_python
    test_data = [3, 5, 6, 9, 10, 12, 17, 18, 20, 24, 33, 34, 36, 40, 48]
    test_list = numpy.array(test_data, dtype=numpy.int32)
    _gbitl = bitstring.lexicographic_bitstring_generator(2, 6)
    assert numpy.array_equal(_gbitl, test_list)


def test_count_bits(c_or_python):
    """Return the number of set bits in the bitstring
    """
    fqe.settings.use_accelerated_code = c_or_python
    assert bitstring.count_bits(0) == 0
    assert bitstring.count_bits(1 + 2 + 4 + 8 + 32) == 5


def test_basic_bit_function():
    """Return the number of set bits in the bitstring
    """
    workbit = (1 << 8) - 1
    assert bitstring.get_bit(workbit, 3) == 8
    assert bitstring.set_bit(workbit, 11) == workbit + 2**11
    assert bitstring.unset_bit(workbit, 2) == workbit - 2**2
    assert bitstring.count_bits_above(workbit, 1) == 6
    assert bitstring.count_bits_below(workbit - 2, 8) == 7
    assert bitstring.count_bits_between(workbit - 2, 1, 5) == 3
    assert bitstring.show_bits(1 + 2, nbits=4) == '0011'
    assert bitstring.show_bits(1 + 2 + 16) == '0000000000010011'
