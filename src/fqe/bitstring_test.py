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

import unittest

from fqe import bitstring


class BitstringTest(unittest.TestCase):
    """Unit tests
    """

    def test_check_conserved_bits_none(self):
        """Make sure that no bits are still conserved bits
        """
        self.assertTrue(bitstring.check_conserved_bits(0, 0))

    def test_check_conserved_bits_true_case(self):
        """Check for a positive conservation result
        """
        conserved = 4
        string0 = 1 + 4 + 8
        self.assertTrue(bitstring.check_conserved_bits(string0, conserved))

    def test_check_conserved_bits_false_case(self):
        """Check for a negative conservation result
        """
        conserved = 2
        string0 = 1 + 4 + 8
        self.assertFalse(bitstring.check_conserved_bits(string0, conserved))

    def test_bit_integer_index_val(self):
        """The index of bits should start at 0.
        """
        _gbiti = bitstring.gbit_index(1)
        self.assertEqual(next(_gbiti), 0)
        _gbiti = bitstring.gbit_index(11)
        self.assertEqual(next(_gbiti), 0)
        self.assertEqual(next(_gbiti), 1)
        self.assertEqual(next(_gbiti), 3)

    def test_bit_integer_index_list(self):
        """Make sure sequential integers are returned for a full bitstring
        """
        test_list = list(range(8))
        start = (1 << 8) - 1
        biti_list = bitstring.integer_index(start)
        self.assertListEqual(biti_list, test_list)

    def test_lexicographic_bitstring_generator_init(self):
        """Check that the first element returned is the initial case.
        """
        _gbitl = bitstring.lexicographic_bitstring_generator(15, 1)
        self.assertListEqual(_gbitl, [15])

    def test_lexicographic_bitstring_generator_list(self):
        """lexicographic bitstrings for a single bit should be the set of binary
        numbers.
        """
        test_list = [2**i for i in range(10)]
        _gbitl = bitstring.lexicographic_bitstring_generator(1, 10)
        self.assertListEqual(_gbitl, test_list)

    def test_lexicographic_bitstring_generator_order(self):
        """Here is a use case of the lexicographic bitstring routine.
        """
        test_list = [3, 5, 6, 9, 10, 12, 17, 18, 20, 24, 33, 34, 36, 40, 48]
        _gbitl = bitstring.lexicographic_bitstring_generator(3, 6)
        self.assertListEqual(_gbitl, test_list)

    def test_count_bits(self):
        """Return the number of set bits in the bitstring
        """
        self.assertEqual(bitstring.count_bits(0), 0)
        self.assertEqual(bitstring.count_bits(1 + 2 + 4 + 8 + 32), 5)

    def test_basic_bit_function(self):
        """Return the number of set bits in the bitstring
        """
        workbit = (1 << 8) - 1
        self.assertEqual(bitstring.get_bit(workbit, 3), 8)
        self.assertEqual(bitstring.set_bit(workbit, 11), workbit + 2**11)
        self.assertEqual(bitstring.unset_bit(workbit, 2), workbit - 2**2)
        self.assertEqual(bitstring.count_bits_above(workbit, 1), 6)
        self.assertEqual(bitstring.count_bits_below(workbit - 2, 8), 7)
        self.assertEqual(bitstring.count_bits_between(workbit - 2, 1, 5), 3)
        self.assertEqual(bitstring.show_bits(1 + 2, nbits=4), '0011')
        self.assertEqual(bitstring.show_bits(1 + 2 + 16), '0000000000010011')
