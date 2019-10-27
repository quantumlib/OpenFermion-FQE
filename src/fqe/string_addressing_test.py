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

"""string addressing unit tests
"""

import unittest
from scipy import special

from fqe import string_addressing

class StringAddressingTests(unittest.TestCase):
    """Unit tests
    """


    def test_build_string_address_error(self):
        """The ground state should be zero in the addressing system.
        """
        nele = 4
        norb = 8
        occ = list(range(8))
        self.assertRaises(ValueError, string_addressing.build_string_address,
                          nele, norb, occ)


    def test_build_string_address_min(self):
        """The ground state should be zero in the addressing system.
        """
        nele = 4
        norb = 8
        occ = list(range(4))
        self.assertEqual(string_addressing.build_string_address(nele, norb,
                                                                occ), 0)


    def test_build_string_address_max(self):
        """The highest excited state should be the binomial coefficient for
        putting m objects into n bins indexed from 0.
        """
        nele = 4
        norb = 8
        occ = list(range(4, 8))
        test = int(special.binom(norb, nele) - 1)
        self.assertEqual(string_addressing.build_string_address(nele, norb,
                                                                occ), test)


    def test_build_string_address_list(self):
        """The highest excited state should be the binomial coefficient for
        putting m objects into n bins indexed from 0.
        """
        nele = 4
        norb = 6
        occ_list = [
            [0, 1, 2, 3],
            [0, 1, 2, 4],
            [0, 1, 2, 5],
            [0, 1, 3, 4],
            [0, 1, 3, 5],
            [0, 1, 4, 5],
            [0, 2, 3, 4],
            [0, 2, 3, 5],
            [0, 2, 4, 5],
            [0, 3, 4, 5],
            [1, 2, 3, 4],
            [1, 2, 3, 5],
            [1, 2, 4, 5],
            [1, 3, 4, 5],
            [2, 3, 4, 5]
            ]
        test = list(range(15))
        addr = []
        for i in occ_list:
            addr.append(string_addressing.build_string_address(nele, norb, i))

        self.assertListEqual(test, addr)


    def test_count_bits_zero(self):
        """The bit vacuum
        """
        self.assertEqual(string_addressing.count_bits(0), 0)


    def test_count_bits_min(self):
        """The bit is the answer and the answer is the bit.
        """
        self.assertEqual(string_addressing.count_bits(1), 1)


    def test_count_bits_full(self):
        """Count a bunch of bits.
        """
        test = (1<<8) - 1
        self.assertEqual(string_addressing.count_bits(test), 8)


    def test_generate_excitations(self):
        """Generate and return pertinent information regarding excitations out
        of a determinant
        """
        string = 1 + 2 + 8
        test = [
            (2, 0, 4 + 2 + 8),
            (4, 0, 16 + 2 + 8),
            (5, 0, 32 + 2 + 8),
            (2, 1, 1 + 4 + 8),
            (4, 1, 1 + 16 + 8),
            (5, 1, 1 + 32 + 8),
            (4, 3, 1 + 2 + 16),
            (5, 3, 1 + 2 + 32)
            ]
        out = string_addressing.generate_excitations(string, 6)
        self.assertListEqual(out, test)


    def test_string_parity_highest_bit(self):
        """There is no parity change by only increasing the highest bit
        """
        self.assertEqual(string_addressing.string_parity(256 + 7, 15), 1)


    def test_string_parity_excited_odd_bit(self):
        """There is a parity change if we excite -1**n below for n odd
        """
        self.assertEqual(string_addressing.string_parity(256 + 11, 15), -1)


    def test_string_parity_excited_even_bit(self):
        """There is no parity change if we excite -1**n below for n even
        """
        self.assertEqual(string_addressing.string_parity(256 + 13, 15), 1)


    def test_string_parity_unexcited(self):
        """If there is no bit change there is no parity change
        """
        self.assertEqual(string_addressing.string_parity(15, 15), 1)


    def test_string_parity_deexcited(self):
        """If there is no bit change there is no parity change
        """
        self.assertEqual(string_addressing.string_parity(15, 256 + 13), 1)
