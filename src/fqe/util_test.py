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
""" util unit tests
"""

import unittest

import numpy

from fqe import util


class UnitTest(unittest.TestCase):
    """unit tests
    """

    def test_alpha_beta_electrons(self):
        """Check to make sure that the correct number of alpha and beta
        electrons are parsed from the number and multiplicity
        """
        self.assertTupleEqual((1, 1), util.alpha_beta_electrons(2, 0))
        self.assertTupleEqual((4, 1), util.alpha_beta_electrons(5, 3))
        self.assertTupleEqual((0, 5), util.alpha_beta_electrons(5, -5))

    def test_alpha_beta_error(self):
        """Check to make sure that the correct number of alpha and beta
        electrons are parsed from the number and multiplicity
        """
        self.assertRaises(ValueError, util.alpha_beta_electrons, -1, 0)
        self.assertRaises(ValueError, util.alpha_beta_electrons, 1, 2)

    def test_bubblesort_order(self):
        """Check that bubble sort works.
        """
        length = 5
        test_list = [(length - 1 - i) for i in range(length)]
        ordered_list = [i for i in range(length)]
        util.bubblesort(test_list)
        self.assertListEqual(ordered_list, test_list)

    def test_bubblesort_permutation_count(self):
        """ Make sure that we are counting the correct number of permutations
        to sort the list
        """
        length = 2
        test_list = [(length - 1 - i) for i in range(length)]
        self.assertEqual(1, util.bubblesort(test_list))
        length = 3
        test_list = [(length - 1 - i) for i in range(length)]
        self.assertEqual(3, util.bubblesort(test_list))
        test_list = [2, 0, 1]
        self.assertEqual(2, util.bubblesort(test_list))

    def test_reverse_bubblesort_permutation_count(self):
        """ Make sure that we are counting the correct number of permutations
        to sort the list
        """
        test_list = [[0, 0], [1, 0]]
        self.assertEqual(1, util.reverse_bubble_list(test_list))
        test_list = [[0, 0], [1, 0], [2, 0]]
        self.assertEqual(3, util.reverse_bubble_list(test_list))
        test_list = [[0, 0], [2, 0], [1, 0]]
        self.assertEqual(2, util.reverse_bubble_list(test_list))

    def test_configuration_key_union_empty(self):
        """The union of no configuration keys should be an empty list
        """
        self.assertListEqual([], util.configuration_key_union())

    def test_configuration_key_union(self):
        """The union of no configuration keys should be an empty list
        """
        configs0 = [(2, 0), (3, 1)]
        configs1 = [(2, 0), (5, 1), (6, -2)]
        testset = set([(2, 0), (3, 1), (5, 1), (6, -2)])
        self.assertSetEqual(
            testset, set(util.configuration_key_union(configs0, configs1)))

    def test_configuration_key_union_many(self):
        """The union of many different keys should be all of them
        """
        configs0 = [(2, 0)]
        configs1 = [(5, 1)]
        configs2 = [(6, -2)]
        configs3 = [(3, -3)]
        refset = set([(2, 0), (5, 1), (6, -2), (3, -3)])
        testset = set(
            util.configuration_key_union(configs0, configs1, configs2,
                                         configs3))
        self.assertSetEqual(testset, refset)

    def test_configuration_key_intersection_none(self):
        """If there are no keys in common the intersection should be zero
        """
        self.assertListEqual([],
                             util.configuration_key_intersection([(2, 0)],
                                                                 [(2, 2)]))

    def test_configuration_key_intersection(self):
        """Check that the intersection returns the intersection
        """
        configs0 = [(10, 0), (3, 1), (5, -1)]
        configs1 = [(2, 0), (3, 1), (3, -1)]
        self.assertListEqual([(3, 1)],
                             util.configuration_key_intersection(
                                 configs0, configs1))

    def test_invert_bitstring_with_mask(self):
        """When inverting the occupation we want to maintain the number of orbitals
        """
        ref = 8
        self.assertEqual(ref, util.invert_bitstring_with_mask(7, 4))
        ref = 8 + 16 + 32 + 64 + 128
        self.assertEqual(ref, util.invert_bitstring_with_mask(7, 8))

    def test_ltlt_index_min(self):
        """If we have a zero dimesnion tensor there should be no pointers to it
        """
        _gtest = util.ltlt_index_generator(0)
        _test = [i for i in _gtest]
        self.assertListEqual(_test, [])

    def test_ltlt_index(self):
        """Access unique elements of a lower triangular lower triangular
        matrix
        """
        index_list = [(0, 0, 0, 0), (1, 0, 0, 0), (1, 0, 1, 0), (1, 1, 0, 0),
                      (1, 1, 1, 0), (1, 1, 1, 1)]
        _gtest = util.ltlt_index_generator(2)
        _test = [i for i in _gtest]
        self.assertListEqual(_test, index_list)

    def test_bitstring_groundstate(self):
        """The ground state bitstring has the n lowest bits flipped
        """
        self.assertEqual(15, util.init_bitstring_groundstate(4))

    def test_qubit_particle_number_sector(self):
        """Find the vectors which are the basis for a particular particle
        number.
        """
        zero = 0
        one = 1
        ref = [
            numpy.array([zero, zero, one, zero], dtype=numpy.int),
            numpy.array([zero, one, zero, zero], dtype=numpy.int)
        ]
        test = util.qubit_particle_number_sector(2, 1)
        for i, j in zip(test, ref):
            self.assertEqual(i.all(), j.all())

    def test_qubit_particle_number_index_spin(self):
        """Find the indexes which point to the correct coefficients in a qubit
        particle number sector and return the total spin.
        """
        ref = [(3, 0), (5, -2), (6, 0), (9, 0), (10, 2), (12, 0)]
        test = util.qubit_particle_number_index_spin(4, 2)
        self.assertListEqual(ref, test)

    def test_qubit_config_sector(self):
        """Find the basis vectors for a particular particle number and spin
        configuration
        """
        zero = 0
        one = 1
        lowstate = [
            zero, zero, one, zero, zero, zero, zero, zero, zero, zero, zero,
            zero, zero, zero, zero, zero
        ]
        highstate = [
            zero, zero, zero, zero, zero, zero, zero, zero, one, zero, zero,
            zero, zero, zero, zero, zero
        ]
        ref = [
            numpy.array(lowstate, dtype=numpy.int),
            numpy.array(highstate, dtype=numpy.int)
        ]
        test = util.qubit_config_sector(4, 1, 1)
        for i, j in zip(test, ref):
            self.assertEqual(i.all(), j.all())
        ref = [numpy.array([zero, zero, zero, one], dtype=numpy.int)]
        test = util.qubit_config_sector(2, 2, 0)
        for i, j in zip(test, ref):
            self.assertEqual(i.all(), j.all())

    def test_qubit_particle_number_index(self):
        """Find the indexes which point to the correct coefficients in a qubit
        particle number sector and return the total spin.
        """
        ref = [1, 2, 4, 8]
        test = util.qubit_particle_number_index(4, 1)
        self.assertListEqual(ref, test)

    def test_qubit_vacuum(self):
        """The qubit vacuum is the first vector in the qubit basis.
        """
        _gs = numpy.array([1. + .0j, 0. + .0j, 0. + .0j, 0. + .0j],
                          dtype=numpy.complex64)
        self.assertListEqual(list(_gs), list(util.init_qubit_vacuum(2)))

    def test_sort_config_keys(self):
        """Keys are sorted by particle number and then by m_s
        """
        ref = [(0, 0), (1, -1), (3, -3), (3, 1), (5, -2), (5, 1)]
        keys = [(5, 1), (5, -2), (0, 0), (1, -1), (3, -3), (3, 1)]
        test = util.sort_configuration_keys(keys)
        self.assertListEqual(test, ref)

    def test_validate_config(self):
        """Make sure that the configuration validation routine identifies
        problematic values
        """
        self.assertRaises(ValueError, util.validate_config, 0, 0, -1)
        self.assertRaises(ValueError, util.validate_config, 3, 0, 2)
        self.assertRaises(ValueError, util.validate_config, 0, 3, 2)
        self.assertRaises(ValueError, util.validate_config, -1, 1, 2)
        self.assertRaises(ValueError, util.validate_config, 1, -1, 2)
        self.assertIsNone(util.validate_config(0, 0, 0))
        self.assertIsNone(util.validate_config(0, 0, 1))

    def test_zero_transform(self):
        """Ensure that things that should transform do and those that shouldn't
        dont
        """
        self.assertFalse(util.zero_transform(1 + 2 + 4, 8, 3, 6))
        self.assertTrue(util.zero_transform(2 + 4, 8, 1, 6))
        self.assertTrue(util.zero_transform(2 + 4, 4, 2, 6))

    def test_parity_sort_list(self):
        """Sort a list of lists according to the parity of the index in the
        0th element.
        """
        unchanged = [
            [6, ['these', 'values', {
                'dont': 6724
            }, tuple(['mat', 'ter'])]],
            [7, ['these', 'values', {
                'dont': 6724
            }, tuple(['mat', 'ter'])]],
            [3, ['these', 'values', {
                'dont': 6724
            }, tuple(['mat', 'ter'])]],
            [15, ['these', 'values', {
                'dont': 6724
            }, tuple(['mat', 'ter'])]]
        ]
        nswap, test = util.paritysort_list(unchanged)
        self.assertEqual(nswap, 0)
        self.assertListEqual(unchanged, test)
