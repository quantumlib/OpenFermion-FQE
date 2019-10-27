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
"""Unit testing for tensor utilities
"""

import unittest
from collections import deque

import numpy

from fqe.tensor import tensor_utils

class TensorUtilsTest(unittest.TestCase):
    """Test cases for tensor utilities
    """


    def test_build_symmetry_operations(self):
        """Check that we build all the symmetry operations correctly.
        """
        one = numpy.identity(4)
        two = numpy.zeros((4, 4))
        two[1, 0] = 1.0
        two[0, 1] = 1.0
        two[2, 2] = 1.0
        two[3, 3] = 1.0
        three = numpy.zeros((4, 4))
        three[0, 0] = 1.0
        three[1, 1] = 1.0
        three[3, 2] = 1.0
        three[2, 3] = 1.0
        ref = [
            [one, 1.0, False],
            [two, 1.0, True],
            [three, 1.0, True]
        ]
        test = [
            [[1, 2, 3, 4], 1.0, False],
            [[2, 1, 3, 4], 1.0, True],
            [[1, 2, 4, 3], 1.0, True]
        ]
        tensor_utils.build_symmetry_operations(test)
        for val in range(3):
            self.assertEqual(ref[val][0].all(), test[val][0].all())


    def test_confirm_symmetry_four_index_all(self):
        """Check that we build all the symmetry operations correctly.
        """
        test = [
            [[1, 2, 3, 4], 1.0, False],
            [[2, 1, 3, 4], -1.0, True],
            [[1, 2, 4, 3], 1.0, True],
            [[1, 4, 2, 3], 1.0, False],
            [[4, 3, 2, 1], 1.0, False]
        ]
        matrix = numpy.zeros((2, 2, 2, 2), dtype=numpy.complex64)
        self.assertIsNone(tensor_utils.confirm_symmetry(matrix, test))


    def test_confirm_symmetry_four_index_conjugate(self):
        """Check that we build complex conjugation correctly
        """
        symm = [
            [[1, 2, 3, 4], 1.0, False],
            [[2, 1, 3, 4], 1.0, True],
            [[1, 2, 4, 3], 1.0, True]
        ]
        matrix = numpy.zeros((2, 2, 2, 2), dtype=numpy.complex64)
        matrix[0, 0, 0, 0] = 1.0
        matrix[1, 0, 0, 0] = 1.5
        matrix[0, 1, 0, 0] = 1.5
        matrix[0, 0, 1, 0] = 2.0
        matrix[0, 0, 0, 1] = 2.0
        matrix[1, 1, 0, 0] = 2.5 - 0.j
        matrix[0, 0, 1, 1] = 3.0 + 0.j
        matrix[0, 1, 1, 1] = 3.5
        matrix[1, 0, 1, 1] = 3.5
        matrix[1, 1, 0, 1] = 4.0
        matrix[1, 1, 1, 0] = 4.0
        matrix[1, 0, 1, 0] = 5.0 + 1.j
        matrix[0, 1, 1, 0] = 5.0 - 1.j
        matrix[1, 0, 0, 1] = 5.0 - 1.j
        matrix[0, 1, 0, 1] = 5.0 + 1.j
        matrix[1, 1, 1, 1] = 6.0
        self.assertIsNone(tensor_utils.confirm_symmetry(matrix, symm))


    def test_confirm_symmetry_real(self):
        """Check that real symmetry is obeyed
        """
        symm = [
            [[1, 2, 3, 4], 1.0, True]
        ]
        matrix = numpy.zeros((2, 2, 2, 2), dtype=numpy.complex64)
        matrix[0, 0, 0, 0] = 1. + 0.j
        matrix[1, 1, 0, 0] = 2. + 0.j
        matrix[0, 0, 1, 1] = 2. + 0.j
        matrix[1, 1, 1, 1] = 3. + 0.j
        self.assertIsNone(tensor_utils.confirm_symmetry(matrix, symm))


    def test_confirm_anti_symmetric(self):
        """Check antisymmetric cases
        """
        symm = [
            [[1, 2], 1.0, False],
            [[2, 1], -1.0, False],
        ]
        temp = numpy.random.rand(4, 4) + numpy.random.rand(4, 4)*1.j
        matrix = temp - temp.T
        self.assertIsNone(tensor_utils.confirm_symmetry(matrix, symm))


    def test_confirm_hermetian(self):
        """Check Hermetian cases
        """
        symm = [
            [[1, 2], 1.0, False],
            [[2, 1], 1.0, True],
        ]
        temp = numpy.random.rand(4, 4) + numpy.random.rand(4, 4)*1.j
        matrix = temp + numpy.conjugate(temp.T)
        self.assertIsNone(tensor_utils.confirm_symmetry(matrix, symm))


    def test_confirm_failure(self):
        """Check failure cases
        """
        symm = [
            [[1, 2], 1.0, False],
            [[2, 1], 1.0, True],
        ]
        temp = numpy.random.rand(4, 4) + numpy.random.rand(4, 4)*1.j
        matrix = temp + numpy.conjugate(temp.T)
        matrix[1, 0] = -matrix[0, 1]
        self.assertRaises(ValueError, tensor_utils.confirm_symmetry, matrix, symm)


    def test_index_queue(self):
        """Genreate tuples as pointers into all elements of an arbitray sized
        matrix to confirm the symmetry operations
        """
        test = deque([tuple([i]) for i in range(10)])
        self.assertEqual(test, tensor_utils.index_queue(1, 10))

        test = deque([
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 1),
            (1, 2),
            (2, 0),
            (2, 1),
            (2, 2)
        ])
        self.assertEqual(test, tensor_utils.index_queue(2, 3))


    def test_validate_unity_errors(self):
        """Check that the validate unity program fails for each possible
        failure case.
        """
        self.assertRaises(ValueError, tensor_utils.validate_unity, [[1, 2, 3, 4, 5, 6], 2.0, True])


    def test_validate_unity_success(self):
        """Check that the validate unity program fails for each possible
        failure case.
        """
        self.assertTrue(tensor_utils.validate_unity([[1, 2, 3, 4, 5, 6], 1.0, False]))


    def test_validate_matrix(self):
        """Check that the validate unity program fails for each possible
        failure case.
        """
        self.assertTrue(tensor_utils.validate_unity([[1, 2, 3, 4, 5, 6], 1.0, False]))
        self.assertRaises(ValueError,
                          tensor_utils.validate_unity, [[1, 5, 3, 4, 5, 6], 1.0,
                                                        False])


    def test_only_untiy(self):
        """If the only symmetry is unity there is no validation to do
        """
        symm = [[[1, 2], 1.0, False]]
        temp = numpy.random.rand(4, 4) + numpy.random.rand(4, 4)*1.j
        self.assertIsNone(tensor_utils.confirm_symmetry(temp, symm))
