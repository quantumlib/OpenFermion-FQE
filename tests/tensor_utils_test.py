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
"""Unit testing for tensor utilities."""

from collections import deque
import pytest

import numpy as np

from fqe.tensor import tensor_utils


def test_build_symmetry_operations():
    """Check that all the symmetry operations are built correctly."""
    one = np.identity(4)

    two = np.zeros((4, 4))
    two[1, 0] = 1.0
    two[0, 1] = 1.0
    two[2, 2] = 1.0
    two[3, 3] = 1.0

    three = np.zeros((4, 4))
    three[0, 0] = 1.0
    three[1, 1] = 1.0
    three[3, 2] = 1.0
    three[2, 3] = 1.0

    ref = [[one, 1.0, False], [two, 1.0, True], [three, 1.0, True]]

    test = [
        [[1, 2, 3, 4], 1.0, False],
        [[2, 1, 3, 4], 1.0, True],
        [[1, 2, 4, 3], 1.0, True],
    ]

    tensor_utils.build_symmetry_operations(test)
    for val in range(3):
        assert ref[val][0].all() == test[val][0].all()


def test_confirm_symmetry_four_index_all():
    """Check that all symmetry operations are built correctly."""
    test = [
        [[1, 2, 3, 4], 1.0, False],
        [[2, 1, 3, 4], -1.0, True],
        [[1, 2, 4, 3], 1.0, True],
        [[1, 4, 2, 3], 1.0, False],
        [[4, 3, 2, 1], 1.0, False],
    ]
    matrix = np.zeros((2, 2, 2, 2), dtype=np.complex64)
    assert tensor_utils.confirm_symmetry(matrix, test) is None


def test_confirm_symmetry_four_index_conjugate():
    """Check that complex conjugation is built correctly."""
    symmetry = [
        [[1, 2, 3, 4], 1.0, False],
        [[2, 1, 3, 4], 1.0, True],
        [[1, 2, 4, 3], 1.0, True],
    ]

    matrix = np.zeros((2, 2, 2, 2), dtype=np.complex64)
    matrix[0, 0, 0, 0] = 1.0
    matrix[1, 0, 0, 0] = 1.5
    matrix[0, 1, 0, 0] = 1.5
    matrix[0, 0, 1, 0] = 2.0
    matrix[0, 0, 0, 1] = 2.0
    matrix[1, 1, 0, 0] = 2.5 - 0.0j
    matrix[0, 0, 1, 1] = 3.0 + 0.0j
    matrix[0, 1, 1, 1] = 3.5
    matrix[1, 0, 1, 1] = 3.5
    matrix[1, 1, 0, 1] = 4.0
    matrix[1, 1, 1, 0] = 4.0
    matrix[1, 0, 1, 0] = 5.0 + 1.0j
    matrix[0, 1, 1, 0] = 5.0 - 1.0j
    matrix[1, 0, 0, 1] = 5.0 - 1.0j
    matrix[0, 1, 0, 1] = 5.0 + 1.0j
    matrix[1, 1, 1, 1] = 6.0

    assert tensor_utils.confirm_symmetry(matrix, symmetry) is None


def test_confirm_symmetry_real():
    """Check that real symmetry is obeyed."""
    symmetry = [[[1, 2, 3, 4], 1.0, True]]

    matrix = np.zeros((2, 2, 2, 2), dtype=np.complex64)
    matrix[0, 0, 0, 0] = 1.0 + 0.0j
    matrix[1, 1, 0, 0] = 2.0 + 0.0j
    matrix[0, 0, 1, 1] = 2.0 + 0.0j
    matrix[1, 1, 1, 1] = 3.0 + 0.0j

    assert tensor_utils.confirm_symmetry(matrix, symmetry) is None


def test_confirm_anti_symmetric():
    """Check antisymmetric cases."""
    symmetry = [
        [[1, 2], 1.0, False],
        [[2, 1], -1.0, False],
    ]

    temp = np.random.rand(4, 4) + np.random.rand(4, 4) * 1.0j
    matrix = temp - temp.T

    assert tensor_utils.confirm_symmetry(matrix, symmetry) is None


def test_confirm_hermitian():
    """Check Hermitian cases."""
    symmetry = [
        [[1, 2], 1.0, False],
        [[2, 1], 1.0, True],
    ]

    temp = np.random.rand(4, 4) + np.random.rand(4, 4) * 1.0j
    matrix = temp + np.conjugate(temp.T)

    assert tensor_utils.confirm_symmetry(matrix, symmetry) is None


def test_confirm_failure():
    """Check failure cases."""
    symmetry = [
        [[1, 2], 1.0, False],
        [[2, 1], 1.0, True],
    ]

    temp = np.random.rand(4, 4) + np.random.rand(4, 4) * 1.0j
    matrix = temp + np.conjugate(temp.T)
    matrix[1, 0] = -matrix[0, 1]

    with pytest.raises(ValueError):
        tensor_utils.confirm_symmetry(matrix, symmetry)


def test_index_queue():
    """Generate tuples as pointers into all elements of an arbitrary-sized
    matrix to confirm the symmetry operations.
    """
    test = deque([tuple([i]) for i in range(10)])
    assert test == tensor_utils.index_queue(1, 10)

    test = deque([
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 2),
    ])

    assert test == tensor_utils.index_queue(2, 3)


def test_validate_unity_errors():
    """Check that the validate unity program fails for each possible
    failure case.
    """
    with pytest.raises(ValueError):
        tensor_utils.validate_unity([[1, 2, 3, 4, 5, 6], 2.0, True])


def test_validate_unity_success():
    """Check that the validate unity program fails for each possible
    failure case.
    """
    assert tensor_utils.validate_unity([[1, 2, 3, 4, 5, 6], 1.0, False])


def test_validate_matrix():
    """Check that the validate unity program fails for each possible
    failure case.
    """
    assert tensor_utils.validate_unity([[1, 2, 3, 4, 5, 6], 1.0, False])

    with pytest.raises(ValueError):
        tensor_utils.validate_unity([[1, 5, 3, 4, 5, 6], 1.0, False])


def test_only_unity():
    """If the only symmetry is unity there is no validation to do."""
    symmetry = [[[1, 2], 1.0, False]]
    temp = np.random.rand(4, 4) + np.random.rand(4, 4) * 1.0j
    assert tensor_utils.confirm_symmetry(temp, symmetry) is None
