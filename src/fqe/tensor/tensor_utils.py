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
"""Tensor utilities to evaluate symmetry and eventually create dense storage."""

from collections import deque
from typing import Any, Deque, List, Tuple

import numpy as np


def build_symmetry_operations(symmetry: List[Any]) -> None:
    """Take the list of allowed permutations and build the symmetry operations
    allowed for the operation, modifying `symmetry` in place.

    Args:
        symmetry: A list containing all the allowed permutations.
    """
    dim = len(symmetry[0][0])
    unit = np.identity(dim, dtype=int)
    for permutation in symmetry:
        perm = unit[:, np.argsort(permutation[0])]
        permutation[0] = perm


def confirm_symmetry(mat: np.ndarray, symmetry: List[Any]) -> None:
    """Digest the allowed permutations to validate the underlying structure.

    Args:
        mat: Matrix to confirm symmetry in.
        symmetry: A list containing all the information regarding symmetry of
            the matrix. The first element can be an identity element with the
            indices in order, a parity of 1.0 and no complex conjugation.
            Each entry should specify the action of the symmetry on the
            indexes, a parity associated with the permutation and whether the
            term should be conjugated. The first term should be the unity
            operation.
    """
    is_unity = validate_unity(symmetry[0])
    if len(symmetry) == 1 and is_unity:
        return
    build_symmetry_operations(symmetry)
    validate_matrix_symmetry(mat, symmetry)


def index_queue(dim: int, highest_index: int) -> Deque[Tuple[int, ...]]:
    """Generate all index pointers into the matrix of interest.

    Args:
        dim: The size of the matrix of interest.
        highest_index: The maximum value allowable in the matrix.

    Returns:
        queue: A queue containing all possible pointers into the matrix.
    """
    i_queue: List[Tuple[int, ...]] = []
    if dim == 1:
        for i in range(highest_index):
            i_queue.append(tuple([i]))
    else:
        total = highest_index**dim - 1
        ticker = [0 for _ in range(dim)]
        i_queue.append(tuple(ticker))

        for _ in range(total):
            for i in reversed(range(dim)):
                ticker[i] += 1
                if ticker[i] < highest_index:
                    break
                ticker[i] = 0
            i_queue.append(tuple(ticker))

    return deque(i_queue)


def validate_matrix_symmetry(matrix: np.ndarray,
                             symmetry: List[Any],
                             threshold: float = 1.0e-8) -> None:
    """Go through every element of the matrix and check that the symmetry
    operations are valid up to a threshold.

    Args:
        matrix: A matrix of interest.
        symmetry: Symmetry that should be validated.
        threshold: The limit at which a symmetry operation is valid.

    Raises:
        ValueError: If there is an error with a symmetry in a permutation.
    """
    all_index = index_queue(len(matrix.shape), matrix.shape[0])
    while all_index:
        index = all_index.popleft()
        value = matrix[index]
        for permu in symmetry[1:]:
            test_index = tuple(np.dot(index, permu[0]))
            test_value = matrix[test_index]

            if permu[2]:
                ref_value = permu[1] * np.conj(value)
            else:
                ref_value = permu[1] * value

            if np.abs(test_value - ref_value) > threshold:
                raise ValueError("Error with symmetry in permutation {} -> {}."
                                 " {} != {}".format(index, test_index,
                                                    ref_value, test_value))

            try:
                all_index.remove(test_index)
            except ValueError:
                pass


def validate_unity(unity_permutation: List[Any]) -> bool:
    """Checks that the input permutation is the unity permutation, i.e., an
    object s of type List[List[int], float, bool] such that

    * s[0][0] > -1,
    * s[0][i] < s[0][j] for i < j,
    * s[1] = 1.0, and
    * s[2] = False.

    Args:
        unity_permutation: A permutation object to compare to the unity
            operation.

    Returns:
        True if the input permutation is the unity permutation, else False.

    Raises:
        ValueError: If the input permutation is invalid.
    """
    if unity_permutation[1] != 1.0:
        raise ValueError("The unity permutation does not have a phase of 1.0")

    if unity_permutation[2]:
        return False

    static = unity_permutation[0]
    lowlimit = -1

    for index in static:
        if index < lowlimit:
            raise ValueError("The first entry is not unity")
        lowlimit = index

    return True
