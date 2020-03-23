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

"""Tensor utitlies to evaluate symmetry and eventually create dense storage
"""
from typing import Any, Deque, List, Tuple

from collections import deque

import numpy


def build_symmetry_operations(symmetry: List[Any]) -> None:
    """Take the list of allowed permutations and build the symmetry operations
    allowed for the operation.

    Args:
        symmetry (list[list[int], float, bool]) - a list containing all the \
            allowed permuations

    Returns:
        symmetry (list[numpy.array(dtype=int), float, bool]) - modifes the \
        symmetry in place
    """
    dim = len(symmetry[0][0])
    unit = numpy.identity(dim, dtype=int)
    for permutation in symmetry:
        perm = unit[:, numpy.argsort(permutation[0])]
        permutation[0] = perm


def confirm_symmetry(mat: numpy.ndarray, symmetry: List[Any]) -> None:
    """Digest the allowed permutations to validate the underlying structure

    Args:
        symmetry (list[list[ints], float, bool]) - a list containing all the \
            information regarding symmetry of the matrix.  The first element \
            can be an identity element with the indexes in order, a partiy of \
            1.0 and no complex conjugation.  Each entry should specify the \
            action of the symmetry on the indexes, a parity associated with \
            the permutation and whether the term should be conjugated.  The \
            first term should be the unity operation.

    Returns:
        nothing - checks for errors in the passed matrix
    """
    is_unity = validate_unity(symmetry[0])
    if len(symmetry) == 1 and is_unity:
        return
    build_symmetry_operations(symmetry)
    validate_matrix_symmetry(mat, symmetry)


def index_queue(dim: int, highest_index: int) -> Deque[Tuple[int, ...]]:
    """Generate all index pointers into the matrix of interest

    Args:
        dim (int) - the size of the matrix of interest

        highest_index (int) - the maximum value allowable in the matrix

    Returns:
        queue (list) - a queue containing all possbile pointers into the \
            matrix
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


def validate_matrix_symmetry(matrix: numpy.ndarray, symmetry: List[Any],
                             threshhold: float = 1.e-8) -> None:
    """Go through every element of the matrix and check that the symmetry
    operations are valid up to a threshhold.

    Args:
        matrix (numpy.array) - a matrix of interest

        symmetry (list[numpy.array(dtype=int), float, bool]) - symmetry that \
            should be validated

        threshold (float) - the limit at which a symmetry operation is valid

    Returns:
        nothing - checks for errors in the passed matrix
    """
    all_index = index_queue(len(matrix.shape), matrix.shape[0])
    while all_index:
        index = all_index.popleft()
        value = matrix[index]
        for permu in symmetry[1:]:
            test_index = tuple(numpy.dot(index, permu[0]))
            test_value = matrix[test_index]

            if permu[2]:
                ref_value = permu[1]*numpy.conj(value)
            else:
                ref_value = permu[1]*value

            if numpy.abs(test_value - ref_value) > threshhold:
                raise ValueError('Error with symmetry in permutation {} -> {}.'
                                 ' {} != {}'.format(index, test_index,
                                                    ref_value, test_value))

            try:
                all_index.remove(test_index)
            except ValueError:
                pass


def validate_unity(unity_permutation: List[Any]) -> bool:
    """Check that the initial permutation passed in is the unity operation and
    return data useful for validating the remaining permutations

    Args:
        unity_permutation (list[int], complex, bool)

    Returns:
        bool - raises errors if the unit permutation is false
    """
    static = unity_permutation[0]
    lowlimit = -1

    for index in static:
        if index < lowlimit:
            raise ValueError('The first entry is not unity')
        lowlimit = index

    if unity_permutation[1] != 1.0:
        raise ValueError('The unity permutation does not have a phase of 1.0')

    if unity_permutation[2]:
        return False

    return True
