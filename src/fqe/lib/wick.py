#   Copyright 2021 Google LLC

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
"""This file implements wrappers to some of the functions in fqe.wick
"""

from ctypes import c_int, c_double
from typing import List, Tuple, Optional

import numpy
from numpy.ctypeslib import ndpointer

from fqe.lib import lib_fqe


def _wickfill(target: numpy.ndarray, source: Optional[numpy.ndarray],
              indices: Optional[numpy.ndarray], factor: float,
              delta: Optional[numpy.ndarray]) -> numpy.ndarray:
    """
    This function is an internal utility that wraps a C-implementation to fill
    in custom RDMs using particle RDMs. The result of Wick's theorem is passed
    as lists (indices and delta) and a factor associated with it. The results
    are stored in target.

    Args:
        target (numpy.ndarray) - output array that stores reordered RDMs

        source (numpy.ndarray) - input array that stores one of the particle \
            RDMs

        indices (numpy.ndarray) - index mapping

        factor (float) - factor associated with this contribution

        delta (numpy.ndarray) - Kronecker delta's due to Wick's theorem

    Returns:
        target (numpy.ndarray) - Returns the output array. If target in the \
            input Args was not C-contigious, this can be a new numpy object.
    """
    func = lib_fqe.wickfill

    func.argtypes = [
        ndpointer(dtype=numpy.complex128, flags=('C_CONTIGUOUS', 'ALIGNED')),
        ndpointer(dtype=numpy.complex128, flags=('C_CONTIGUOUS', 'ALIGNED')),
        ndpointer(dtype=numpy.uint32,
                  flags=('C_CONTIGUOUS', 'ALIGNED')), c_double,
        ndpointer(dtype=numpy.uint32, flags=('C_CONTIGUOUS', 'ALIGNED')), c_int,
        c_int, c_int
    ]

    norb = target.shape[0]
    srank = len(source.shape) // 2 if source is not None else 0
    trank = len(target.shape) // 2

    if indices is None or len(indices) == 0:
        indices = numpy.zeros((1,), dtype=numpy.uint32)
    if delta is None or len(delta) == 0:
        delta = numpy.zeros((1,), dtype=numpy.uint32)
    if source is None or len(source) == 0:
        source = numpy.zeros((1,), dtype=numpy.complex128)

    # Fixes if target or source is not C contigious
    if not target.flags['C']:
        target = target.copy()
    if not source.flags['C']:
        source = source.copy()

    func(target, source, indices, factor, delta, norb, trank, srank)
    return target
