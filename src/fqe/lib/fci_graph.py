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
"""This file implements wrappers to some of the functions in fqe.fci_graph
"""

from ctypes import c_int, c_bool, POINTER

import numpy
from numpy.ctypeslib import ndpointer

from fqe.lib import lib_fqe

def _build_mapping_strings(strings, zmat, nele: int, norb: int):
    func = lib_fqe.build_mapping_strings
    c_ptr_map = POINTER(c_int * 3)

    func.argtypes = [
        POINTER(c_ptr_map),
        ndpointer(
            shape=(norb, norb),
            dtype=numpy.int32,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(norb ** 2, 2),
            dtype=numpy.int32,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        c_int,
        ndpointer(
            shape=(len(strings),),
            dtype=numpy.uint32,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        c_int,
        c_bool,
        ndpointer(
            shape=(nele, norb),
            dtype=numpy.int32,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        c_int
    ]

    # Cast strings if not ndarray
    if not isinstance(strings, numpy.ndarray) \
            or strings.dtype != numpy.uint32 \
            or not strings.flags['C']:
        strings = numpy.array(list(strings), dtype=numpy.uint32)

    maplengths = numpy.zeros((norb, norb), dtype=numpy.int32)
    exc_dexc = numpy.asarray(
        [(i, j) for i in range(norb) for j in range(norb)],
        dtype=numpy.int32
    ).reshape(-1, 2)  # reshape needed if exc_dexc is zero-size array

    # Count maplengths
    func(None, maplengths, exc_dexc, len(exc_dexc), strings, len(strings),
         True, zmat, norb)

    # allocate maps
    out = {
        tuple(ed): numpy.zeros((ml, 3), dtype=numpy.int32)
        for ed, ml in zip(exc_dexc, maplengths.ravel())
    }
    out_data = [out[tuple(ed)].ctypes.data_as(c_ptr_map) for ed in exc_dexc]
    # Fill in maps
    func(
        (c_ptr_map * len(exc_dexc))(*out_data),
        maplengths,
        exc_dexc,
        len(exc_dexc),
        strings,
        len(strings),
        False,
        zmat,
        norb
    )

    return out
