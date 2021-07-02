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
"""This file implements some wrappers to linear algebra functions.
"""

from ctypes import c_int, c_int64

import numpy
from numpy.ctypeslib import ndpointer
from numpy import ndarray as Nparray

from fqe.lib import lib_fqe, c_double_complex

from scipy.linalg.cython_blas cimport zaxpy
from libc.stdint cimport uintptr_t
include "blas_helpers.pxi"

def _zimatadd(outp: 'Nparray', inp: 'Nparray', alpha: complex):
    """Wrapper to C function `zimatadd`.

    Performs `outp += alpha * inp.T`

    Returns:
        nothing: Implace adding in outp
    """
    func = lib_fqe.zimatadd
    if outp.ndim != 2:
        raise ValueError(f'outp of shape {outp.shape} not two-dimensional')

    dim1, dim2 = outp.shape

    func.argtypes = [
        c_int,
        c_int,
        ndpointer(
            shape=(dim1, dim2),
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(dim2, dim1),
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        c_double_complex,
        c_int64
    ]

    cdef blasfunctions blas_functions
    blas_functions.zaxpy = zaxpy
    cdef object pointer = <uintptr_t>&blas_functions

    alpha_c = c_double_complex(alpha.real, alpha.imag)
    func(outp.shape[0], outp.shape[1], outp, inp, alpha_c, pointer)
