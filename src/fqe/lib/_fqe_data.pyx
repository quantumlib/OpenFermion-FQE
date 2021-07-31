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
"""This file implements wrappers to some of the functions in fqe.fqe_data
"""

from ctypes import c_int, c_bool, c_double, POINTER, c_int64, byref, c_uint64
from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy
from numpy.ctypeslib import ndpointer
from openfermion import up_index, down_index
from fqe.bitstring import integer_index

from scipy.linalg.cython_blas cimport zaxpy, zscal
from libc.stdint cimport uintptr_t

from fqe.lib import lib_fqe, c_double_complex
include "blas_helpers.pxi"

if TYPE_CHECKING:
    from numpy import ndarray as Nparray
    from fqe.fqe_data import FqeData
    from openfermion import BinaryCode


def _lm_apply_array1(coeff, h1e, dexc, lena, lenb, norb, alpha=True, out=None):
    func = lib_fqe.lm_apply_array1
    len1 = lena if alpha else lenb
    len2 = lenb if alpha else lena
    outshape = (lena, lenb)
    ndexc = dexc.shape[1]

    func.argtypes = [
        ndpointer(
            shape=(lena, lenb),
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=outshape,
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(len1, ndexc, 3),
            dtype=numpy.int32,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        c_int,
        c_int,
        c_int,
        ndpointer(
            shape=(norb, norb),
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        c_int,
        c_bool,
        c_int64
    ]

    dtype = numpy.complex128
    if out is None:
        out = numpy.zeros(outshape, dtype=dtype)

    if h1e.dtype != dtype or not h1e.flags['C']:
        h1e = numpy.asarray(h1e, dtype=dtype).copy()

    if coeff.dtype != dtype or not coeff.flags['C']:
        coeff = numpy.asarray(coeff, dtype=dtype).copy()

    cdef blasfunctions blas_functions
    blas_functions.zaxpy = <zaxpy_func>zaxpy
    cdef object pointer = <uintptr_t>&blas_functions

    func(coeff, out, dexc, lena, lenb, ndexc, h1e, norb, alpha, pointer)
    return out

def _lm_apply_array1_alpha_column(coeff, h1e, index, exc, exc2, lena, lenb, icol):
    func = lib_fqe.lm_apply_array1_column_alpha
    nexc0 = exc.shape[0]
    nexc1 = exc.shape[1]
    nexc2 = exc2.shape[0]

    func.argtypes = [
        ndpointer(
            shape=(lena, lenb),
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(nexc0,),
            dtype=numpy.int32,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(nexc0, nexc1, 3),
            dtype=numpy.int32,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(nexc2,),
            dtype=numpy.int32,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        ndpointer(
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        c_int,
        c_int64
    ]
    dtype = numpy.complex128
    if h1e.dtype != dtype or not h1e.flags['C']:
        h1e = numpy.asarray(h1e, dtype=dtype).copy()

    cdef blasfunctions blas_functions
    blas_functions.zaxpy = <zaxpy_func>zaxpy
    blas_functions.zscal = <zscal_func>zscal
    cdef object pointer = <uintptr_t>&blas_functions

    func(coeff, index, exc, exc2, lena, lenb, nexc0, nexc1, nexc2, h1e, icol, pointer)


def _sparse_apply_array1(coeff, h1e, dexc, lena, lenb, norb, jorb, alpha=True, out=None):
    func = lib_fqe.lm_apply_array1_sparse
    len1 = lena if alpha else lenb
    len2 = lenb if alpha else lena
    outshape = (len1, len2)
    ndexc = dexc.shape[1]

    func.argtypes = [
        ndpointer(
            shape=(lena, lenb),
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=outshape,
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(len1, ndexc, 3),
            dtype=numpy.int32,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        c_int,
        c_int,
        c_int,
        ndpointer(
            shape=(norb, norb),
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        c_int,
        c_int,
        c_bool,
        c_int64
    ]

    dtype = numpy.complex128
    if out is None:
        out = numpy.zeros(outshape, dtype=dtype)

    if h1e.dtype != dtype or not h1e.flags['C']:
        h1e = numpy.asarray(h1e, dtype=dtype).copy()

    if coeff.dtype != dtype or not coeff.flags['C']:
        coeff = numpy.asarray(coeff, dtype=dtype).copy()

    cdef blasfunctions blas_functions
    blas_functions.zaxpy = <zaxpy_func>zaxpy
    cdef object pointer = <uintptr_t>&blas_functions

    func(coeff, out, dexc, lena, lenb, ndexc, h1e, norb, jorb, alpha, pointer)
    return out


def _lm_apply_array12_same_spin(coeff, h2e, dexc, len1, len2, norb, alpha=True,
                                out=None, dtype=None):
    func = lib_fqe.lm_apply_array12_same_spin
    ndexc = dexc.shape[1]
    outshape = (len1, len2)

    func.argtypes = [
        ndpointer(
            shape=outshape,
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=outshape,
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(len1, ndexc, 3),
            dtype=numpy.int32,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        c_int,
        c_int,
        c_int,
        ndpointer(
            shape=(norb, norb, norb, norb),
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        c_int,
        c_bool,
        c_int64
    ]

    dtype = numpy.complex128

    if out is None:
        out = numpy.zeros(outshape, dtype=dtype)

    if (h2e.dtype != dtype or not h2e.flags['C']):
        h2e = numpy.asarray(h2e, dtype=dtype).copy()

    if (coeff.dtype != dtype or not coeff.flags['C']):
        coeff = numpy.asarray(coeff, dtype=dtype).copy()

    cdef blasfunctions blas_functions
    blas_functions.zaxpy = <zaxpy_func>zaxpy
    cdef object pointer = <uintptr_t>&blas_functions

    func(coeff, out, dexc, len1, len2, ndexc, h2e, norb, alpha, pointer)
    return out


def _lm_apply_array12_diff_spin(coeff, h2e, adexc, bdexc, lena, lenb, norb,
                                out=None, dtype=None):
    func = lib_fqe.lm_apply_array12_diff_spin
    nadexc = adexc.shape[1]
    nbdexc = bdexc.shape[1]
    outshape = (lena, lenb)

    func.argtypes = [
        ndpointer(
            shape=outshape,
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=outshape,
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(lena, nadexc, 3),
            dtype=numpy.int32,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(lenb, nbdexc, 3),
            dtype=numpy.int32,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        c_int,
        c_int,
        c_int,
        c_int,
        ndpointer(
            shape=(norb, norb, norb, norb),
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        c_int,
    ]

    dtype = numpy.complex128

    if out is None:
        out = numpy.zeros(outshape, dtype=dtype)

    if (h2e.dtype != dtype or not h2e.flags['C']):
        h2e = numpy.asarray(h2e, dtype=dtype).copy()

    if (coeff.dtype != dtype or not coeff.flags['C']):
        coeff = numpy.asarray(coeff, dtype=dtype).copy()

    func(coeff, out, adexc, bdexc, lena, lenb, nadexc, nbdexc, h2e, norb)
    return out


def _make_dvec_part(coeff, maps, arange, brange, norb, lena, lenb, is_alpha,
                    out=None, dtype=None):
    func = lib_fqe.make_dvec_part
    anum = len(arange)
    bnum = len(brange)

    func.argtypes = [
        c_int, c_int,
        c_int, c_int,
        c_int, c_int,
        ndpointer(
            shape=(len(maps), 4),
            dtype=numpy.int32,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        c_int,
        ndpointer(
            shape=(lena if is_alpha else lenb, lenb if is_alpha else lena),
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(norb, norb, anum, bnum),
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        c_bool,
        c_int64
    ]

    if dtype is None:
        dtype = coeff.dtype

    if out is None:
        out = numpy.zeros((norb, norb, anum, bnum), dtype=dtype)

    if coeff.dtype != dtype or not coeff.flags['C']:
        coeff = numpy.asarray(coeff, dtype=dtype).copy()

    if maps.dtype != numpy.int32 or not maps.flags['C']:
        maps = numpy.asarray(maps, dtype=numpy.int32).copy()

    cdef blasfunctions blas_functions
    blas_functions.zaxpy = <zaxpy_func>zaxpy
    cdef object pointer = <uintptr_t>&blas_functions

    func(lena, lenb, arange.start, brange.start, anum, bnum,
         maps, len(maps), coeff, out, is_alpha, pointer)

    return out


def _make_coeff_part(out, coeff, maps, range_1, range_2):
    func = lib_fqe.make_coeff_part

    func.argtypes = [
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        ndpointer(
            shape=(len(maps), 4),
            dtype=numpy.int32,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        c_int,
        ndpointer(
            shape=(coeff.shape[0], coeff.shape[1], len(range_1), len(range_2)),
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            ndim=2,
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        c_int64
    ]

    dtype = numpy.complex128

    if out.dtype != dtype or not out.flags['C']:
        out = numpy.asarray(out, dtype=dtype).copy()

    if coeff.dtype != dtype or not coeff.flags['C']:
        coeff = numpy.asarray(coeff, dtype=dtype).copy()

    if maps.dtype != numpy.int32 or not maps.flags['C']:
        maps = numpy.asarray(maps, dtype=numpy.int32).copy()

    cdef blasfunctions blas_functions
    blas_functions.zaxpy = <zaxpy_func>zaxpy
    cdef object pointer = <uintptr_t>&blas_functions

    func(out.shape[0], out.shape[1], range_1.start, range_2.start,
         coeff.shape[2], coeff.shape[3], maps, len(maps), coeff, out,
         pointer)

    return out


def _make_dvec(dvec: 'Nparray', coeff: 'Nparray', mappings: List[Nparray],
               lena: int, lenb: int, is_alpha_mapping: bool) -> 'Nparray':
    func = lib_fqe.zdvec_make
    c_ptr_map = POINTER(c_int * 3)
    nmaps = len(mappings)

    func.argtypes = [
        POINTER(c_ptr_map),
        ndpointer(
            shape=(nmaps,),
            dtype=numpy.int32,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        c_int,
        ndpointer(
            shape=(lena, lenb),
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(nmaps, lena, lenb),
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        c_int,
        c_int,
        c_bool,
        c_int64
    ]

    cdef blasfunctions blas_functions
    blas_functions.zaxpy = <zaxpy_func>zaxpy
    cdef object pointer = <uintptr_t>&blas_functions
    func(
        (c_ptr_map * len(mappings))(
            *[mp.ctypes.data_as(c_ptr_map) for mp in mappings]
        ),
        numpy.array([len(x) for x in mappings], dtype=numpy.int32),
        nmaps,
        coeff,
        dvec.reshape(-1, lena, lenb),
        lena,
        lenb,
        is_alpha_mapping,
        pointer
    )

    return dvec


def _make_coeff(dvec: 'Nparray', coeff: 'Nparray', mappings: List[Nparray],
                lena: int, lenb: int, is_alpha_mapping: bool) -> 'Nparray':
    func = lib_fqe.zcoeff_make
    c_ptr_map = POINTER(c_int * 3)
    nmaps = len(mappings)

    func.argtypes = [
        POINTER(c_ptr_map),
        ndpointer(
            shape=(nmaps,),
            dtype=numpy.int32,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        c_int,
        ndpointer(
            shape=(lena, lenb),
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(nmaps, lena, lenb),
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        c_int,
        c_int,
        c_int,
        c_int64
    ]

    cdef blasfunctions blas_functions
    blas_functions.zaxpy = <zaxpy_func>zaxpy
    cdef object pointer = <uintptr_t>&blas_functions
    func(
        (c_ptr_map * len(mappings))(
            *[mp.ctypes.data_as(c_ptr_map) for mp in mappings]
        ),
        numpy.array([len(x) for x in mappings], dtype=numpy.int32),
        nmaps,
        coeff,
        dvec.reshape(-1, lena, lenb),
        lena,
        lenb,
        is_alpha_mapping,
        pointer
    )

    return coeff


def _apply_diagonal_coulomb(data: 'Nparray',
                            alpha_strings: 'Nparray',
                            beta_strings: 'Nparray',
                            diag: 'Nparray',
                            array: 'Nparray',
                            lena: int,
                            lenb: int,
                            nalpha: int,
                            nbeta: int,
                            norb: int) -> 'Nparray':

    if not diag.dtype == numpy.complex128:
        diag = diag.astype(numpy.complex128)
    if not array.dtype == numpy.complex128:
        array = array.astype(numpy.complex128)

    func = lib_fqe.zdiagonal_coulomb_apply
    func.argtypes = [
        ndpointer(
            shape=(lena,),
            dtype=numpy.uint64,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(lenb,),
            dtype=numpy.uint64,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(norb,),
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(norb, norb),
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(lena, lenb),
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        c_int,
        c_int,
        c_int,
        c_int,
        c_int
    ]
    func(alpha_strings, beta_strings, diag, array, data,
         lena, lenb, nalpha, nbeta, norb)


def _diagonal_coulomb(data: 'Nparray',
                      alpha_strings: 'Nparray',
                      beta_strings: 'Nparray',
                      diagonal: 'Nparray',
                      array: 'Nparray',
                      lena: int,
                      lenb: int,
                      nalpha: int,
                      nbeta: int,
                      norb: int) -> 'Nparray':

    func = lib_fqe.zdiagonal_coulomb
    func.argtypes = [
        ndpointer(
            shape=(lena,),
            dtype=numpy.uint64,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(lenb,),
            dtype=numpy.uint64,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(norb,),
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(norb, norb),
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(lena, lenb),
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        c_int,
        c_int,
        c_int,
        c_int,
        c_int
    ]
    func(alpha_strings, beta_strings, diagonal, array, data,
         lena, lenb, nalpha, nbeta, norb)
    return data

def _lm_apply_array12_same_spin_opt(coeff, h1e, h2e, dexc, len1, len2, norb,
                                    alpha=True, out=None, dtype=None):
    """Apply the same-spin (alpha or beta) part of a dense, spin-conserving operator
    by calling an optimized C function.

    Args:
        coeff (numpy.array) - Wavefunction coefficients
        h1e (numpy.array) - Tensor of 1-electron matrix elements
        h2e (numpy.array) - Tensor of 2-electron matrix elements
        dexc (numpy.array) - (index, orbital pair, parity) for excitations
            from each alpha (beta) string
        len1 (int) - number of alpha (beta) strings
        len2 (int) - number of beta (alpha) strings
        norb (int) - number of orbitals
        alpha (bool) - True (False) for alpha (beta) part
        out (numpy.array) - output array to increment

    Returns:
        numpy.array - output array
    """
    func = lib_fqe.lm_apply_array12_same_spin_opt
    ndexc = dexc.shape[1]
    outshape = (len1, len2)
    lend = len1 if alpha else len2

    func.argtypes = [
        ndpointer(
            shape=outshape,
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=outshape,
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(lend, ndexc, 3),
            dtype=numpy.int32,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        c_int,
        c_int,
        c_int,
        ndpointer(
            shape=(norb, norb),
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(norb, norb, norb, norb),
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        c_int,
        c_bool,
        c_int64
    ]

    dtype = numpy.complex128

    if out is None:
        out = numpy.zeros(outshape, dtype=dtype)

    if (h1e.dtype != dtype or not h1e.flags['C']):
        h1e = numpy.asarray(h1e, dtype=dtype).copy()

    if (h2e.dtype != dtype or not h2e.flags['C']):
        h2e = numpy.asarray(h2e, dtype=dtype).copy()

    if (coeff.dtype != dtype or not coeff.flags['C']):
        coeff = numpy.asarray(coeff, dtype=dtype).copy()

    cdef blasfunctions blas_functions
    blas_functions.zaxpy = <zaxpy_func>zaxpy
    cdef object pointer = <uintptr_t>&blas_functions

    func(coeff, out, dexc, len1, len2, ndexc, h1e, h2e, norb, alpha, pointer)
    return out


def _lm_apply_array12_diff_spin_opt(coeff, h2e, adexc, bdexc, lena, lenb, norb,
                                    out=None, dtype=None):
    """Apply the opposite-spin part of a dense, spin-conserving operator
    by calling an optimized C function.

    Args:
        coeff (numpy.array) - Wavefunction coefficients
        h2e (numpy.array) - Tensor of 2-electron matrix elements
        adexc (numpy.array) - (index, orbital pair, parity) for excitations
            from each alpha string
        bdexc (numpy.array) - (index, orbital pair, parity) for excitations
            from each beta string
        lena (int) - number of alpha strings
        lenb (int) - number of beta strings
        norb (int) - number of orbitals
        out (numpy.array) - output array to increment

    Returns:
        numpy.array - output array
    """
    func = lib_fqe.lm_apply_array12_diff_spin_opt
    nadexc = adexc.shape[1]
    nbdexc = bdexc.shape[1]
    outshape = (lena, lenb)

    func.argtypes = [
        ndpointer(
            shape=outshape,
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=outshape,
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(lena, nadexc, 3),
            dtype=numpy.int32,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(lenb, nbdexc, 3),
            dtype=numpy.int32,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        c_int,
        c_int,
        c_int,
        c_int,
        ndpointer(
            shape=(norb, norb, norb, norb),
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        c_int,
        c_int64
    ]

    dtype = numpy.complex128

    if out is None:
        out = numpy.zeros(outshape, dtype=dtype)

    if (h2e.dtype != dtype or not h2e.flags['C']):
        h2e = numpy.asarray(h2e, dtype=dtype).copy()

    if (coeff.dtype != dtype or not coeff.flags['C']):
        coeff = numpy.asarray(coeff, dtype=dtype).copy()

    cdef blasfunctions blas_functions
    blas_functions.zaxpy = <zaxpy_func>zaxpy
    cdef object pointer = <uintptr_t>&blas_functions
    func(coeff, out, adexc, bdexc, lena, lenb, nadexc, nbdexc, h2e, norb,
         pointer)
    return out


def _apply_array12_lowfillingab(coeff, alpha_array, beta_array,
                                nalpha, nbeta, intermediate):
    dtype = numpy.complex128
    assert(intermediate.dtype == dtype)
    assert(coeff.dtype == dtype)
    func = lib_fqe.apply_array12_lowfillingab
    func.argtypes = [
        ndpointer(
            shape=coeff.shape,
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=alpha_array.shape,
            dtype=numpy.int32,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=beta_array.shape,
            dtype=numpy.int32,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        ndpointer(
            shape=intermediate.shape,
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        )
    ]
    norb, na1, _ = alpha_array.shape
    nb1 = beta_array.shape[1]
    nca = coeff.shape[0]
    ncb = coeff.shape[1]
    nia = intermediate.shape[2]
    nib = intermediate.shape[3]
    func(coeff, alpha_array, beta_array,
         nalpha, nbeta, na1, nb1, nca, ncb, nia, nib, norb, intermediate)


def _apply_array12_lowfillingab2(alpha_array, beta_array,
                                 nalpha, nbeta,
                                 intermediate, out):
    dtype = numpy.complex128
    assert(intermediate.dtype == dtype)
    func = lib_fqe.apply_array12_lowfillingab2
    func.argtypes = [
        ndpointer(
            shape=intermediate.shape,
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=alpha_array.shape,
            dtype=numpy.int32,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=beta_array.shape,
            dtype=numpy.int32,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        ndpointer(
            shape=out.shape,
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        )
    ]
    norb, na1, _ = alpha_array.shape
    nb1 = beta_array.shape[1]
    nia = intermediate.shape[2]
    nib = intermediate.shape[3]
    noa = out.shape[0]
    nob = out.shape[1]
    func(intermediate, alpha_array, beta_array,
         nalpha, nbeta, na1, nb1, nia, nib, noa, nob, norb, out)


def _apply_array12_lowfillingaa(coeff, alpha_array, intermediate, alpha=True):
    func = lib_fqe.apply_array12_lowfillingaa
    func.argtypes = [
        ndpointer(
            shape=coeff.shape,
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=alpha_array.shape,
            dtype=numpy.int32,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        c_bool,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        ndpointer(
            shape=intermediate.shape,
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        )
    ]

    nlt, na, _ = alpha_array.shape
    ni1, ni2, ni3 = intermediate.shape
    nc1, nc2 = coeff.shape
    func(coeff, alpha_array, alpha, nlt, na, ni1, ni2, ni3, nc1, nc2, intermediate)
    #for ijn in range(nlt):
    #    for k in range(na):
    #        source = alpha_array[ijn, k, 0]
    #        target = alpha_array[ijn, k, 1]
    #        parity = alpha_array[ijn, k, 2]
    #        if alpha:
    #            work = coeff[source, :] * parity
    #            intermediate[ijn, target, :] += work
    #            pass
    #        else:
    #            work = coeff[:, source] * parity
    #            intermediate[ijn, :, target] += work


def _apply_array12_lowfillingaa2(intermediate, alpha_array, out, alpha=True):
    func = lib_fqe.apply_array12_lowfillingaa2
    func.argtypes = [
        ndpointer(
            shape=intermediate.shape,
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=alpha_array.shape,
            dtype=numpy.int32,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        c_bool,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        ndpointer(
            shape=out.shape,
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        )
    ]

    nlt, na, _ = alpha_array.shape
    ni1, ni2, ni3 = intermediate.shape
    no1, no2 = out.shape
    func(intermediate, alpha_array, alpha, nlt, na, ni1, ni2, ni3, no1, no2, out)
    #for ijn in range(nlt):
    #    for k in range(na):
    #        source = alpha_array[ijn, k, 0]
    #        target = alpha_array[ijn, k, 1]
    #        parity = alpha_array[ijn, k, 2]
    #        if alpha:
    #            out[source, :] -= intermediate[ijn, target, :] * parity
    #        else:
    #            out[:, source] -= intermediate[ijn, :, target] * parity


def _make_Hcomp(norb, nlt, h2e, h2ecomp):
    dtype = numpy.complex128
    assert(h2e.dtype == dtype)
    assert(h2ecomp.dtype == dtype)
    func = lib_fqe.make_Hcomp
    func.argtypes = [
        c_int,
        c_int,
        ndpointer(
            shape=h2e.shape,
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=h2ecomp.shape,
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        )
    ]
    func(norb, nlt, h2e, h2ecomp)


def _apply_individual_nbody1_accumulate(coeff, ocoeff, icoeff, amap,
                                        btarget, bsource, bparity):
    dtype = numpy.complex128
    assert(ocoeff.dtype == dtype)
    n = amap.shape[0]
    nao = ocoeff.shape[0]
    nbo = ocoeff.shape[1]
    nai = icoeff.shape[0]
    nbi = icoeff.shape[1]
    nt = btarget.shape[0]
    #assert(nb == ocoeff.shape[1])

    func = lib_fqe.apply_individual_nbody1_accumulate
    func.argtypes = [
        c_double_complex,
        ndpointer(
            shape=ocoeff.shape,
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=icoeff.shape,
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        ndpointer(
            shape=amap.shape,
            dtype=numpy.int64,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=btarget.shape,
            dtype=numpy.int64,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=bsource.shape,
            dtype=numpy.int64,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=bparity.shape,
            dtype=numpy.int64,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
    ]
    cc = c_double_complex(coeff.real, coeff.imag)
    func(cc, ocoeff, icoeff, n, nao, nbo, nai, nbi, nt, amap, btarget, bsource, bparity)


def _prepare_cirq_from_to_metadata(fqedata: 'FqeData',
                                   binarycode: Optional['BinaryCode']
                                   ) -> Tuple(numpy.ndarray, numpy.ndarray,
                                              numpy.ndarray, numpy.ndarray):
    """Generates some metadata for the cirq-fqe conversion.

    Args:
        fqedata(fqe_data.FqeData) - an FqeData to fill or read from.

        binarycode (Optional[openfermion.ops.BinaryCode]) - binary code to \
            encode the fermions to the qbit bosons. If None given,
            Jordan-Wigner transform is assumed.

    Returns:
        aswaps (numpy.ndarray) - for each alpha state, aswaps[state][i] gives \
            the number of electrons in orbitals j > i
        boccs (numpy.ndarray) - the occupied openfermion orbitals for each \
            beta state
        cirq_aid (numpy.ndarray) - For each alpha-state the corresponding part \
            of the cirq index.
        cirq_bid (numpy.ndarray) - For each beta-state the corresponding part \
            of the cirq index.

        the cirq_id for (alpha_state, beta_state) is then given by \
        cirq_aid[beta_state] XOR cirq_bid[alpha_state]. This can be done since \
        the encoder is a linear map modulo 2.
    """

    norb = fqedata.norb()
    # Get the alpha and beta dets
    alphadets = fqedata._core.string_alpha_all()
    betadets = fqedata._core.string_beta_all()

    #### These are for the conversion between fqe and openfermion
    # The occupied openfermion orbitals for each alpha state
    aoccs = [[up_index(x) for x in integer_index(astr)] for astr in alphadets]
    # For each alpha state, aswaps[state][i] gives the number of electrons in
    # orbitals j > i
    aswaps = numpy.array([[sum(ii > x for ii in aocc) for x in range(2*norb)]
                          for aocc in aoccs], dtype=numpy.int32)

    # The occupied openfermion orbitals for each beta state
    boccs = numpy.array([[down_index(x) for x in integer_index(bstr)]
                         for bstr in betadets], dtype=numpy.int32)

    nqubit = norb * 2
    # Since cirq starts counting from the leftmost bit in a bitstring
    pow_of_two = 2 ** (nqubit - numpy.arange(nqubit, dtype=numpy.int64) - 1)
    if binarycode is None:
        cirq_aid = numpy.array([pow_of_two[aocc].sum() for aocc in aoccs])
        cirq_bid = numpy.array([pow_of_two[bocc].sum() for bocc in boccs])
    else:
        def occ_to_cirq_ids(occs):
            cirq_ids = numpy.zeros(len(occs), dtype=numpy.int64)
            for ii, occ in enumerate(occs):
                of_state = numpy.zeros(nqubit, dtype=int)
                of_state[occ] = 1
                # Encode the occupation state to the qbit spin state
                cirq_state = numpy.mod(binarycode.encoder.dot(of_state), 2)
                cirq_ids[ii] = numpy.dot(pow_of_two, cirq_state)
            return cirq_ids

        cirq_aid = occ_to_cirq_ids(aoccs)
        cirq_bid = occ_to_cirq_ids(boccs)
    return aswaps, boccs, cirq_aid, cirq_bid


def _to_cirq(fqedata : 'FqeData', cwfn: numpy.ndarray,
             binarycode: Optional['BinaryCode'] = None) -> None:
    """Interoperability between cirq and the openfermion-fqe.  This takes an
    FqeData and fills a cirq compatible wavefunction

    Args:
        fqedata(fqe_data.FqeData) - an FqeData to fill the cirq wavefunction \
            with.

        cwfn (numpy.array(numpy.dtype=complex64)) - a cirq state to fill

        binarycode (Optional[openfermion.ops.BinaryCode]) - binary code to \
            encode the fermions to the qbit bosons. If None given,
            Jordan-Wigner transform is assumed.

    Returns:
        nothing - mutates cwfn in place
    """
    # Preparing metadata
    aswaps, boccs, cirq_aid, cirq_bid = \
        _prepare_cirq_from_to_metadata(fqedata, binarycode)

    func = lib_fqe.to_cirq

    norbs = fqedata.norb()
    nbeta = fqedata.nbeta()
    lena = fqedata.lena()
    lenb = fqedata.lenb()

    func.argtypes = [
        ndpointer(
            shape=(2 ** (2 * norbs),),
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(lena, lenb),
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        c_int,
        c_int,
        ndpointer(
            shape=(lena,),
            dtype=numpy.int64,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(lenb,),
            dtype=numpy.int64,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(lena, norbs * 2),
            dtype=numpy.int32,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(lenb, nbeta),
            dtype=numpy.int32,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        c_int,
        c_int
    ]

    func(cwfn, fqedata.coeff, lena, lenb, cirq_aid, cirq_bid, aswaps, boccs,
         nbeta, norbs)


def _from_cirq(fqedata : 'FqeData', cwfn: numpy.ndarray,
               binarycode: Optional['BinaryCode'] = None) -> None:
    """For the given FqeData structure, find the projection onto the cirq
    wavefunction and set the coefficients to the proper value.

    Args:
        fqedata (fqe_data.FqeData) - an FqeData to fill from the cirq \
            wavefunction

        cwfn (numpy.array(numpy.dtype=complex64)) - a cirq state to full in \
            with

        binarycode (Optional[openfermion.ops.BinaryCode]) - binary code to \
            encode the fermions to the qbit bosons. If None given,
            Jordan-Wigner transform is assumed.

    Returns:
        nothing - mutates fqedata in place
    """
    aswaps, boccs, cirq_aid, cirq_bid = \
        _prepare_cirq_from_to_metadata(fqedata, binarycode)

    func = lib_fqe.from_cirq

    norbs = fqedata.norb()
    nbeta = fqedata.nbeta()
    lena = fqedata.lena()
    lenb = fqedata.lenb()

    func.argtypes = [
        ndpointer(
            shape=(2 ** (2 * norbs),),
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(lena, lenb),
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        c_int,
        c_int,
        ndpointer(
            shape=(lena,),
            dtype=numpy.int64,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(lenb,),
            dtype=numpy.int64,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(lena, norbs * 2),
            dtype=numpy.int32,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(lenb, nbeta),
            dtype=numpy.int32,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        c_int,
        c_int
    ]

    func(cwfn, fqedata.coeff, lena, lenb, cirq_aid, cirq_bid, aswaps, boccs,
         nbeta, norbs)

def _sparse_scale(xi, yi, factor, data):
    dtype = numpy.complex128
    assert(data.dtype == dtype)
    fac = dtype(factor)
    ni1 = xi.shape[0]
    ni2 = xi.shape[1]
    ni = xi.size
    nd1 = data.shape[0]
    nd2 = data.shape[1]
    func = lib_fqe.sparse_scale
    func.argtypes = [
        ndpointer(
            shape=(ni1, ni2),
            dtype=numpy.int64,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(ni1, ni2),
            dtype=numpy.int64,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        c_double_complex,
        c_int,
        c_int,
        c_int,
        ndpointer(
            shape=(nd1, nd2),
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        )
    ]
    cfac = c_double_complex(fac.real, fac.imag)
    func(xi, yi, cfac, ni, nd1, nd2, data)


def _apply_diagonal_inplace(data, aarray, barray, astrings, bstrings):
    norb = aarray.size
    lena = astrings.size
    lenb = bstrings.size
    func1 = lib_fqe.apply_diagonal_inplace_real
    func1.argtypes = [
        ndpointer(
            shape=(lena, lenb),
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(norb,),
            dtype=numpy.float64,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(norb,),
            dtype=numpy.float64,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(lena,),
            dtype=numpy.uint64,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(lenb,),
            dtype=numpy.uint64,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        c_int,
        c_int
    ]
    func2 = lib_fqe.apply_diagonal_inplace
    func2.argtypes = [
        ndpointer(
            shape=(lena, lenb),
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(norb,),
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(norb,),
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(lena,),
            dtype=numpy.uint64,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(lenb,),
            dtype=numpy.uint64,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        c_int,
        c_int
    ]
    if aarray.dtype == numpy.float64:
        assert barray.dtype == numpy.float64
        func1(data, aarray, barray, astrings, bstrings, lena, lenb)
    else:
        func2(data, aarray, barray, astrings, bstrings, lena, lenb)


def _evolve_diagonal_inplace(data, aarray, barray, astrings, bstrings):
    norb = aarray.size
    lena = astrings.size
    lenb = bstrings.size
    func1 = lib_fqe.evolve_diagonal_inplace_real
    func1.argtypes = [
        ndpointer(
            shape=(lena, lenb),
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(norb,),
            dtype=numpy.float64,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(norb,),
            dtype=numpy.float64,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(lena,),
            dtype=numpy.uint64,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(lenb,),
            dtype=numpy.uint64,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        c_int,
        c_int
    ]
    func2 = lib_fqe.evolve_diagonal_inplace
    func2.argtypes = [
        ndpointer(
            shape=(lena, lenb),
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(norb,),
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(norb,),
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(lena,),
            dtype=numpy.uint64,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(lenb,),
            dtype=numpy.uint64,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        c_int,
        c_int
    ]
    if data.dtype == numpy.float64:
        func1(data, aarray, barray, astrings, bstrings, lena, lenb)
    else:
        assert data.dtype == numpy.complex128
        func2(data, aarray, barray, astrings, bstrings, lena, lenb)


def _evaluate_map_each(out: Nparray, strings: Nparray,
                       length: int, pmask :int, hmask: int):

    func = lib_fqe.evaluate_map_each
    func.argtypes = [
        ndpointer(
            shape=(length,),
            dtype=numpy.int64,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(length,),
            dtype=numpy.uint64,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        c_int,
        c_int,
        c_int
    ]
    return func(out, strings, length, pmask, hmask)


def _calculate_dvec1(alpha_array: Nparray,
                     beta_array: Nparray,
                     norb: int,
                     nalpha: int,
                     nbeta: int,
                     coeff: Nparray,
                     dvec: Nparray):
    func = lib_fqe.calculate_dvec1
    func.argtypes = [
        ndpointer(
            shape=alpha_array.shape,
            dtype=numpy.int32,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=beta_array.shape,
            dtype=numpy.int32,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        ndpointer(
            shape=coeff.shape,
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=dvec.shape,
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        )
    ]
    na = alpha_array.shape[1]
    nb = beta_array.shape[1]
    nc1, nc2 = coeff.shape
    nd1, nd2, nd3, nd4 = dvec.shape
    func(alpha_array, beta_array, norb, nalpha, nbeta,
         na, nb, nc1, nc2, nd1, nd2, nd3, nd4, coeff, dvec)


def _calculate_dvec2(alpha_array: Nparray,
                     beta_array: Nparray,
                     norb: int,
                     nalpha: int,
                     nbeta: int,
                     coeff: Nparray,
                     dvec: Nparray):
    func = lib_fqe.calculate_dvec2
    func.argtypes = [
        ndpointer(
            shape=alpha_array.shape,
            dtype=numpy.int32,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=beta_array.shape,
            dtype=numpy.int32,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        ndpointer(
            shape=coeff.shape,
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=dvec.shape,
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        )
    ]
    na = alpha_array.shape[1]
    nb = beta_array.shape[1]
    nc1, nc2 = coeff.shape
    nd1, nd2, nd3, nd4 = dvec.shape
    func(alpha_array, beta_array, norb, nalpha, nbeta,
         na, nb, nc1, nc2, nd1, nd2, nd3, nd4, coeff, dvec)


def _calculate_coeff1(alpha_array: Nparray,
                      beta_array: Nparray,
                      norb: int,
                      i: int,
                      j: int,
                      nalpha: int,
                      nbeta: int,
                      dvec: Nparray,
                      out: Nparray):
    func = lib_fqe.calculate_coeff1
    func.argtypes = [
        ndpointer(
            shape=alpha_array.shape,
            dtype=numpy.int32,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=beta_array.shape,
            dtype=numpy.int32,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        ndpointer(
            shape=dvec.shape,
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=out.shape,
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        )
    ]
    na = alpha_array.shape[1]
    nb = beta_array.shape[1]
    nd1, nd2, nd3, nd4 = dvec.shape
    no1, no2 = out.shape
    func(alpha_array, beta_array, norb, i, j, nalpha, nbeta,
         na, nb, nd1, nd2, nd3, nd4, no1, no2, dvec, out)
    #for k in range(alpha_array.shape[1]):
    #    sourcea = alpha_array[j, k, 0]
    #    targeta = alpha_array[j, k, 1]
    #    paritya = alpha_array[j, k, 2]
    #    paritya *= (-1)**nalpha
    #    for l in range(beta_array.shape[1]):
    #        sourceb = beta_array[i, l, 0]
    #        targetb = beta_array[i, l, 1]
    #        parityb = beta_array[i, l, 2]
    #        work = dvec[i + norb, j, targeta, targetb]
    #        out[sourcea,
    #             sourceb] += work * paritya * parityb

def _calculate_coeff2(alpha_array: Nparray,
                      beta_array: Nparray,
                      norb: int,
                      i: int,
                      j: int,
                      nalpha: int,
                      nbeta: int,
                      dvec: Nparray,
                      out: Nparray):
    func = lib_fqe.calculate_coeff2
    func.argtypes = [
        ndpointer(
            shape=alpha_array.shape,
            dtype=numpy.int32,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=beta_array.shape,
            dtype=numpy.int32,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        ndpointer(
            shape=dvec.shape,
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=out.shape,
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        )
    ]
    na = alpha_array.shape[1]
    nb = beta_array.shape[1]
    nd1, nd2, nd3, nd4 = dvec.shape
    no1, no2 = out.shape
    func(alpha_array, beta_array, norb, i, j, nalpha, nbeta,
         na, nb, nd1, nd2, nd3, nd4, no1, no2, dvec, out)
    #for k in range(alpha_array.shape[1]):
    #    sourcea = alpha_array[i, k, 0]
    #    targeta = alpha_array[i, k, 1]
    #    paritya = alpha_array[i, k, 2]
    #    paritya *= (-1)**(nalpha - 1)
    #    for l in range(beta_array.shape[1]):
    #        sourceb = beta_array[j, l, 0]
    #        targetb = beta_array[j, l, 1]
    #        parityb = beta_array[j, l, 2]
    #        work = dvec[i, j + norb, targeta, targetb]
    #        out[sourcea,
    #             sourceb] += work * paritya * parityb

def _calculate_dvec1_j(alpha_array: Nparray,
                       beta_array: Nparray,
                       norb: int,
                       j: int,
                       nalpha: int,
                       nbeta: int,
                       coeff: Nparray,
                       dvec: Nparray):
    func = lib_fqe.calculate_dvec1_j
    func.argtypes = [
        ndpointer(
            shape=alpha_array.shape,
            dtype=numpy.int32,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=beta_array.shape,
            dtype=numpy.int32,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        ndpointer(
            shape=coeff.shape,
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=dvec.shape,
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        )
    ]
    na = alpha_array.shape[1]
    nb = beta_array.shape[1]
    nc1, nc2 = coeff.shape
    nd1, nd2, nd3 = dvec.shape
    func(alpha_array, beta_array, norb, j, nalpha, nbeta,
         na, nb, nc1, nc2, nd1, nd2, nd3, coeff, dvec)
    #for k in range(alpha_array.shape[1]):
    #    sourcea = alpha_array[j, k, 0]
    #    targeta = alpha_array[j, k, 1]
    #    paritya = alpha_array[j, k, 2]
    #    paritya *= (-1)**(nalpha - 1)
    #    for l in range(beta_array.shape[1]):
    #        sourceb = beta_array[i, l, 0]
    #        targetb = beta_array[i, l, 1]
    #        parityb = beta_array[i, l, 2]
    #        work = coeff[sourcea, sourceb]
    #        dvec[i + norb, targeta,
    #              targetb] += work * paritya * parityb


def _calculate_dvec2_j(alpha_array: Nparray,
                       beta_array: Nparray,
                       norb: int,
                       j: int,
                       nalpha: int,
                       nbeta: int,
                       coeff: Nparray,
                       dvec: Nparray):
    func = lib_fqe.calculate_dvec2_j
    func.argtypes = [
        ndpointer(
            shape=alpha_array.shape,
            dtype=numpy.int32,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=beta_array.shape,
            dtype=numpy.int32,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        ndpointer(
            shape=coeff.shape,
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=dvec.shape,
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        )
    ]
    na = alpha_array.shape[1]
    nb = beta_array.shape[1]
    nc1, nc2 = coeff.shape
    nd1, nd2, nd3 = dvec.shape
    func(alpha_array, beta_array, norb, j, nalpha, nbeta,
         na, nb, nc1, nc2, nd1, nd2, nd3, coeff, dvec)
    #for k in range(alpha_array.shape[1]):
    #    sourcea = alpha_array[i, k, 0]
    #    targeta = alpha_array[i, k, 1]
    #    paritya = alpha_array[i, k, 2]
    #    paritya *= (-1)**(nalpha)
    #    for l in range(beta_array.shape[1]):
    #        sourceb = beta_array[j - norb, l, 0]
    #        targetb = beta_array[j - norb, l, 1]
    #        parityb = beta_array[j - norb, l, 2]
    #        work = coeff[sourcea, sourceb]
    #        dvec[i, targeta,
    #              targetb] += work * paritya * parityb

def _make_nh123(norb: int,
                h4e: Nparray,
                nh1e: Nparray,
                nh2e: Nparray,
                nh3e: Nparray):

    dtype = h4e.dtype
    if dtype == numpy.float64:
        func = lib_fqe.make_nh123_real
        func.argtypes = [
            c_int,
            ndpointer(
                shape=h4e.shape,
                dtype=numpy.float64,
                flags=('C_CONTIGUOUS', 'ALIGNED')
            ),
            ndpointer(
                shape=nh1e.shape,
                dtype=numpy.float64,
                flags=('C_CONTIGUOUS', 'ALIGNED')
            ),
            ndpointer(
                shape=nh2e.shape,
                dtype=numpy.float64,
                flags=('C_CONTIGUOUS', 'ALIGNED')
            ),
            ndpointer(
                shape=nh3e.shape,
                dtype=numpy.float64,
                flags=('C_CONTIGUOUS', 'ALIGNED')
            )
        ]
        func(norb, h4e, nh1e, nh2e, nh3e)
    else:
        func = lib_fqe.make_nh123
        func.argtypes = [
            c_int,
            ndpointer(
                shape=h4e.shape,
                dtype=numpy.complex128,
                flags=('C_CONTIGUOUS', 'ALIGNED')
            ),
            ndpointer(
                shape=nh1e.shape,
                dtype=numpy.complex128,
                flags=('C_CONTIGUOUS', 'ALIGNED')
            ),
            ndpointer(
                shape=nh2e.shape,
                dtype=numpy.complex128,
                flags=('C_CONTIGUOUS', 'ALIGNED')
            ),
            ndpointer(
                shape=nh3e.shape,
                dtype=numpy.complex128,
                flags=('C_CONTIGUOUS', 'ALIGNED')
            )
        ]
        func(norb, h4e, nh1e, nh2e, nh3e)
