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

from ctypes import c_int, c_bool, POINTER, c_int64
from typing import List

import numpy
from numpy.ctypeslib import ndpointer
from numpy import ndarray as Nparray

from fqe.lib import lib_fqe

from scipy.linalg.cython_blas cimport zaxpy
from libc.stdint cimport uintptr_t
include "blas_helpers.pxi"

def _lm_apply_array1(coeff, h1e, dexc, lena, lenb, norb, alpha=True, out=None):
    func = lib_fqe.lm_apply_array1
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
    blas_functions.zaxpy = zaxpy
    cdef object pointer = <uintptr_t>&blas_functions

    func(coeff, out, dexc, lena, lenb, ndexc, h1e, norb, alpha, pointer)
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
    blas_functions.zaxpy = zaxpy
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
    blas_functions.zaxpy = zaxpy
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
    blas_functions.zaxpy = zaxpy
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
    blas_functions.zaxpy = zaxpy
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
    blas_functions.zaxpy = zaxpy
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
            dtype=numpy.uint32,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(lenb,),
            dtype=numpy.uint32,
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
    blas_functions.zaxpy = zaxpy
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
    blas_functions.zaxpy = zaxpy
    cdef object pointer = <uintptr_t>&blas_functions
    func(coeff, out, adexc, bdexc, lena, lenb, nadexc, nbdexc, h2e, norb,
         pointer)
    return out
