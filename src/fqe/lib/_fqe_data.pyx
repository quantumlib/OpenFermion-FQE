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
from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy
from numpy.ctypeslib import ndpointer
from openfermion import up_index, down_index
from fqe.bitstring import integer_index

from scipy.linalg.cython_blas cimport zaxpy
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


def _apply_array12_lowfillingab(coeff, alpha_array, beta_array,
                                nalpha, nbeta, na1, nb1, norb,
                                intermediate):
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
    nca = coeff.shape[0]
    ncb = coeff.shape[1]
    nia = intermediate.shape[2]
    nib = intermediate.shape[3]
    func(coeff, alpha_array, beta_array,
         nalpha, nbeta, na1, nb1, nca, ncb, nia, nib, norb, intermediate)


def _apply_array12_lowfillingab2(alpha_array, beta_array,
                                 nalpha, nbeta, na1, nb1, norb,
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
    nia = intermediate.shape[2]
    nib = intermediate.shape[3]
    noa = out.shape[0]
    nob = out.shape[1]
    func(intermediate, alpha_array, beta_array,
         nalpha, nbeta, na1, nb1, nia, nib, noa, nob, norb, out)


def _apply_individual_nbody1(coeff, ocoeff, icoeff, amap,
                             btarget, bsource, bparity):
    dtype = numpy.complex128
    assert(ocoeff.dtype == dtype)
    assert(icoeff.dtype == dtype)
    aarray = numpy.asarray(amap, dtype=numpy.int32)
    n = aarray.shape[0]
    na = icoeff.shape[0]
    nb = icoeff.shape[1]
    nt = btarget.shape[0]
    assert(nb == ocoeff.shape[1])

    func = lib_fqe.apply_individual_nbody1
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
        ndpointer(
            shape=aarray.shape,
            dtype=numpy.int32,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=btarget.shape,
            dtype=numpy.int32,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=bsource.shape,
            dtype=numpy.int32,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=bparity.shape,
            dtype=numpy.int32,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
    ]
    cc = c_double_complex(coeff.real, coeff.imag)
    func(cc, ocoeff, icoeff, n, na, nb, nt, aarray, btarget, bsource, bparity)
    #for i in range(n):
    #    sourcea = aarray[i, 0]
    #    targeta = aarray[i, 1]
    #    paritya = aarray[i, 2]
    #    for j in range(nt):
    #        ocoeff[targeta, btarget[j]] = \
    #            coeff * paritya * numpy.multiply(
    #                icoeff[sourcea, bsource[j]], bparity[j])


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
            cirq_ids = numpy.zeros(len(aoccs), dtype=numpy.int64)
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
