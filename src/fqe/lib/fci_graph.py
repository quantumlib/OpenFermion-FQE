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

from ctypes import c_int, c_bool, c_ulonglong, POINTER
from typing import TYPE_CHECKING, Dict, Tuple

import numpy
from numpy.ctypeslib import ndpointer
import scipy
from scipy import special

from fqe.lib import lib_fqe

if TYPE_CHECKING:
    from numpy import ndarray as Nparray


def _calculate_Z_matrix(out, norb, nele):
    func = lib_fqe.calculate_Z_matrix
    func.argtypes = [
        ndpointer(shape=(nele, norb),
                  dtype=numpy.int32,
                  flags=('C_CONTIGUOUS', 'ALIGNED')), c_int, c_int
    ]
    func(out, norb, nele)


def _map_deexc(out, inp, index, idx):
    func = lib_fqe.map_deexc
    lena = out.shape[0]
    lk = out.shape[1]
    size = inp.shape[0]
    func.argtypes = [
        ndpointer(shape=(lena, lk, 3),
                  dtype=numpy.int32,
                  flags=('C_CONTIGUOUS', 'ALIGNED')),
        ndpointer(shape=(size, 3),
                  dtype=numpy.int32,
                  flags=('C_CONTIGUOUS', 'ALIGNED')), c_int, c_int,
        ndpointer(shape=(lena,),
                  dtype=numpy.uint32,
                  flags=('C_CONTIGUOUS', 'ALIGNED')), c_int
    ]
    func(out, inp, lk, size, index, idx)


def _build_mapping_strings(strings, zmat, nele: int, norb: int):
    func = lib_fqe.build_mapping_strings
    c_ptr_map = POINTER(c_int * 3)

    func.argtypes = [
        POINTER(c_ptr_map),
        ndpointer(shape=(norb, norb),
                  dtype=numpy.int32,
                  flags=('C_CONTIGUOUS', 'ALIGNED')),
        ndpointer(shape=(norb**2, 2),
                  dtype=numpy.int32,
                  flags=('C_CONTIGUOUS', 'ALIGNED')), c_int,
        ndpointer(dtype=numpy.uint64, flags=('C_CONTIGUOUS', 'ALIGNED')), c_int,
        c_bool,
        ndpointer(shape=(nele, norb),
                  dtype=numpy.int32,
                  flags=('C_CONTIGUOUS', 'ALIGNED')), c_int
    ]

    maplengths = numpy.zeros((norb, norb), dtype=numpy.int32)
    exc_dexc = numpy.asarray(
        [(i, j) for i in range(norb) for j in range(norb)],
        dtype=numpy.int32).reshape(
            -1, 2)  # reshape needed if exc_dexc is zero-size array

    # Count maplengths
    func(None, maplengths, exc_dexc, len(exc_dexc), strings, len(strings), True,
         zmat, norb)

    # allocate maps
    out = {
        tuple(ed): numpy.zeros((ml, 3), dtype=numpy.int32)
        for ed, ml in zip(exc_dexc, maplengths.ravel())
    }
    out_data = [out[tuple(ed)].ctypes.data_as(c_ptr_map) for ed in exc_dexc]
    # Fill in maps
    func((c_ptr_map * len(exc_dexc))(*out_data), maplengths, exc_dexc,
         len(exc_dexc), strings, len(strings), False, zmat, norb)

    return out


def _calculate_string_address(zmat, nele: int, norb: int, strings: 'Nparray'):
    length = strings.size
    out = numpy.zeros((length,), dtype=numpy.uint64)
    func = lib_fqe.calculate_string_address
    func.argtypes = [
        ndpointer(shape=(length,),
                  dtype=numpy.uint64,
                  flags=('C_CONTIGUOUS', 'ALIGNED')),
        ndpointer(shape=(length,),
                  dtype=numpy.uint64,
                  flags=('C_CONTIGUOUS', 'ALIGNED')), c_int,
        ndpointer(shape=(nele, norb),
                  dtype=numpy.int32,
                  flags=('C_CONTIGUOUS', 'ALIGNED')), c_int
    ]
    func(out, strings, length, zmat, norb)
    return out


def _c_map_to_deexc_alpha_icol(exc: 'Nparray', diag: 'Nparray',
                               index: 'Nparray', strings: 'Nparray', norb: int,
                               mappings: Dict[Tuple[int, int], 'Nparray']):
    func = lib_fqe.map_to_deexc_alpha_icol
    c_ptr_map = POINTER(c_int * 3)
    nmaps = len(mappings)
    if nmaps != norb**2:
        raise ValueError('number of mappings passed should be norb ** 2')

    list_map = [mappings[(i, j)] for j in range(norb) for i in range(norb)]
    exc0 = exc.shape[1]
    exc1 = exc.shape[2]
    ldiag = diag.shape[1]

    func.argtypes = [
        POINTER(c_ptr_map),
        ndpointer(shape=(nmaps,),
                  dtype=numpy.int32,
                  flags=('C_CONTIGUOUS', 'ALIGNED')),
        ndpointer(dtype=numpy.uint64, flags=('C_CONTIGUOUS', 'ALIGNED')), c_int,
        ndpointer(shape=(norb, exc0, exc1, 3),
                  dtype=numpy.int32,
                  flags=('C_CONTIGUOUS', 'ALIGNED')),
        ndpointer(shape=(norb, ldiag),
                  dtype=numpy.int32,
                  flags=('C_CONTIGUOUS', 'ALIGNED')),
        ndpointer(shape=(norb, exc0),
                  dtype=numpy.int32,
                  flags=('C_CONTIGUOUS', 'ALIGNED')), c_int, c_int, c_int, c_int
    ]
    func(
        (c_ptr_map * nmaps)(*[mp.ctypes.data_as(c_ptr_map) for mp in list_map]),
        numpy.array([len(x) for x in list_map], dtype=numpy.int32), strings,
        len(strings), exc, diag, index, norb, exc0, exc1, ldiag)


def _make_mapping_each(out: 'Nparray', strings: 'Nparray', length: int,
                       dag: 'Nparray', undag: 'Nparray'):
    func = lib_fqe.make_mapping_each
    func.argtypes = [
        ndpointer(shape=(length, 3),
                  dtype=numpy.uint64,
                  flags=('C_CONTIGUOUS', 'ALIGNED')),
        ndpointer(shape=(length,),
                  dtype=numpy.uint64,
                  flags=('C_CONTIGUOUS', 'ALIGNED')), c_int,
        ndpointer(dtype=numpy.int32, flags=('C_CONTIGUOUS', 'ALIGNED')), c_int,
        ndpointer(dtype=numpy.int32, flags=('C_CONTIGUOUS', 'ALIGNED')), c_int
    ]
    return func(out, strings, length, dag, dag.size, undag, undag.size)


def _make_mapping_each_set(istrings: 'Nparray', dnv: int, norb: int, nele: int):
    nsize = int(special.binom(norb - dnv, nele - dnv))
    msize = int(special.binom(norb, dnv))
    length = istrings.size

    mapping_down = numpy.zeros((msize, nsize, 3), dtype=numpy.uint64)
    mapping_up = numpy.zeros((msize, nsize, 3), dtype=numpy.uint64)

    func = lib_fqe.make_mapping_each_set
    func.argtypes = [
        ndpointer(shape=(msize, nsize, 3),
                  dtype=numpy.uint64,
                  flags=('C_CONTIGUOUS', 'ALIGNED')),
        ndpointer(shape=(msize, nsize, 3),
                  dtype=numpy.uint64,
                  flags=('C_CONTIGUOUS', 'ALIGNED')),
        ndpointer(shape=(length,),
                  dtype=numpy.uint64,
                  flags=('C_CONTIGUOUS', 'ALIGNED')), c_int, c_int, c_int,
        c_int, c_int
    ]
    func(mapping_down, mapping_up, istrings, length, msize, nsize, dnv, norb)
    return mapping_down, mapping_up
