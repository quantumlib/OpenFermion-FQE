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
"""This file implements wrappers to some of the functions in fqe.bitstring
"""

from ctypes import c_int, c_bool, c_ulonglong, POINTER

import numpy
from numpy.ctypeslib import ndpointer

from fqe.lib import lib_fqe


def _count_bits(string: int):
    func = lib_fqe.count_bits
    func.argtypes = [c_ulonglong]
    return func(c_ulonglong(string))


def _get_occupation(string: int):
    func = lib_fqe.get_occupation
    out = numpy.zeros((64,), dtype=numpy.int32)
    func.argtypes = [
        ndpointer(dtype=numpy.int32, flags=('C_CONTIGUOUS', 'ALIGNED')),
        c_ulonglong
    ]
    count = func(out, string)
    return out[:count]


def _lexicographic_bitstring_generator(out, norb: int, nele: int):
    func = lib_fqe.lexicographic_bitstring_generator
    func.argtypes = [
        ndpointer(dtype=numpy.uint64, flags=('C_CONTIGUOUS', 'ALIGNED')), c_int,
        c_int
    ]
    func(out, norb, nele)
