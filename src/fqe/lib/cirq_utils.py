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
"""This file implements wrappers to some of the functions in fqe.cirq_utils
"""

from typing import List, Optional
from ctypes import c_int, c_double

import numpy
from numpy.ctypeslib import ndpointer
from numpy import ndarray as Nparray

from fqe.lib import lib_fqe


def _detect_cirq_sectors(state: Nparray, thresh: float, param: Nparray,
                         norb: int, nlena: int, nlenb: int, cirq_aid: Nparray,
                         cirq_bid: Nparray, anumb: Nparray,
                         bnumb: Nparray) -> None:
    func = lib_fqe.detect_cirq_sectors

    func.argtypes = [
        ndpointer(shape=(nlena * nlenb,),
                  dtype=numpy.complex128,
                  flags=('C_CONTIGUOUS', 'ALIGNED')), c_double,
        ndpointer(shape=param.shape,
                  dtype=numpy.int32,
                  flags=('C_CONTIGUOUS', 'ALIGNED')), c_int, c_int, c_int,
        ndpointer(shape=(nlena,),
                  dtype=numpy.int64,
                  flags=('C_CONTIGUOUS', 'ALIGNED')),
        ndpointer(shape=(nlenb,),
                  dtype=numpy.int64,
                  flags=('C_CONTIGUOUS', 'ALIGNED')),
        ndpointer(shape=(nlena,),
                  dtype=numpy.int32,
                  flags=('C_CONTIGUOUS', 'ALIGNED')),
        ndpointer(shape=(nlenb,),
                  dtype=numpy.int32,
                  flags=('C_CONTIGUOUS', 'ALIGNED'))
    ]

    func(state, thresh, param, norb, nlena, nlenb, cirq_aid, cirq_bid, anumb,
         bnumb)
