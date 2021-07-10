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

from openfermion import up_index, down_index, BinaryCode
from fqe.bitstring import integer_index
from fqe.lib import lib_fqe


def detect_cirq_sectors(state: Nparray, thresh: float,
                        binarycode: Optional['BinaryCode'] = None
                        ) -> List[List[int]]:
    """Detects and returns the different sectors which have non-zero weight (up
    to the threshold) in the cirq state. This is optimized code for the
    Jordan-Wigner transformation.

    Args:
        state (numpy.array(dtype=numpy.complex128)) - a cirq wavefunction

        thresh (double) - set the limit at which a cirq element should be \
            considered zero and not make a contribution to the FQE wavefunction

        binarycode (Optional[openfermion.ops.BinaryCode]) - binary code to \
            encode the fermions to the qbit bosons. If None given,
            Jordan-Wigner transform is assumed.

    Returns:
        sectors (list[list[n, ms, norb]]) - the dectected sectors in the cirq \
        state.  The lists are comprised of

              p[0] (integer) - number of particles;
              p[1] (integer) - z component of spin angular momentum;
              p[2] (integer) - number of spatial orbitals
    """
    func = lib_fqe.detect_cirq_sectors
    nqubit = int(numpy.log2(state.size))
    norb = nqubit // 2

    nlena = 2 ** norb
    nlenb = 2 ** norb

    # occupations of all possible alpha and beta strings
    aoccs = [[up_index(x) for x in integer_index(astr)]
             for astr in range(nlena)]
    boccs = [[down_index(x) for x in integer_index(bstr)]
             for bstr in range(nlenb)]

    # Since cirq starts counting from the leftmost bit in a bitstring
    pow_of_two = 2 ** (nqubit - numpy.arange(nqubit, dtype=numpy.int64) - 1)
    if binarycode is None:
        # cirq index for each alpha or beta string
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

        # cirq index for each alpha or beta string
        cirq_aid = occ_to_cirq_ids(aoccs)
        cirq_bid = occ_to_cirq_ids(boccs)

    # Number of alpha or beta electrons for each alpha or beta string
    anumb = numpy.array([len(x) for x in aoccs], dtype=numpy.int32)
    bnumb = numpy.array([len(x) for x in boccs], dtype=numpy.int32)
    param = numpy.zeros((2 * norb + 1, 2 * norb + 1), dtype=numpy.int32)

    func.argtypes = [
        ndpointer(
            shape=(nlena * nlenb,),
            dtype=numpy.complex128,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        c_double,
        ndpointer(
            shape=param.shape,
            dtype=numpy.int32,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        c_int,
        c_int,
        c_int,
        ndpointer(
            shape=(nlena,),
            dtype=numpy.int64,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(nlenb,),
            dtype=numpy.int64,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(nlena,),
            dtype=numpy.int32,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        ),
        ndpointer(
            shape=(nlenb,),
            dtype=numpy.int32,
            flags=('C_CONTIGUOUS', 'ALIGNED')
        )
    ]

    func(state, thresh, param, norb, nlena, nlenb, cirq_aid, cirq_bid, anumb,
         bnumb)

    return [[pnum, sz - norb, norb] for pnum, sz in zip(*numpy.nonzero(param))]
