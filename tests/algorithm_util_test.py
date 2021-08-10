#   Copyright 2020 Google LLC

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
"""Tests for utility functions for algorithms """

import numpy

from fqe.wavefunction import Wavefunction
from tests.unittest_data import build_wfn
from fqe.algorithm.algorithm_util import valdemaro_reconstruction


def test_valdemaro_reconstruction():
    nele = 4
    norb = 3
    s_z = 0
    wfn = Wavefunction(param=[[nele, s_z, norb]])
    work, energy = build_wfn.restricted_wfn_energy()
    wfn.set_wfn(strategy='from_data', raw_data={(nele, s_z): work})

    _, tpdm = wfn.sector((nele, s_z)).get_openfermion_rdms()
    rdm3 = 6 * valdemaro_reconstruction(tpdm / 2, nele)
    rdm3_reference = wfn.sector((nele, s_z)).get_three_pdm()

    assert numpy.allclose(rdm3, rdm3_reference, rtol=1e-3, atol=4e-3)
