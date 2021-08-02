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
"""Tests for the calculation of properties
"""

import numpy
import pytest
from numpy import linalg
from scipy.special import binom

import fqe
from fqe.wavefunction import Wavefunction
from fqe.hamiltonians import restricted_hamiltonian
from fqe.fqe_ops.fqe_ops import (
    NumberOperator,
    S2Operator,
    SzOperator,
    TimeReversalOp,
)

from tests.unittest_data import build_lih_data, build_nh_data

def test_lih_energy():
    """Checking total energy with LiH
    """
    eref = -8.877719570384043
    norb = 6
    nalpha = 2
    nbeta = 2
    nele = nalpha + nbeta
    h1e, h2e, lih_ground = build_lih_data.build_lih_data('energy')

    elec_hamil = restricted_hamiltonian.RestrictedHamiltonian((h1e, h2e))
    wfn = Wavefunction([[nele, nalpha - nbeta, norb]])
    wfn.set_wfn(strategy='from_data',
                raw_data={(nele, nalpha - nbeta): lih_ground})

    ecalc = wfn.expectationValue(elec_hamil)
    assert round(abs(eref-ecalc), 8) == 0

def test_lih_dipole():
    """Calculate the LiH dipole
    """
    norb = 6
    nalpha = 2
    nbeta = 2
    nele = nalpha + nbeta
    au2debye = 2.5417464157449032

    dip_ref, dip_mat, lih_ground = build_lih_data.build_lih_data('dipole')

    wfn = Wavefunction([[nele, nalpha - nbeta, norb]])
    wfn.set_wfn(strategy='from_data',
                raw_data={(nele, nalpha - nbeta): lih_ground})

    hwfn_x = wfn._apply_array(tuple([dip_mat[0]]), e_0=0. + 0.j)
    hwfn_y = wfn._apply_array(tuple([dip_mat[1]]), e_0=0. + 0.j)
    hwfn_z = wfn._apply_array(tuple([dip_mat[2]]), e_0=0. + 0.j)
    calc_dip = numpy.array([fqe.vdot(wfn, hwfn_x).real, \
                          fqe.vdot(wfn, hwfn_y).real, \
                          fqe.vdot(wfn, hwfn_z).real])*au2debye
    for card in range(3):
        err = abs(calc_dip[card] - dip_ref[card])
        assert err < 1.e-5

def test_lih_ops():
    """Check the value of the operators on LiH
    """
    norb = 6
    nalpha = 2
    nbeta = 2
    nele = nalpha + nbeta

    _, _, lih_ground = build_lih_data.build_lih_data('energy')

    wfn = Wavefunction([[nele, nalpha - nbeta, norb]])
    wfn.set_wfn(strategy='from_data',
                raw_data={(nele, nalpha - nbeta): lih_ground})

    operator = S2Operator()
    assert round(abs(wfn.expectationValue(operator)-0. + 0.j), 7) == 0
    operator = SzOperator()
    assert round(abs(wfn.expectationValue(operator)-0. + 0.j), 7) == 0
    operator = TimeReversalOp()
    assert round(abs(wfn.expectationValue(operator)-1. + 0.j), 7) == 0
    operator = NumberOperator()
    assert round(abs(wfn.expectationValue(operator)-4. + 0.j), 7) == 0
    assert round(abs(wfn.expectationValue(operator, wfn)-4. + 0.j), 7) == 0
