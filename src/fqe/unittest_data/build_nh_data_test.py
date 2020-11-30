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
"""Tests for NH data."""

# pylint: disable=too-many-locals

import numpy as np
from scipy.special import binom

import fqe
from fqe.hamiltonians import general_hamiltonian
from fqe.unittest_data.build_nh_data import build_nh_data


def test_nh_energy():
    """Checks total relativistic energy with NH."""
    eref = -57.681266930627
    norb = 6
    nele, h1e, h2e = build_nh_data()

    elec_hamil = general_hamiltonian.General((h1e, h2e))
    maxb = min(norb, nele)
    minb = nele - maxb
    ndim = int(binom(norb * 2, nele))
    hci = np.zeros((ndim, ndim), dtype=np.complex128)

    for i in range(0, ndim):
        wfn = fqe.get_number_conserving_wavefunction(nele, norb)
        cnt = 0
        for nbeta in range(minb, maxb + 1):
            coeff = wfn.get_coeff((nele, nele - 2 * nbeta))
            size = coeff.size
            if cnt <= i < cnt + size:
                coeff.flat[i - cnt] = 1.0
            cnt += size

        result = wfn.apply(elec_hamil)

        cnt = 0
        for nbeta in range(minb, maxb + 1):
            coeff = result.get_coeff((nele, nele - 2 * nbeta))
            for j in range(coeff.size):
                hci[cnt + j, i] = coeff.flat[j]
            cnt += coeff.size

    assert np.std(hci - hci.T.conj()) < 1.0e-8

    eigenvals, eigenvecs = np.linalg.eigh(hci)

    assert np.isclose(eref, eigenvals[0], atol=1e-8)

    orig = eigenvecs[:, 0]
    wfn = fqe.get_number_conserving_wavefunction(nele, norb)
    cnt = 0
    for nbeta in range(minb, maxb + 1):
        nalpha = nele - nbeta
        vdata = np.zeros(
            (int(binom(norb, nalpha)), int(binom(norb, nbeta))),
            dtype=np.complex128,
        )
        for i in range(vdata.size):
            vdata.flat[i] = orig[cnt + i]
        wfn._civec[(nele, nalpha - nbeta)].coeff += vdata
        cnt += vdata.size

    hwfn = wfn.apply(elec_hamil)
    ecalc = fqe.vdot(wfn, hwfn).real
    assert np.isclose(eref, ecalc, atol=1e-8)
