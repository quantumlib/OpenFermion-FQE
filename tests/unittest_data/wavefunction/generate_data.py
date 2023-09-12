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
""" fci_graph unit tests: generate reference data
"""
import os
import numpy
import pickle
from fqe.wavefunction import Wavefunction
from openfermion import FermionOperator
from fqe.hamiltonians import sparse_hamiltonian, diagonal_hamiltonian
from fqe import get_restricted_hamiltonian
from fqe import get_gso_hamiltonian
from fqe import get_diagonalcoulomb_hamiltonian

datadir = os.path.join(os.path.dirname(os.path.abspath(__file__)))

spin_broken_cases = [
    ((4, -4, 6), (4, -2, 6), (4, 0, 6), (4, 2, 6), (4, 4, 6)),
    ((3, -3, 7), (3, -1, 7), (3, 1, 7), (3, 3, 7)),
]

particle_broken_cases = [
    ((0, 0, 4), (2, 0, 4), (4, 0, 4), (6, 0, 4), (8, 0, 4)),
    ((1, 1, 2), (3, 1, 2)),
]

conserving_cases = [((7, 1, 8),), ((8, 0, 6),), ((3, -3, 7),), ((2, 2, 6),)]

all_cases = spin_broken_cases + particle_broken_cases + conserving_cases


def generate_data(param):
    import sys
    sys.path.append(r'../')
    from build_hamiltonian import build_H1, build_H2

    nele, sz, norbs = tuple(set(x) for x in zip(*param))
    pseed = sum(nele) * 100 + sum(sz) * 10 + sum(norbs)
    rng = numpy.random.default_rng(pseed + 958850)
    assert len(norbs) == 1
    norbs = norbs.pop()
    number_conserving = len(nele) == 1
    spin_conserving = len(sz) == 1
    assert number_conserving or spin_conserving
    broken = None
    if not number_conserving:
        broken = ['number']
    if not spin_conserving:
        broken = ['spin']

    data = {}
    wfn = Wavefunction(param, broken=broken)
    for key, value in wfn._civec.items():
        cr = rng.uniform(-0.5, 0.5, size=value.coeff.shape)
        ci = rng.uniform(-0.5, 0.5, size=value.coeff.shape)
        value.coeff = cr + 1.j * ci
    wfn.normalize()

    data['wfn'] = wfn
    if not spin_conserving:
        data['number_sectors'] = wfn._number_sectors()

    # Sparse ham
    fop = FermionOperator('1^ 1', 1.5)
    hamil = sparse_hamiltonian.SparseHamiltonian(fop)
    hamil._conserve_number = number_conserving

    try:
        out = wfn.apply(hamil)
    except ValueError:
        # Don't generate an output
        out = None
    try:
        evolved = wfn.time_evolve(0.1, hamil)
    except ValueError:
        # Don't generate an output
        evolved = None

    data['apply_sparse'] = {
        'hamil': hamil,
        'wfn_out': out,
        'wfn_evolve': evolved
    }

    # Diagonal ham
    diag = rng.uniform(size=norbs * 2)
    hamil = diagonal_hamiltonian.Diagonal(diag)
    hamil._conserve_number = number_conserving
    try:
        out = wfn.apply(hamil)
    except ValueError:
        # Don't generate an output
        out = None
    try:
        evolved = wfn.time_evolve(0.1, hamil)
    except ValueError:
        # Don't generate an output
        evolved = None

    data['apply_diagonal'] = {
        'hamil': hamil,
        'wfn_out': out,
        'wfn_evolve': evolved
    }

    # array ham
    full = not (spin_conserving and number_conserving)
    h1e = build_H1(norbs, full=full) / 10
    h2e = build_H2(norbs, full=full) / 20
    e_0 = rng.uniform() + rng.uniform() * 1j
    if not full:
        hamil = get_restricted_hamiltonian((
            h1e,
            h2e,
        ), e_0=e_0)
    else:
        hamil = get_gso_hamiltonian((
            h1e,
            h2e,
        ), e_0=e_0)
    hamil._conserve_number = number_conserving
    try:
        out = wfn.apply(hamil)
    except ValueError:
        # Don't generate an output
        out = None
    try:
        evolved = wfn.time_evolve(0.1, hamil)
    except ValueError:
        # Don't generate an output
        evolved = None
    #except RuntimeError

    data['apply_array'] = {
        'hamil': hamil,
        'wfn_out': out,
        'wfn_evolve': evolved
    }

    # Quadratic Ham
    if not full:
        hamil = get_restricted_hamiltonian((h1e,))
    else:
        hamil = get_gso_hamiltonian((h1e,))
    hamil._conserve_number = number_conserving
    try:
        out = wfn.apply(hamil)
    except ValueError:
        # Don't generate an output
        out = None
    try:
        evolved = wfn.time_evolve(0.1, hamil)
    except ValueError:
        # Don't generate an output
        evolved = None

    data['apply_quadratic'] = {
        'hamil': hamil,
        'wfn_out': out,
        'wfn_evolve': evolved
    }

    # Diagonal Coulomb Ham
    vijkl = numpy.zeros((norbs, norbs, norbs, norbs), dtype=numpy.complex128)
    for i in range(norbs):
        for j in range(norbs):
            vijkl[i, j, i, j] += 4 * (i % norbs + 1) * (j % norbs + 1) * 0.21
    hamil = get_diagonalcoulomb_hamiltonian(vijkl)
    hamil._conserve_number = number_conserving
    try:
        out = wfn.apply(hamil)
    except ValueError:
        # Don't generate an output
        out = None
    try:
        evolved = wfn.time_evolve(0.1, hamil)
    except ValueError:
        # Don't generate an output
        evolved = None

    data['apply_dc'] = {'hamil': hamil, 'wfn_out': out, 'wfn_evolve': evolved}

    return data


def regenerate_reference_data():
    """
    Regenerates the reference data
    """
    for param in spin_broken_cases:
        nele = param[0][0]
        norb = param[0][2]
        filename = f'{nele:02d}XX{norb:02d}.pickle'
        with open(os.path.join(datadir, filename), 'wb') as f:
            pickle.dump(generate_data(param), f)

    for param in particle_broken_cases:
        sz = param[0][1]
        norb = param[0][2]
        filename = f'XX{sz:02d}{norb:02d}.pickle'
        with open(os.path.join(datadir, filename), 'wb') as f:
            pickle.dump(generate_data(param), f)

    for param in conserving_cases:
        nele = param[0][0]
        sz = param[0][1]
        norb = param[0][2]
        filename = f'{nele:02d}{sz:02d}{norb:02d}.pickle'
        with open(os.path.join(datadir, filename), 'wb') as f:
            pickle.dump(generate_data(param), f)


if __name__ == "__main__":
    regenerate_reference_data()
