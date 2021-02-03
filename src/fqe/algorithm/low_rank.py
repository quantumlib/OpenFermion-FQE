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
"""Functions for time-evolving a wavefunction."""
from itertools import product

import numpy as np

import openfermion as of
from openfermion import givens_decomposition_square

from fqe.wavefunction import Wavefunction
from fqe.hamiltonians.diagonal_coulomb import DiagonalCoulomb


def evolve_fqe_givens_sector(wfn: Wavefunction, u: np.ndarray,
                             sector='alpha') -> Wavefunction:
    """Evolve a wavefunction by u generated from a 1-body Hamiltonian.

    Args:
        wfn: FQE Wavefunction on n-orbitals
        u: (n x n) unitary matrix.
        sector: Optional either 'alpha' or 'beta' indicating which sector
                to rotate
    Returns:
        New evolved wfn object.
    """
    if sector == 'alpha':
        sigma = 0
    elif sector == 'beta':
        sigma = 1
    else:
        raise ValueError("Bad section variable.  Either (alpha) or (beta)")

    if not np.isclose(u.shape[0], wfn.norb()):
        raise ValueError(
            "unitary is not specified for the correct number of orbitals")

    rotations, diagonal = givens_decomposition_square(u.copy())
    # Iterate through each layer and time evolve by the appropriate
    # fermion operators
    for layer in rotations:
        for givens in layer:
            i, j, theta, phi = givens
            if not np.isclose(phi, 0):
                op = of.FermionOperator(
                    ((2 * j + sigma, 1), (2 * j + sigma, 0)), coefficient=-phi)
                wfn = wfn.time_evolve(1.0, op)
            if not np.isclose(theta, 0):
                op = of.FermionOperator(((2 * i + sigma, 1),
                                         (2 * j + sigma, 0)),
                                        coefficient=-1j * theta) + \
                     of.FermionOperator(((2 * j + sigma, 1),
                                         (2 * i + sigma, 0)),
                                        coefficient=1j * theta)
                wfn = wfn.time_evolve(1.0, op)

    # evolve the last diagonal phases
    for idx, final_phase in enumerate(diagonal):
        if not np.isclose(final_phase, 1.0):
            op = of.FermionOperator(
                ((2 * idx + sigma, 1), (2 * idx + sigma, 0)),
                -np.angle(final_phase))
            wfn = wfn.time_evolve(1.0, op)

    return wfn


def evolve_fqe_givens(wfn: Wavefunction, u: np.ndarray) -> Wavefunction:
    """Evolve a wavefunction by u generated from a 1-body Hamiltonian.

    Args:
        wfn: 2^{n} x 1 vector.
        u: (n//2 x n//2) unitary matrix.

    Returns:
        New evolved wfn object.
    """
    rotations, diagonal = givens_decomposition_square(u.copy())
    # Iterate through each layer and time evolve by the appropriate
    # fermion operators
    for layer in rotations:
        for givens in layer:
            i, j, theta, phi = givens
            if not np.isclose(phi, 0):
                op = of.FermionOperator(((2 * j, 1), (2 * j, 0)),
                                        coefficient=-phi)
                wfn = wfn.time_evolve(1.0, op)
                op = of.FermionOperator(((2 * j + 1, 1), (2 * j + 1, 0)),
                                        coefficient=-phi)
                wfn = wfn.time_evolve(1.0, op)
            if not np.isclose(theta, 0):
                op = of.FermionOperator(
                    ((2 * i, 1),
                     (2 * j, 0)), coefficient=-1j * theta) + of.FermionOperator(
                         ((2 * j, 1), (2 * i, 0)), coefficient=1j * theta)
                wfn = wfn.time_evolve(1.0, op)
                op = of.FermionOperator(
                    ((2 * i + 1, 1), (2 * j + 1, 0)),
                    coefficient=-1j * theta) + of.FermionOperator(
                        ((2 * j + 1, 1), (2 * i + 1, 0)),
                        coefficient=1j * theta)
                wfn = wfn.time_evolve(1.0, op)

    # evolve the last diagonal phases
    for idx, final_phase in enumerate(diagonal):
        if not np.isclose(final_phase, 1.0):
            op = of.FermionOperator(((2 * idx, 1), (2 * idx, 0)),
                                    -np.angle(final_phase))
            wfn = wfn.time_evolve(1.0, op)
            op = of.FermionOperator(((2 * idx + 1, 1), (2 * idx + 1, 0)),
                                    -np.angle(final_phase))
            wfn = wfn.time_evolve(1.0, op)

    return wfn


def evolve_fqe_givens_unrestricted(wfn: Wavefunction,
                                   u: np.ndarray) -> Wavefunction:
    """Evolve a wavefunction by u generated from a 1-body Hamiltonian.

    Args:
        wfn: 2^{n} x 1 vector.
        u: (n x n) unitary matrix.

    Returns:
        New evolved wfn object.
    """
    rotations, diagonal = givens_decomposition_square(u.copy())
    # Iterate through each layer and time evolve by the appropriate
    # fermion operators
    for layer in rotations:
        for givens in layer:
            i, j, theta, phi = givens
            if not np.isclose(phi, 0):
                op = of.FermionOperator(((j, 1), (j, 0)), coefficient=-phi)
                wfn = wfn.time_evolve(1.0, op)
            if not np.isclose(theta, 0):
                op = of.FermionOperator(
                    ((i, 1),
                     (j, 0)), coefficient=-1j * theta) + of.FermionOperator(
                         ((j, 1), (i, 0)), coefficient=1j * theta)
                wfn = wfn.time_evolve(1.0, op)

    # evolve the last diagonal phases
    for idx, final_phase in enumerate(diagonal):
        if not np.isclose(final_phase, 1.0):
            op = of.FermionOperator(((idx, 1), (idx, 0)),
                                    -np.angle(final_phase))
            wfn = wfn.time_evolve(1.0, op)

    return wfn


def evolve_fqe_charge_charge_unrestricted(wfn: Wavefunction,
                                          vij_mat: np.ndarray,
                                          time=1) -> Wavefunction:
    r"""Utility for testing evolution of a full 2^{n} wavefunction via

    :math:`exp{-i time * \sum_{i,j}v_{i, j}n_{i}n_{j}}.`

    Args:
        wfn: fqe_wf with sdim = n
        vij_mat: List[(n x n] matrices
        time: evolution time.

    Returns:
        New evolved 2^{n} x 1 vector
    """
    nso = vij_mat.shape[0]
    for p, q in product(range(nso), repeat=2):
        if np.isclose(vij_mat[p, q], 0):
            continue
        fop = of.FermionOperator(((p, 1), (p, 0), (q, 1), (q, 0)),
                                 coefficient=vij_mat[p, q])
        wfn = wfn.time_evolve(time, fop)
    return wfn


def evolve_fqe_charge_charge_alpha_beta(wfn: Wavefunction,
                                        vij_mat: np.ndarray,
                                        time=1) -> Wavefunction:
    r"""Utility for testing evolution of a full 2^{n} wavefunction via

    :math:`exp{-i time * \sum_{i,j}v_{i, j}n_{i, alpha}n_{j, beta}}.`

    Args:
        wfn: fqe_wf with sdim = n
        vij_mat: List[(n x n] matrices n is the spatial orbital rank
        time: evolution time.

    Returns:
        New evolved 2^{2 * n} x 1 vector
    """
    norbs = vij_mat.shape[0]
    for p, q in product(range(norbs), repeat=2):
        if np.isclose(vij_mat[p, q], 0):
            continue
        fop = of.FermionOperator(
            ((2 * p, 1), (2 * p, 0), (2 * q + 1, 1), (2 * q + 1, 0)),
            coefficient=vij_mat[p, q])
        wfn = wfn.time_evolve(time, fop)
    return wfn


def evolve_fqe_charge_charge_sector(wfn: Wavefunction,
                                    vij_mat: np.ndarray,
                                    sector='alpha',
                                    time=1) -> Wavefunction:
    r"""Utility for testing evolution of a full 2^{n} wavefunction via

    :math:`exp{-i time * \sum_{i,j}v_{i, j}n_{i, alpha}n_{j, beta}}.`

    Args:
        wfn: fqe_wf with sdim = n
        vij_mat: List[(n x n] matrices n is the spatial orbital rank
        time: evolution time.

    Returns:
        New evolved 2^{2 * n} x 1 vector
    """
    if sector == 'alpha':
        sigma = 0
    elif sector == 'beta':
        sigma = 1
    else:
        raise ValueError("Sector must be alpha or beta.")

    norbs = vij_mat.shape[0]
    for p, q in product(range(norbs), repeat=2):
        if np.isclose(vij_mat[p, q], 0):
            continue
        fop = of.FermionOperator(((2 * p + sigma, 1), (2 * p + sigma, 0),
                                  (2 * q + sigma, 1), (2 * q + sigma, 0)),
                                 coefficient=vij_mat[p, q])
        wfn = wfn.time_evolve(time, fop)
    return wfn


def evolve_fqe_diagaonal_coulomb(wfn: Wavefunction, vij_mat: np.ndarray,
                                 time=1) -> Wavefunction:
    r"""Utility for testing evolution of a full 2^{n} wavefunction via

    :math:`exp{-i time * \sum_{i,j, sigma, tau}v_{i, j}n_{i\sigma}n_{j\tau}}.`

    Args:
        wfn: 2^{n} x 1 vector.
        vij_mat: List[(n//2 x n//2)] matrices
        time: evolution time.

    Returns:
        New evolved 2^{n} x 1 vector
    """
    dc_ham = DiagonalCoulomb(vij_mat)
    return wfn.time_evolve(time, dc_ham)


def double_factor_trotter_evolution(initial_wfn: Wavefunction,
                                    basis_change_unitaries, vij_mats,
                                    deltat) -> Wavefunction:
    r"""Doubled Factorized Trotter Evolution

    Evolves an initial according to the double factorized algorithm where each
    Trotter step is determined by the strategy in arXiv:1808.02625.

    Args:
        initial_wfn: Initial wavefunction to evolve.
        basis_change_unitaries: List L + 1 unitaries. The first
            unitary is U1 :math:`e^{-iTdt}` where T is the one-electron
            component of the evolution.  he remaining unitaries are
            :math:`U_{i}U_{i-1}^{\dagger}.` All unitaries are expressed with
            respect to the number of spatial basis functions.
        vij_mats: list matrices of rho-rho interactions where
            i, j indices of the matrix index the :math:`n_{i} n_{j}` integral.
            Evolution is performed with respect to :math:`n_{i\sigma} n_{j\tau}`
            where sigma and tau are up or down electron spins--a total of 4
            Hamiltonian terms per i, j term.
        deltat: evolution time prefactor for all v_ij Hamiltonians.

    Returns:
        The final wavefunction from a single Trotter evolution.
    """
    if len(basis_change_unitaries) - 1 != len(vij_mats):
        raise ValueError(
            "number of basis changes is not consistent with len(vij)")

    intermediate_wfn = evolve_fqe_givens(initial_wfn, basis_change_unitaries[0])
    for step in range(1, len(basis_change_unitaries)):
        intermediate_wfn = evolve_fqe_diagaonal_coulomb(intermediate_wfn,
                                                        vij_mats[step - 1],
                                                        deltat)
        intermediate_wfn = evolve_fqe_givens(intermediate_wfn,
                                             basis_change_unitaries[step])

    return intermediate_wfn
