import numpy as np
import openfermion as of
from openfermion import givens_decomposition_square

import fqe
from fqe.hamiltonians.restricted_hamiltonian import RestrictedHamiltonian
from fqe.hamiltonians.diagonal_coulomb import DiagonalCoulomb


def evolve_fqe_givens(wfn: fqe.Wavefunction, u: np.ndarray) -> np.ndarray:
    """
    Evolve a wavefunction by u generated from a 1-body Hamiltonian
    Args:
        wfn np.ndarray: 2^{n} x 1 vector
        u: (n//2 x n//2) unitary matrix

    Returns:
        New evolved wfn object
    """
    rotations, diagonal = givens_decomposition_square(u.copy())
    # Iterate through each layer and time evolve by the appropriate
    # fermion operators
    for layer in rotations:
        for givens in layer:
            i, j, theta, phi = givens
            if not np.isclose(phi, 0):
                op = of.FermionOperator(((2 * j, 1), (2 * j, 0)), coefficient=-phi)
                wfn = wfn.time_evolve(1., op)
                op = of.FermionOperator(((2 * j + 1, 1), (2 * j + 1, 0)),
                                        coefficient=-phi)
                wfn = wfn.time_evolve(1., op)
            if not np.isclose(theta, 0):
                op = of.FermionOperator(((2 * i, 1), (2 * j, 0)),
                                        coefficient=-1j * theta) + of.FermionOperator(
                    ((2 * j, 1), (2 * i, 0)), coefficient=1j * theta)
                wfn = wfn.time_evolve(1., op)
                op = of.FermionOperator(((2 * i + 1, 1), (2 * j + 1, 0)),
                                        coefficient=-1j * theta) + of.FermionOperator(
                    ((2 * j + 1, 1), (2 * i + 1, 0)), coefficient=1j * theta)
                wfn = wfn.time_evolve(1., op)

    # evolve the last diagonal phases
    for idx, final_phase in enumerate(diagonal):
        if not np.isclose(final_phase, 1.0):
            op = of.FermionOperator(((2 * idx, 1), (2 * idx, 0)),
                                    -np.angle(final_phase))
            wfn = wfn.time_evolve(1., op)
            op = of.FermionOperator(((2 * idx + 1, 1), (2 * idx + 1, 0)),
                                    -np.angle(final_phase))
            wfn = wfn.time_evolve(1., op)

    return wfn



def evolve_fqe_diagaonal_coulomb(wfn: fqe.Wavefunction, vij_mat: np.ndarray,
                                 time=1) -> fqe.Wavefunction:
    """
    Utility for testing evolution of a full 2^{n} wavefunction

    exp{-i time * \sum_{i,j, sigma, tau}v_{i, j}n_{i\sigma}n_{j\tau}}

    Args:
        wfn np.ndarray: 2^{n} x 1 vector
        vij_mat List[np.ndarray]: List[(n//2 x n//2)] matrices
        time float: evolution time

    Returns:
        New evolved 2^{n} x 1 vector
    """
    dc_ham = DiagonalCoulomb(vij_mat)
    return wfn.time_evolve(time, dc_ham)


def double_factor_trotter_evolution(initial_wfn: fqe.Wavefunction,
                                    basis_change_unitaries,
                                    vij_mats, deltat) -> fqe.Wavefunction:
    """
    Doubled Factorized Trotter Evolution

    Evolves an initial according to the double factorized algorithm where each
    Trotter step is determined by the strategy in arXiv:1808.02625.

    Args:
        initial_wfn fqe.Wavefunction: initial wavefunction to evolve
        basis_change_unitaries List[np.ndarray]: List L + 1 unitaries. The first
            unitary is U1 e^{-iTdt} where T is the one-electron component of
            the evolution.  The remaining unitaries are U_{i}U_{i-1}^{\dagger}.
            All unitaries are expressed with respect to the number of spatial
            basis functions.
        vij_mats List[np.ndarray]: list matrices of rho-rho interactions where
            i, j indices of the matrix index the n_{i}n_{j} integral.  Evolution
            is performed with respect to n_{i\sigma}n_{j\tau} where sigma and
            tau are up or down electron spins--a total of 4 Hamiltonian terms
            per i,j term.
        deltat float: evolution time prefactor for all v_ij Hamiltonians

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

        # intermediate_wfn.print_wfn()
        # print(intermediate_wfn.norm())
    return intermediate_wfn
