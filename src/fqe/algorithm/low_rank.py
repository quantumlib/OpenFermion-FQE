import numpy as np
import openfermion as of
from openfermion import givens_decomposition_square

import fqe
from fqe.hamiltonians.restricted_hamiltonian import RestrictedHamiltonian
from fqe.hamiltonians.diagonal_coulomb import DiagonalCoulomb


def evolve_fqe_givens(fqe_wfn: fqe.Wavefunction,
                      u: np.ndarray) -> fqe.Wavefunction:
    """
    Evolve a wavefunction by a
    Args:
        wfn np.ndarray: 2^{n} x 1 vector
        u: (n//2 x n//2) unitary matrix

    Returns:
        New evolved 2^{n} x 1 vector
    """
    rotations, diagonal = givens_decomposition_square(u.copy())
    for layer in rotations:
        # TODO: We only need one hamiltonian per layer. Not n per layer.
        for givens in layer:
            i, j, theta, phi = givens
            zero_op = np.zeros((u.shape[0], u.shape[0]))
            zero_op[j, j] = -phi
            fqe_wfn = fqe_wfn.time_evolve(1., RestrictedHamiltonian((zero_op,)))

            zero_op = np.zeros((u.shape[0], u.shape[0]), dtype=np.complex128)
            zero_op[i, j] = -1j * theta
            zero_op[j, i] = 1j * theta
            assert of.is_hermitian(zero_op)
            fqe_wfn = fqe_wfn.time_evolve(1., RestrictedHamiltonian((zero_op,)))

    # evolve the last diagonal phases
    # TODO: We only need one hamiltonian for the last layer
    for idx, final_phase in enumerate(diagonal):
        if not np.isclose(final_phase, 1.0):
            zero_op = np.zeros((u.shape[0], u.shape[0]))
            zero_op[idx, idx] = -np.angle(final_phase)
            fqe_wfn = fqe_wfn.time_evolve(1., RestrictedHamiltonian((zero_op,)))

    return fqe_wfn


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

        intermediate_wfn.print_wfn()
        print(intermediate_wfn.norm())
    return intermediate_wfn
