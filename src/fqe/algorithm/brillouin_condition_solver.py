import copy

import numpy as np

from openfermion import MolecularData, make_reduced_hamiltonian

from fqe.hamiltonians.restricted_hamiltonian import RestrictedHamiltonian
from fqe.algorithm.brillouin_calculator import (get_fermion_op,
                                                get_acse_residual_fqe,
                                                get_acse_residual_fqe_parallel,
                                                get_tpdm_grad_fqe,
                                                get_tpdm_grad_fqe_parallel, )

try:
    from joblib import Parallel, delayed
    PARALLELIZABLE = True
except ImportError:
    PARALLELIZABLE = False


class BrillouinCondition:
    def __init__(self, molecule=MolecularData, iter_max=30,
                 run_parallel=False, verbose=True):
        oei, tei = molecule.get_integrals()
        elec_hamil = RestrictedHamiltonian((oei, np.einsum('ijlk', -0.5 * tei)))
        moleham = molecule.get_molecular_hamiltonian()
        reduced_ham = make_reduced_hamiltonian(moleham, molecule.n_electrons)
        self.molecule = molecule
        self.reduced_ham = reduced_ham
        self.elec_hamil = elec_hamil
        self.iter_max = iter_max
        self.sdim = elec_hamil.dim()
        if PARALLELIZABLE and run_parallel:
            self.parallel = True
        else:
            self.parallel = False
        self.verbose = verbose

        # store results
        self.acse_energy = []

    def bc_solve(self, initial_wf):
        """
        Propogate BC differential eq. until convergence

        :param initial_wf: Initial wavefunction to evolve
        """
        fqe_wf = copy.deepcopy(initial_wf)
        sdim = self.sdim
        iter_max = self.iter_max
        iter = 0
        h = 1.0E-4
        self.acse_energy = [fqe_wf.expectationValue(self.elec_hamil).real]
        while iter < iter_max:
            if self.parallel:
                acse_residual = get_acse_residual_fqe_parallel(fqe_wf,
                                                               self.elec_hamil,
                                                               sdim)
                acse_res_op = get_fermion_op(acse_residual)
                tpdm_grad = get_tpdm_grad_fqe(fqe_wf, acse_residual, sdim)

            else:
                acse_residual = get_acse_residual_fqe(fqe_wf, self.elec_hamil,
                                                      sdim)
                acse_res_op = get_fermion_op(acse_residual)
                tpdm_grad = get_tpdm_grad_fqe_parallel(fqe_wf, acse_residual,
                                                       sdim)

            # epsilon_opt = - Tr[K, D'(lambda)] / Tr[K, D''(lambda)]
            # K is reduced Hamiltonian
            # get approximate D'' by short propagation
            # TODO: do this with cumulant reconstruction instead of wf prop.
            fqe_wfh = fqe_wf.time_evolve(h, 1j * acse_res_op)
            acse_residualh = get_acse_residual_fqe(fqe_wfh, self.elec_hamil,
                                                   sdim)
            tpdm_gradh = get_tpdm_grad_fqe(fqe_wfh, acse_residualh, sdim)
            tpdm_gradgrad = (1/h) * (tpdm_gradh - tpdm_grad)
            epsilon = -np.einsum('ijkl,ijkl', self.reduced_ham.two_body_tensor,
                                 tpdm_grad)
            epsilon /= np.einsum('ijkl,ijkl', self.reduced_ham.two_body_tensor,
                                 tpdm_gradgrad)
            epsilon = epsilon.real

            fqe_wf = fqe_wf.time_evolve(epsilon, 1j * acse_res_op)
            current_energy = fqe_wf.expectationValue(self.elec_hamil).real
            self.acse_energy.append(current_energy.real)

            print_string = "Iter {: 5f}\tcurrent energy {: 5.10f}\t".format(
                iter, current_energy)
            print_string += "|dE| {: 5.10f}\tStep size {: 5.10f}".format(
                np.abs(self.acse_energy[-2] - self.acse_energy[-1]), epsilon)

            if self.verbose:
                print(print_string)

            if (iter >= 1 and
               np.abs(self.acse_energy[-2] - self.acse_energy[-1]) < 0.5E-4):
                break
            iter += 1
