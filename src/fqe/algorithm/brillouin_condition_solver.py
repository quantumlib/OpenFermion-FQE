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
"""The generalized Brillouin conditions are a series of stationarity conditions
for a Lie algebraic variational principle. This module implements solving for
stationarity based on a subset of the these conditions.
"""
import copy

import numpy as np

from openfermion import (
    MolecularData,
    make_reduced_hamiltonian,
    InteractionOperator,
)
from openfermion.chem.molecular_data import spinorb_from_spatial

from fqe.hamiltonians.restricted_hamiltonian import RestrictedHamiltonian
from fqe.algorithm.brillouin_calculator import (
    get_fermion_op, get_acse_residual_fqe, get_acse_residual_fqe_parallel,
    get_tpdm_grad_fqe, get_tpdm_grad_fqe_parallel, two_rdo_commutator_antisymm,
    two_rdo_commutator_symm)

try:
    from joblib import Parallel

    PARALLELIZABLE = True
except ImportError:
    PARALLELIZABLE = False


class BrillouinCondition:
    """This object provide an interface to solving for stationarity with
    respect to the 2-particle Brillouin condition.
    """

    def __init__(
            self,
            molecule=MolecularData,
            iter_max=30,
            run_parallel=False,
            verbose=True,
    ):
        oei, tei = molecule.get_integrals()
        elec_hamil = RestrictedHamiltonian((oei, np.einsum("ijlk", -0.5 * tei)))
        oei, tei = molecule.get_integrals()
        soei, stei = spinorb_from_spatial(oei, tei)
        astei = np.einsum('ijkl', stei) - np.einsum('ijlk', stei)
        molecular_hamiltonian = InteractionOperator(0, soei, 0.25 * astei)

        # moleham = molecule.get_molecular_hamiltonian()
        reduced_ham = make_reduced_hamiltonian(molecular_hamiltonian,
                                               molecule.n_electrons)
        self.molecule = molecule
        self.reduced_ham = reduced_ham
        self.elec_hamil = elec_hamil
        self.iter_max = iter_max
        self.sdim = elec_hamil.dim()
        # change to use multiplicity to derive this for open shell
        self.nalpha = molecule.n_electrons // 2
        self.nbeta = molecule.n_electrons // 2
        self.sz = self.nalpha - self.nbeta
        if PARALLELIZABLE and run_parallel:
            self.parallel = True
        else:
            self.parallel = False
        self.verbose = verbose

        # store results
        self.acse_energy = []

    def bc_solve(self, initial_wf):
        """Propagate BC differential equation until convergence.

        Args:
            initial_wf: Initial wavefunction to evolve.
        """
        fqe_wf = copy.deepcopy(initial_wf)
        sdim = self.sdim
        iter_max = self.iter_max
        iteration = 0
        h = 1.0e-4
        self.acse_energy = [fqe_wf.expectationValue(self.elec_hamil).real]
        while iteration < iter_max:
            if self.parallel:
                acse_residual = get_acse_residual_fqe_parallel(
                    fqe_wf, self.elec_hamil, sdim)
                acse_res_op = get_fermion_op(acse_residual)
                tpdm_grad = get_tpdm_grad_fqe_parallel(fqe_wf, acse_residual,
                                                       sdim)

            else:
                acse_residual = get_acse_residual_fqe(fqe_wf, self.elec_hamil,
                                                      sdim)
                acse_res_op = get_fermion_op(acse_residual)
                tpdm_grad = get_tpdm_grad_fqe(fqe_wf, acse_residual, sdim)

            # epsilon_opt = - Tr[K, D'(lambda)] / Tr[K, D''(lambda)]
            # K is reduced Hamiltonian
            # get approximate D'' by short propagation
            # TODO: do this with cumulant reconstruction instead of wf prop.
            fqe_wfh = fqe_wf.time_evolve(h, 1j * acse_res_op)
            acse_residualh = get_acse_residual_fqe(fqe_wfh, self.elec_hamil,
                                                   sdim)
            tpdm_gradh = get_tpdm_grad_fqe(fqe_wfh, acse_residualh, sdim)
            tpdm_gradgrad = (1 / h) * (tpdm_gradh - tpdm_grad)
            epsilon = -np.einsum("ijkl,ijkl", self.reduced_ham.two_body_tensor,
                                 tpdm_grad)
            epsilon /= np.einsum("ijkl,ijkl", self.reduced_ham.two_body_tensor,
                                 tpdm_gradgrad)
            epsilon = epsilon.real

            fqe_wf = fqe_wf.time_evolve(epsilon, 1j * acse_res_op)
            current_energy = fqe_wf.expectationValue(self.elec_hamil).real
            self.acse_energy.append(current_energy.real)

            print_string = "Iter {: 5f}\tcurrent energy {: 5.10f}\t".format(
                iteration, current_energy)
            print_string += "|dE| {: 5.10f}\tStep size {: 5.10f}".format(
                np.abs(self.acse_energy[-2] - self.acse_energy[-1]), epsilon)

            if self.verbose:
                print(print_string)

            if (iteration >= 1 and
                    np.abs(self.acse_energy[-2] - self.acse_energy[-1]) <
                    0.5e-4):
                break
            iteration += 1

    def bc_solve_rdms(self, initial_wf):
        """Propagate BC differential equation until convergence.

        State is evolved and then 3-RDM is measured.  This information is
        used to construct a new state

        Args:
            initial_wf: Initial wavefunction to evolve.
        """
        fqe_wf = copy.deepcopy(initial_wf)
        iter_max = self.iter_max
        iteration = 0
        sector = (self.nalpha + self.nbeta, self.sz)
        h = 1.0e-4
        self.acse_energy = [fqe_wf.expectationValue(self.elec_hamil).real]
        while iteration < iter_max:
            # extract FqeData object each iteration in case fqe_wf is copied
            fqe_data = fqe_wf.sector(sector)
            # get RDMs from FqeData
            d3 = fqe_data.get_three_pdm()
            _, tpdm = fqe_data.get_openfermion_rdms()

            # get ACSE Residual and 2-RDM gradient
            acse_residual = two_rdo_commutator_symm(
                self.reduced_ham.two_body_tensor, tpdm, d3)
            tpdm_grad = two_rdo_commutator_antisymm(acse_residual, tpdm, d3)
            acse_res_op = get_fermion_op(acse_residual)

            # epsilon_opt = - Tr[K, D'(lambda)] / Tr[K, D''(lambda)]
            # K is reduced Hamiltonian
            # get approximate D'' by short propagation
            # TODO: do this with cumulant reconstruction instead of wf prop.
            fqe_wfh = fqe_wf.time_evolve(h, 1j * acse_res_op)
            fqe_datah = fqe_wfh.sector(sector)
            d3h = fqe_datah.get_three_pdm()
            _, tpdmh = fqe_datah.get_openfermion_rdms()
            acse_residualh = two_rdo_commutator_symm(
                self.reduced_ham.two_body_tensor, tpdmh, d3h)
            tpdm_gradh = two_rdo_commutator_antisymm(acse_residualh, tpdmh, d3h)

            tpdm_gradgrad = (1 / h) * (tpdm_gradh - tpdm_grad)
            epsilon = -np.einsum("ijkl,ijkl", self.reduced_ham.two_body_tensor,
                                 tpdm_grad)
            epsilon /= np.einsum("ijkl,ijkl", self.reduced_ham.two_body_tensor,
                                 tpdm_gradgrad)
            epsilon = epsilon.real

            fqe_wf = fqe_wf.time_evolve(epsilon, 1j * acse_res_op)
            current_energy = fqe_wf.expectationValue(self.elec_hamil).real
            self.acse_energy.append(current_energy.real)

            print_string = "Iter {: 5f}\tcurrent energy {: 5.10f}\t".format(
                iteration, current_energy)
            print_string += "|dE| {: 5.10f}\tStep size {: 5.10f}".format(
                np.abs(self.acse_energy[-2] - self.acse_energy[-1]), epsilon)

            if self.verbose:
                print(print_string)

            if (iteration >= 1 and
                    np.abs(self.acse_energy[-2] - self.acse_energy[-1]) <
                    0.5e-4):
                break
            iteration += 1
