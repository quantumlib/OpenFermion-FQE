"""Infrastructure for ADAPT VQE algorithm"""
from typing import List
import os
import copy
import openfermion as of
import fqe
from itertools import product
import numpy as np
import scipy as sp

from openfermion import (MolecularData, make_reduced_hamiltonian,
                         InteractionOperator, )
from openfermion.chem.molecular_data import spinorb_from_spatial

from fqe.hamiltonians.restricted_hamiltonian import RestrictedHamiltonian
from fqe.fqe_decorators import build_hamiltonian
from fqe.algorithm.brillouin_calculator import (
    get_fermion_op,
    get_acse_residual_fqe,
    get_acse_residual_fqe_parallel,
    get_tpdm_grad_fqe,
    get_tpdm_grad_fqe_parallel,
    two_rdo_commutator_antisymm,
    two_rdo_commutator_symm
)

class OperatorPool:

    def __init__(self, norbs: int, occ: List[int], virt: List[int]):
        """
        Define an operator pool
        """
        self.norbs = norbs
        self.occ = occ
        self.virt = virt
        self.op_pool = []

    def singlet_t2(self):
        """
        Generate singlet rotations
        T_{ij}^{ab} = E(ai)E(bj)
        """
        for oidx, oo_i in enumerate(self.occ):
            for ojdx, oo_j in enumerate(self.occ):
                for vadx, vv_a in enumerate(self.virt):
                    for vbdx, vv_b in enumerate(self.virt):
                        term = of.FermionOperator()
                        for sigma, tau in product(range(2), repeat=2):
                            op = ((2 * vv_a + sigma, 1), (2 * vv_b + tau, 1),
                                  (2 * oo_j + tau, 0), (2 * oo_i + sigma, 0))
                            if (2 * vv_a + sigma == 2 * vv_b + tau or
                                    2 * oo_j + tau == 2 * oo_i + sigma):
                                continue
                            fop = of.FermionOperator(op, coefficient=0.5)
                            fop = fop - of.hermitian_conjugated(fop)
                            fop = of.normal_ordered(fop)
                            term += fop
                        self.op_pool.append(term)

    def two_body_sz_adapted(self):
        """
        Doubles generators with aa, bb, alpha-beta, beta-alpha
        """
        # alpha-alpha block
        # beta-beta
        for i, j, k, l in product(range(self.norbs), repeat=4):
            if i != j and k != l:
                op_aa = ((2 * i, 1), (2 * j, 1), (2 * k, 0), (2 * l, 0))
                op_bb = ((2 * i + 1, 1), (2 * j + 1, 1), (2 * k + 1, 0),
                         (2 * l + 1, 0))
                fop_aa = of.FermionOperator(op_aa)
                fop_aa = fop_aa - of.hermitian_conjugated(fop_aa)
                fop_bb = of.FermionOperator(op_bb)
                fop_bb = fop_bb - of.hermitian_conjugated(fop_bb)
                fop_aa = of.normal_ordered(fop_aa)
                fop_bb = of.normal_ordered(fop_bb)
                self.op_pool.append(fop_aa)
                self.op_pool.append(fop_bb)

            op_ab = ((2 * i, 1), (2 * j + 1, 1), (2 * k + 1, 0), (2 * l, 0))
            op_ba =  ((2 * i + 1, 1), (2 * j, 1), (2 * k, 0), (2 * l + 1, 0))
            fop_ab = of.FermionOperator(op_ab)
            fop_ab = fop_ab - of.hermitian_conjugated(fop_ab)
            fop_ab = of.normal_ordered(fop_ab)
            self.op_pool.append(fop_ab)

            fop_ba = of.FermionOperator(op_ba)
            fop_ba = fop_ba - of.hermitian_conjugated(fop_ba)
            fop_ba = of.normal_ordered(fop_ba)
            self.op_pool.append(fop_ba)


class ADAPT:
    def __init__(self, oei: np.ndarray, tei: np.ndarray, operator_pool,
                 iter_max=30, verbose=True
                 ):
        elec_hamil = RestrictedHamiltonian(
            (oei, np.einsum("ijlk", -0.5 * tei))
        )
        soei, stei = spinorb_from_spatial(oei, tei)
        astei = np.einsum('ijkl', stei) - np.einsum('ijlk', stei)
        molecular_hamiltonian = InteractionOperator(
            0, soei, 0.25 * astei)

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
        self.nele = self.nalpha + self.nbeta
        self.verbose = verbose
        self.operator_pool = operator_pool

    def vbc(self, initial_wf: fqe.Wavefunction):
        """The variational Brillouin condition method"""
        operator_pool = []
        existing_parameters = []
        iteration = 0
        while True:
            # get current wavefunction
            wf = copy.deepcopy(initial_wf)
            for op, coeff in zip(operator_pool, existing_parameters):
                fqe_op = build_hamiltonian(1j * op, self.sdim,
                                           conserve_number=True)
                wf = wf.time_evolve(coeff, fqe_op)

            # calculate rdms for grad
            opdm, tpdm = wf.sector((self.nele, self.sz)).get_openfermion_rdms()
            d3 = wf.sector((self.nele, self.sz)).get_three_pdm()
            # get ACSE Residual and 2-RDM gradient
            acse_residual = two_rdo_commutator_symm(
                self.reduced_ham.two_body_tensor, tpdm,
                d3)

            # calculate grad of each operator in the pool
            pool_grad = []
            for operator in self.operator_pool.op_pool:
                grad_val = 0
                for op_term, coeff in operator.terms.items():
                    idx = [xx[0] for xx in op_term]
                    grad_val += acse_residual[tuple(idx)] * coeff
                pool_grad.append(grad_val)

            max_grad_term_idx = np.argmax(np.abs(pool_grad))
            operator_pool.append(self.operator_pool.op_pool[max_grad_term_idx])
            existing_parameters.append(0)

            new_parameters, current_e = self.optimize_param(operator_pool,
                                                 existing_parameters,
                                                 initial_wf)
            existing_parameters = new_parameters.tolist()
            print(iteration, current_e, max(np.abs(pool_grad)))
            if max(np.abs(pool_grad)) < 1.0E-3:
                break
            iteration += 1


    def adapt_vqe(self, initial_wf: fqe.Wavefunction):
        operator_pool = []
        existing_parameters = []
        iteration = 0
        while True:
            # get current wavefunction
            wf = copy.deepcopy(initial_wf)
            for op, coeff in zip(operator_pool, existing_parameters):
                fqe_op = build_hamiltonian(1j * op, self.sdim,
                                           conserve_number=True)
                wf = wf.time_evolve(coeff, fqe_op)

            # calculate rdms for grad
            opdm, tpdm = wf.sector((self.nele, self.sz)).get_openfermion_rdms()
            d3 = wf.sector((self.nele, self.sz)).get_three_pdm()
            # get ACSE Residual and 2-RDM gradient
            acse_residual = two_rdo_commutator_symm(
                self.reduced_ham.two_body_tensor, tpdm,
                d3)

            # calculate grad of each operator in the pool
            pool_grad = []
            for operator in self.operator_pool.op_pool:
                grad_val = 0
                for op_term, coeff in operator.terms.items():
                    idx = [xx[0] for xx in op_term]
                    grad_val += acse_residual[tuple(idx)] * coeff
                pool_grad.append(grad_val)

            max_grad_term_idx = np.argmax(np.abs(pool_grad))
            operator_pool.append(self.operator_pool.op_pool[max_grad_term_idx])
            existing_parameters.append(0)

            new_parameters, current_e = self.optimize_param(operator_pool,
                                                 existing_parameters,
                                                 initial_wf)
            existing_parameters = new_parameters.tolist()
            print(iteration, current_e, max(np.abs(pool_grad)))
            if max(np.abs(pool_grad)) < 1.0E-3:
                break
            iteration += 1

    def optimize_param(self, pool: List[of.FermionOperator], existing_params,
                       initial_wf) -> fqe.wavefunction:

        def cost_func(params):
            assert len(params) == len(pool)
            wf = copy.deepcopy(initial_wf)
            for op, coeff in zip(pool, params):
                if np.isclose(coeff, 0):
                    continue
                fqe_op = build_hamiltonian(1j * op, self.sdim,
                                           conserve_number=True)
                wf = wf.time_evolve(coeff, fqe_op)

            return wf.expectationValue(self.elec_hamil).real

        res = sp.optimize.minimize(cost_func, existing_params, method='COBYLA')
        return res.x, res.fun










def get_h2_molecule(bd):
    from openfermionpyscf import run_pyscf
    geometry = [['H', [0, 0, 0]],
                ['H', [0, 0, bd]]]
    molecule = of.MolecularData(geometry=geometry, charge=0,
                                multiplicity=1,
                                basis='sto-3g')
    molecule.filename = os.path.join(os.getcwd(), molecule.name)
    molecule = run_pyscf(molecule, run_scf=True, run_fci=True)
    return molecule


def get_h4_molecule(bd):
    from openfermionpyscf import run_pyscf
    geometry = [['H', [0, 0, 0]],
                ['H', [0, 0, bd]],
                ['H', [0, 0, 2 * bd]],
                ['H', [0, 0, 3 * bd]],
                ]
    molecule = of.MolecularData(geometry=geometry, charge=0,
                                multiplicity=1,
                                basis='sto-3g')
    molecule.filename = os.path.join(os.getcwd(), molecule.name)
    molecule = run_pyscf(molecule, run_scf=True, run_fci=True)
    return molecule


def get_n2_molecule(bd):
    from openfermionpyscf import run_pyscf
    geometry = [['N', [0, 0, 0]],
                ['N', [0, 0, bd]]]
    molecule = of.MolecularData(geometry=geometry, charge=0,
                                multiplicity=1,
                                basis='sto-3g')
    molecule.filename = os.path.join(os.getcwd(), molecule.name)
    molecule = run_pyscf(molecule, run_scf=True, run_fci=True)
    return molecule


def get_lih_molecule(bd):
    from openfermionpyscf import run_pyscf
    geometry = [['Li', [0, 0, 0]],
                ['H', [0, 0, bd]]]
    molecule = of.MolecularData(geometry=geometry, charge=0,
                                multiplicity=1,
                                basis='sto-3g')
    molecule.filename = os.path.join(os.getcwd(), molecule.name)
    molecule = run_pyscf(molecule, run_scf=True, run_fci=True)
    return molecule


def get_beh2_molecule(bd):
    from openfermionpyscf import run_pyscf
    geometry = [['H', [0, 0, -bd]],
                ['Be', [0, 0, 0]],
                ['H', [0, 0, bd]]]
    molecule = of.MolecularData(geometry=geometry, charge=0,
                                multiplicity=1,
                                basis='sto-3g')
    molecule.filename = os.path.join(os.getcwd(), molecule.name)
    molecule = run_pyscf(molecule, run_scf=True, run_fci=True)
    return molecule


if __name__ == "__main__":
    # molecule = get_h4_molecule(1.2)
    # molecule = get_lih_molecule(2.0)
    molecule = get_beh2_molecule(2.3)
    # molecule = get_h2_molecule(0.8)
    print("HF ", molecule.hf_energy - molecule.nuclear_repulsion)
    print("FCI ", molecule.fci_energy - molecule.nuclear_repulsion)
    norbs = molecule.n_orbitals
    nalpha = molecule.n_electrons // 2
    nbeta = molecule.n_electrons // 2
    nele = nalpha + nbeta
    sz = nalpha - nbeta
    nocc = molecule.n_electrons // 2
    occ = list(range(nocc))
    virt = list(range(nocc, norbs))
    sop = OperatorPool(norbs, occ, virt)
    oei, tei = molecule.get_integrals()
    # sop.singlet_t2()
    sop.two_body_sz_adapted()
    adapt = ADAPT(oei, tei, sop)

    fqe_wf = fqe.Wavefunction([[molecule.n_electrons, sz, molecule.n_orbitals]])
    fqe_wf.set_wfn(strategy='hartree-fock')
    generator = of.FermionOperator()
    fop = of.FermionOperator(((2, 1), (3, 1), (5, 0), (4, 0)), coefficient=-0.5)
    generator += fop - of.hermitian_conjugated(fop)
    fop = of.FermionOperator(((3, 1), (2, 1), (4, 0), (5, 0)), coefficient=-0.5)
    generator += fop - of.hermitian_conjugated(fop)
    # print(of.normal_ordered(1j * generator))
    # generator = of.normal_ordered(generator)
    # fqe_op = build_hamiltonian(1j * generator, norbs, conserve_number=True)
    # fqe_wf = fqe_wf.time_evolve(1.0, 1j * generator)
    # fqe_wf = fqe_wf.time_evolve(1.0, fqe_op)

    #  print(fqe_wf.expectationValue(adapt.elec_hamil).real, molecule.hf_energy - molecule.nuclear_repulsion)
    adapt.adapt_vqe(fqe_wf)
