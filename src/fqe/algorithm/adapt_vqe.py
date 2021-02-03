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
"""Infrastructure for ADAPT VQE algorithm"""
from typing import List, Tuple, Union, Dict
import copy

from itertools import product
import numpy as np
import scipy as sp

import openfermion as of
from openfermion import (
    make_reduced_hamiltonian,
    InteractionOperator,
)
from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermion.linalg import wedge

import fqe
from fqe.wavefunction import Wavefunction
from fqe.hamiltonians.restricted_hamiltonian import RestrictedHamiltonian
from fqe.hamiltonians.hamiltonian import Hamiltonian as ABCHamiltonian
from fqe.fqe_decorators import build_hamiltonian
from fqe.algorithm.brillouin_calculator import (
    get_fermion_op,
    two_rdo_commutator_symm,
    one_rdo_commutator_symm,
)
from fqe.algorithm.generalized_doubles_factorization import \
    doubles_factorization_svd, doubles_factorization_takagi

from fqe.algorithm.low_rank import evolve_fqe_charge_charge_unrestricted, \
    evolve_fqe_givens_unrestricted


def valdemaro_reconstruction(tpdm, n_electrons):
    """
    Build a 3-RDM by cumulant expansion and setting 3rd cumulant to zero

    d3 approx = D ^ D ^ D + 3 (2C) ^ D

    tpdm has normalization (n choose 2) where n is the number of electrons

    Args:
        tpdm (np.ndarray): four-tensor representing the two-RDM
        n_electrons (int): number of electrons in the system
    Returns:
        six-tensor reprsenting the three-RDM
    """
    opdm = (2 / (n_electrons - 1)) * np.einsum('ijjk', tpdm)
    unconnected_tpdm = wedge(opdm, opdm, (1, 1), (1, 1))
    unconnected_d3 = wedge(opdm, unconnected_tpdm, (1, 1), (2, 2))
    return 3 * wedge(tpdm, opdm, (2, 2), (1, 1)) - 2 * unconnected_d3


class OperatorPool:

    def __init__(self, norbs: int, occ: List[int], virt: List[int]):
        """
        Routines for defining operator pools

        Args:
            norbs: number of spatial orbitals
            occ: list of indices of the occupied orbitals
            virt: list of indices of the virtual orbitals
        """
        self.norbs = norbs
        self.occ = occ
        self.virt = virt
        self.op_pool: List[of.FermionOperator] = []

    def singlet_t2(self):
        """
        Generate singlet rotations
        T_{ij}^{ab} = T^(v1a, v2b)_{o1a, o2b} + T^(v1b, v2a)_{o1b, o2a} +
                      T^(v1a, v2a)_{o1a, o2a} + T^(v1b, v2b)_{o1b, o2b}

        where v1,v2 are indices of the virtual obritals and o1, o2 are
        indices of the occupied orbitals with respect to the Hartree-Fock
        reference.
        """
        for oo_i in self.occ:
            for oo_j in self.occ:
                for vv_a in self.virt:
                    for vv_b in self.virt:
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

    def generalized_two_body(self):
        """
        Doubles generators each with distinct Sz expectation value.

        """
        for i, j, k, l in product(range(2 * self.norbs), repeat=4):
            if i != j and k != l:
                op = ((i, 1), (j, 1), (k, 0), (l, 0))
                fop_aa = of.FermionOperator(op)
                fop_aa = fop_aa - of.hermitian_conjugated(fop_aa)
                self.op_pool.append(fop_aa)

    def two_body_sz_adapted(self):
        """
        Doubles generators each with distinct Sz expectation value.

        G^{isigma, jtau, ktau, lsigma) for sigma, tau in 0, 1
        """
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
            fop_ab = of.FermionOperator(op_ab)
            fop_ab = fop_ab - of.hermitian_conjugated(fop_ab)
            fop_ab = of.normal_ordered(fop_ab)
            if not np.isclose(fop_ab.induced_norm(), 0):
                self.op_pool.append(fop_ab)

    def one_body_sz_adapted(self):
        # alpha-alpha rotation
        # beta-beta rotation
        for i, j in product(range(self.norbs), repeat=2):
            if i > j:
                op_aa = ((2 * i, 1), (2 * j, 0))
                op_bb = ((2 * i + 1, 1), (2 * j + 1, 0))
                fop_aa = of.FermionOperator(op_aa)
                fop_aa = fop_aa - of.hermitian_conjugated(fop_aa)
                fop_bb = of.FermionOperator(op_bb)
                fop_bb = fop_bb - of.hermitian_conjugated(fop_bb)
                fop_aa = of.normal_ordered(fop_aa)
                fop_bb = of.normal_ordered(fop_bb)
                self.op_pool.append(fop_aa)
                self.op_pool.append(fop_bb)


class ADAPT:

    def __init__(self,
                 oei: np.ndarray,
                 tei: np.ndarray,
                 operator_pool,
                 n_alpha: int,
                 n_beta: int,
                 iter_max=30,
                 verbose=True,
                 stopping_epsilon=1.0E-3,
                 delta_e_eps=1.0E-6):
        """
        ADAPT-VQE object.

        Args:
            oei: one electron integrals in the spatial basis
            tei: two-electron integrals in the spatial basis
            operator_pool: Object with .op_pool that is a list of antihermitian
                           FermionOperators
            n_alpha: Number of alpha-electrons
            n_beta: Number of beta-electrons
            iter_max: Maximum ADAPT-VQE steps to take
            verbose: Print the iteration information
            stopping_epsilon: define the <[G, H]> value that triggers stopping

        """
        elec_hamil = RestrictedHamiltonian((oei, np.einsum("ijlk", -0.5 * tei)))
        soei, stei = spinorb_from_spatial(oei, tei)
        astei = np.einsum('ijkl', stei) - np.einsum('ijlk', stei)
        molecular_hamiltonian = InteractionOperator(0, soei, 0.25 * astei)

        reduced_ham = make_reduced_hamiltonian(molecular_hamiltonian,
                                               n_alpha + n_beta)
        self.reduced_ham = reduced_ham
        self.k2_ham = of.get_fermion_operator(reduced_ham)
        self.k2_fop = build_hamiltonian(self.k2_ham,
                                        elec_hamil.dim(),
                                        conserve_number=True)
        self.elec_hamil = elec_hamil
        self.iter_max = iter_max
        self.sdim = elec_hamil.dim()
        # change to use multiplicity to derive this for open shell
        self.nalpha = n_alpha
        self.nbeta = n_beta
        self.sz = self.nalpha - self.nbeta
        self.nele = self.nalpha + self.nbeta
        self.verbose = verbose
        self.operator_pool = operator_pool
        self.stopping_eps = stopping_epsilon
        self.delta_e_eps = delta_e_eps

    def adapt_vqe(self,
                  initial_wf: Wavefunction,
                  opt_method: str = 'L-BFGS-B',
                  opt_options=None,
                  v_reconstruct: bool = True,
                  num_ops_add: int = 1):
        """
        Run ADAPT-VQE using

        Args:
            initial_wf: Initial wavefunction at the start of the calculation
            opt_method: scipy optimizer to use
            opt_options: options  for scipy optimizer
            v_reconstruct: use valdemoro reconstruction
            num_ops_add: add this many operators from the pool to the
                         wavefunction
        """
        if opt_options is None:
            opt_options = {}
        operator_pool = []
        operator_pool_fqe: List[ABCHamiltonian] = []
        existing_parameters: List[float] = []
        self.gradients = []
        self.energies = [initial_wf.expectationValue(self.k2_fop)]
        iteration = 0
        while iteration < self.iter_max:
            # get current wavefunction
            wf = copy.deepcopy(initial_wf)
            for fqe_op, coeff in zip(operator_pool_fqe, existing_parameters):
                wf = wf.time_evolve(coeff, fqe_op)

            # calculate rdms for grad
            _, tpdm = wf.sector((self.nele, self.sz)).get_openfermion_rdms()
            if v_reconstruct:
                d3 = 6 * valdemaro_reconstruction(tpdm / 2, self.nele)
            else:
                d3 = wf.sector((self.nele, self.sz)).get_three_pdm()

            # get ACSE Residual and 2-RDM gradient
            acse_residual = two_rdo_commutator_symm(
                self.reduced_ham.two_body_tensor, tpdm, d3)
            one_body_residual = one_rdo_commutator_symm(
                self.reduced_ham.two_body_tensor, tpdm)

            # calculate grad of each operator in the pool
            pool_grad = []
            for operator in self.operator_pool.op_pool:
                grad_val = 0
                for op_term, coeff in operator.terms.items():
                    idx = [xx[0] for xx in op_term]
                    if len(idx) == 4:
                        grad_val += acse_residual[tuple(idx)] * coeff
                    elif len(idx) == 2:
                        grad_val += one_body_residual[tuple(idx)] * coeff
                pool_grad.append(grad_val)

            max_grad_terms_idx = \
                np.argsort(np.abs(pool_grad))[::-1][:num_ops_add]

            pool_terms = [
                self.operator_pool.op_pool[i] for i in max_grad_terms_idx
            ]
            operator_pool.extend(pool_terms)
            fqe_ops: List[ABCHamiltonian] = []
            for f_op in pool_terms:
                fqe_ops.append(
                    build_hamiltonian(1j * f_op,
                                      self.sdim,
                                      conserve_number=True))
            operator_pool_fqe.extend(fqe_ops)
            existing_parameters.extend([0] * len(fqe_ops))

            new_parameters, current_e = self.optimize_param(
                operator_pool_fqe,
                existing_parameters,
                initial_wf,
                opt_method,
                opt_options=opt_options)

            existing_parameters = new_parameters.tolist()
            if self.verbose:
                print(iteration, current_e, max(np.abs(pool_grad)))
            self.energies.append(current_e)
            self.gradients.append(pool_grad)
            if max(np.abs(pool_grad)) < self.stopping_eps or np.abs(
                    self.energies[-2] - self.energies[-1]) < self.delta_e_eps:
                break
            iteration += 1

    def optimize_param(
            self,
            pool: Union[List[of.FermionOperator], List[ABCHamiltonian]],
            existing_params: Union[List, np.ndarray],
            initial_wf: Wavefunction,
            opt_method: str,
            opt_options=None) -> Tuple[np.ndarray, float]:
        """Optimize a wavefunction given a list of generators

        Args:
            pool: generators of rotation
            existing_params: parameters for the generators
            initial_wf: initial wavefunction
            opt_method: Scpy.optimize method
        """
        if opt_options is None:
            opt_options = {}

        def cost_func(params):
            assert len(params) == len(pool)
            # compute wf for function call
            wf = copy.deepcopy(initial_wf)
            for op, coeff in zip(pool, params):
                if np.isclose(coeff, 0):
                    continue
                if isinstance(op, ABCHamiltonian):
                    fqe_op = op
                else:
                    print("Found a OF Hamiltonian")
                    fqe_op = build_hamiltonian(1j * op,
                                               self.sdim,
                                               conserve_number=True)
                if isinstance(fqe_op, ABCHamiltonian):
                    wf = wf.time_evolve(coeff, fqe_op)
                else:
                    raise ValueError("Can't evolve operator type {}".format(
                        type(fqe_op)))

            # compute gradients
            grad_vec = np.zeros(len(params), dtype=np.complex128)
            # avoid extra gradient computation if we can
            if opt_method not in ['Nelder-Mead', 'COBYLA']:
                for pidx, _ in enumerate(params):
                    # evolve e^{iG_{n-1}g_{n-1}}e^{iG_{n-2}g_{n-2}}x
                    # G_{n-3}e^{-G_{n-3}g_{n-3}...|0>
                    grad_wf = copy.deepcopy(initial_wf)
                    for gidx, (op, coeff) in enumerate(zip(pool, params)):
                        if isinstance(op, ABCHamiltonian):
                            fqe_op = op
                        else:
                            fqe_op = build_hamiltonian(1j * op,
                                                       self.sdim,
                                                       conserve_number=True)
                        if not np.isclose(coeff, 0):
                            grad_wf = grad_wf.time_evolve(coeff, fqe_op)
                            # if looking at the pth parameter then apply the
                            # operator to the state
                        if gidx == pidx:
                            grad_wf = grad_wf.apply(fqe_op)

                    # grad_val = grad_wf.expectationValue(self.elec_hamil,
                    # brawfn=wf)
                    grad_val = grad_wf.expectationValue(self.k2_fop, brawfn=wf)

                    grad_vec[pidx] = -1j * grad_val + 1j * grad_val.conj()
                    assert np.isclose(grad_vec[pidx].imag, 0)
            return (wf.expectationValue(self.k2_fop).real,
                    np.array(grad_vec.real, order='F'))

        res = sp.optimize.minimize(cost_func,
                                   existing_params,
                                   method=opt_method,
                                   jac=True,
                                   options=opt_options)
        return res.x, res.fun
