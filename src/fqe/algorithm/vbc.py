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
from scipy.linalg import expm

import openfermion as of
from openfermion import (
    make_reduced_hamiltonian,
    InteractionOperator,
)
from openfermion.chem.molecular_data import spinorb_from_spatial

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


def valdemaro_reconstruction_functional(tpdm, n_electrons, true_opdm=None):
    """
    d3 approx = D ^ D ^ D + 3 (2C) ^ D

    tpdm has normalization (n choose 2) where n is the number of electrons

    :param tpdm: four-tensor representing the two-RDM
    :return: six-tensor reprsenting the three-RDM
    """
    opdm = (2 / (n_electrons - 1)) * np.einsum('ijjk', tpdm)
    if true_opdm is not None:
        assert np.allclose(opdm, true_opdm)

    unconnected_tpdm = of.wedge(opdm, opdm, (1, 1), (1, 1))
    unconnected_d3 = of.wedge(opdm, unconnected_tpdm, (1, 1), (2, 2))
    return 3 * of.wedge(tpdm, opdm, (2, 2), (1, 1)) - 2 * unconnected_d3


class SumOfSquaresOperator:

    def __init__(self,
                 basis_rotation: List[np.ndarray],
                 charge_charge_matrix: List[np.ndarray],
                 sdim: int,
                 one_body_rotation: np.ndarray,
                 one_body_generator=None):
        """
        A representation of two body operators expressed as sum of squares
        because they are all squares of normal operators.  The list is a
        sum of antihermitian operators that are to be implemented with the
        same evolution coefficient
        """
        self.basis_rotation = basis_rotation
        self.charge_charge = charge_charge_matrix
        self.sdim = sdim
        self.one_body_rotation = one_body_rotation
        self.one_body_generator = one_body_generator

    def time_evolve(self, wf):
        for v, cc in zip(self.basis_rotation, self.charge_charge):
            wf = evolve_fqe_givens_unrestricted(wf, v.conj().T)
            wf = evolve_fqe_charge_charge_unrestricted(wf, cc)
            wf = evolve_fqe_givens_unrestricted(wf, v)
        wf = evolve_fqe_givens_unrestricted(wf, self.one_body_rotation)
        return wf


class VBC:
    """Variational Brillouin Condition"""

    def __init__(self,
                 oei: np.ndarray,
                 tei: np.ndarray,
                 n_alpha: int,
                 n_beta: int,
                 iter_max=30,
                 verbose=True,
                 stopping_epsilon=1.0E-3,
                 delta_e_eps=1.0E-6):
        """
        Args:
            oei: one electron integrals in the spatial basis
            tei: two-electron integrals in the spatial basis
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
        self.stopping_eps = stopping_epsilon
        self.delta_e_eps = delta_e_eps

    def get_svd_tensor_decomp(self, residual,
                              update_rank) -> SumOfSquaresOperator:
        """
        Residual must be real
        """
        ul, vl, one_body_residual, _, _, one_body_op = \
            doubles_factorization_svd(residual,
                                      eig_cutoff=update_rank)
        # add back in tbe 1-body term after all sum-of-squares terms
        assert of.is_hermitian(1j * one_body_op)
        if not np.isclose((1j * one_body_op).induced_norm(), 0):
            # enforce symmetry in one-body sector
            one_body_residual[::2, ::2] = 0.5 * \
                                          (one_body_residual[::2,
                                           ::2] + one_body_residual[1::2, 1::2])
            one_body_residual[1::2, 1::2] = one_body_residual[::2, ::2]

        basis_list = []
        cc_list = []
        for ll in range(len(ul)):
            Smat = ul[ll] + vl[ll]
            Dmat = ul[ll] - vl[ll]
            op1mat = Smat + 1j * Smat.T
            op2mat = Smat - 1j * Smat.T
            op3mat = Dmat + 1j * Dmat.T
            op4mat = Dmat - 1j * Dmat.T

            w1, v1 = sp.linalg.schur(op1mat)
            w1 = np.diagonal(w1)
            oww1 = np.outer(w1, w1)
            basis_list.append(v1)
            cc_list.append((-1 / 16) * oww1.imag)

            w2, v2 = sp.linalg.schur(op2mat)
            w2 = np.diagonal(w2)
            oww2 = np.outer(w2, w2)
            basis_list.append(v2)
            cc_list.append((-1 / 16) * oww2.imag)

            w3, v3 = sp.linalg.schur(op3mat)
            w3 = np.diagonal(w3)
            oww3 = np.outer(w3, w3)
            basis_list.append(v3)
            cc_list.append((1 / 16) * oww3.imag)

            w4, v4 = sp.linalg.schur(op4mat)
            w4 = np.diagonal(w4)
            oww4 = np.outer(w4, w4)
            basis_list.append(v4)
            cc_list.append((1 / 16) * oww4.imag)

        sos_op = SumOfSquaresOperator(basis_rotation=basis_list,
                                      charge_charge_matrix=cc_list,
                                      sdim=self.sdim,
                                      one_body_rotation=expm(one_body_residual))
        return sos_op

    def get_takagi_tensor_decomp(self, residual, update_utc):
        one_body_residual = -np.einsum('pqrq->pr', residual)
        if not np.isclose(np.linalg.norm(one_body_residual), 0):
            # enforce symmetry in one-body sector
            one_body_residual[::2, ::2] = 0.5 * \
                                          (one_body_residual[::2,
                                           ::2] + one_body_residual[
                                                  1::2, 1::2])
            one_body_residual[1::2, 1::2] = one_body_residual[::2, ::2]

        Zlp, Zlm, _, one_body_residual = doubles_factorization_takagi(residual)

        basis_list = []
        cc_list = []
        if update_utc is None:
            update_utc = len(Zlp)

        for ll in range(update_utc):
            op1mat = Zlp[ll]
            op2mat = Zlm[ll]
            w1, v1 = sp.linalg.schur(op1mat)
            w1 = np.diagonal(w1)

            w2, v2 = sp.linalg.schur(op2mat)
            w2 = np.diagonal(w2)
            oww1 = np.outer(w1, w1)
            oww2 = np.outer(w2, w2)

            basis_list.append(v1)
            cc_list.append((-1 / 4) * oww1.imag)

            basis_list.append(v2)
            cc_list.append((-1 / 4) * oww2.imag)

        sos_op = SumOfSquaresOperator(basis_rotation=basis_list,
                                      charge_charge_matrix=cc_list,
                                      sdim=self.sdim,
                                      one_body_rotation=expm(one_body_residual))
        return sos_op

    def vbc(self,
            initial_wf: Wavefunction,
            opt_method: str = 'L-BFGS-B',
            opt_options=None,
            num_opt_var=None,
            v_reconstruct=False,
            generator_decomp=None,
            generator_rank=None):
        """The variational Brillouin condition method

        Solve for the 2-body residual and then variationally determine
        the step size.  This exact simulation cannot be implemented without
        Trotterization.  A proxy for the approximate evolution is the update_
        rank pameter which limites the rank of the residual.

        Args:
            initial_wf: initial wavefunction
            opt_method: scipy optimizer name
            num_opt_var: Number of optimization variables to consider
            v_reconstruct: use valdemoro reconstruction of 3-RDM to calculate
                           the residual
            generator_decomp: None, takagi, or svd
            generator_rank: number of generator terms to take
        """
        if opt_options is None:
            opt_options = {}
        self.num_opt_var = num_opt_var
        nso = 2 * self.sdim
        operator_pool: List[Union[ABCHamiltonian, SumOfSquaresOperator]] = []
        operator_pool_fqe: List[
            Union[ABCHamiltonian, SumOfSquaresOperator]] = []
        existing_parameters: List[float] = []
        self.energies = []
        self.energies = [initial_wf.expectationValue(self.k2_fop)]
        self.residuals = []
        iteration = 0
        while iteration < self.iter_max:
            # get current wavefunction
            wf = copy.deepcopy(initial_wf)
            for op, coeff in zip(operator_pool_fqe, existing_parameters):
                if np.isclose(coeff, 0):
                    continue
                if isinstance(op, ABCHamiltonian):
                    wf = wf.time_evolve(coeff, op)
                elif isinstance(op, SumOfSquaresOperator):
                    for v, cc in zip(op.basis_rotation, op.charge_charge):
                        wf = evolve_fqe_givens_unrestricted(wf, v.conj().T)
                        wf = evolve_fqe_charge_charge_unrestricted(
                            wf, coeff * cc)
                        wf = evolve_fqe_givens_unrestricted(wf, v)
                    wf = evolve_fqe_givens_unrestricted(wf,
                                                        op.one_body_rotation)
                else:
                    raise ValueError("Can't evolve operator type {}".format(
                        type(op)))

            # calculate rdms for grad
            _, tpdm = wf.sector((self.nele, self.sz)).get_openfermion_rdms()
            if v_reconstruct:
                d3 = 6 * valdemaro_reconstruction_functional(
                    tpdm / 2, self.nele)
            else:
                d3 = wf.sector((self.nele, self.sz)).get_three_pdm()

            # get ACSE Residual and 2-RDM gradient
            acse_residual = two_rdo_commutator_symm(
                self.reduced_ham.two_body_tensor, tpdm, d3)

            if generator_decomp is None:
                fop = get_fermion_op(acse_residual)
            elif generator_decomp is 'svd':
                new_residual = np.zeros_like(acse_residual)
                for p, q, r, s in product(range(nso), repeat=4):
                    new_residual[p, q, r, s] = (acse_residual[p, q, r, s] -
                                                acse_residual[s, r, q, p]) / 2

                fop = self.get_svd_tensor_decomp(new_residual, generator_rank)
            elif generator_decomp is 'takagi':
                fop = self.get_takagi_tensor_decomp(acse_residual,
                                                    generator_rank)
            else:
                raise ValueError(
                    "Generator decomp must be None, svd, or takagi")

            operator_pool.extend([fop])
            fqe_ops: List[Union[ABCHamiltonian, SumOfSquaresOperator]] = []
            if isinstance(fop, ABCHamiltonian):
                fqe_ops.append(fop)
            elif isinstance(fop, SumOfSquaresOperator):
                fqe_ops.append(fop)
            else:
                fqe_ops.append(
                    build_hamiltonian(1j * fop, self.sdim,
                                      conserve_number=True))

            operator_pool_fqe.extend(fqe_ops)
            existing_parameters.extend([0])

            if self.num_opt_var is not None:
                if len(operator_pool_fqe) < self.num_opt_var:
                    pool_to_op = operator_pool_fqe
                    params_to_op = existing_parameters
                    current_wf = copy.deepcopy(initial_wf)
                else:
                    pool_to_op = operator_pool_fqe[-self.num_opt_var:]
                    params_to_op = existing_parameters[-self.num_opt_var:]
                    current_wf = copy.deepcopy(initial_wf)
                    for fqe_op, coeff in zip(
                            operator_pool_fqe[:-self.num_opt_var],
                            existing_parameters[:-self.num_opt_var]):
                        current_wf = current_wf.time_evolve(coeff, fqe_op)
                    temp_cwf = copy.deepcopy(current_wf)
                    for fqe_op, coeff in zip(pool_to_op, params_to_op):
                        if np.isclose(coeff, 0):
                            continue
                        temp_cwf = temp_cwf.time_evolve(coeff, fqe_op)

                new_parameters, current_e = self.optimize_param(
                    pool_to_op,
                    params_to_op,
                    current_wf,
                    opt_method,
                    opt_options=opt_options)

                if len(operator_pool_fqe) < self.num_opt_var:
                    existing_parameters = new_parameters.tolist()
                else:
                    existing_parameters[-self.num_opt_var:] = \
                        new_parameters.tolist()
            else:
                new_parameters, current_e = self.optimize_param(
                    operator_pool_fqe,
                    existing_parameters,
                    initial_wf,
                    opt_method,
                    opt_options=opt_options)
                existing_parameters = new_parameters.tolist()

            if self.verbose:
                print(iteration, current_e, np.max(np.abs(acse_residual)),
                      len(existing_parameters))
            self.energies.append(current_e)
            self.residuals.append(acse_residual)
            if np.max(np.abs(acse_residual)) < self.stopping_eps or np.abs(
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
                    wf = wf.time_evolve(coeff, op)
                elif isinstance(op, SumOfSquaresOperator):
                    for v, cc in zip(op.basis_rotation, op.charge_charge):
                        wf = evolve_fqe_givens_unrestricted(wf, v.conj().T)
                        wf = evolve_fqe_charge_charge_unrestricted(
                            wf, coeff * cc)
                        wf = evolve_fqe_givens_unrestricted(wf, v)
                    wf = evolve_fqe_givens_unrestricted(wf,
                                                        op.one_body_rotation)
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
