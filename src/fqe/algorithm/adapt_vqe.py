"""Infrastructure for ADAPT VQE algorithm"""
from typing import List, Union
import copy
import openfermion as of
import fqe
from itertools import product
import numpy as np
import scipy as sp

from openfermion import (make_reduced_hamiltonian,
                         InteractionOperator, )
from openfermion.chem.molecular_data import spinorb_from_spatial

from fqe.hamiltonians.restricted_hamiltonian import RestrictedHamiltonian
from fqe.hamiltonians.general_hamiltonian import General as GeneralFQEHamiltonian
from fqe.hamiltonians.sparse_hamiltonian import SparseHamiltonian
from fqe.hamiltonians.hamiltonian import Hamiltonian as ABCHamiltonian
from fqe.fqe_decorators import build_hamiltonian
from fqe.algorithm.brillouin_calculator import (
    get_fermion_op,
    two_rdo_commutator_symm,
    one_rdo_commutator_symm,
)


# def get_fermion_op(coeff_tensor, ladder_order) -> of.FermionOperator:
#     """Returns an openfermion.FermionOperator from the given coeff_tensor.
#
#     Given A[i, j, k, l] of A = \sum_{ijkl}A[i, j, k, l]i^ j^ k^ l
#     return the FermionOperator A.
#
#     Args:
#         coeff_tensor: Coefficients for 4-mode operator
#     Returns:
#         A FermionOperator object
#     """
#     if len(coeff_tensor.shape) not in (2, 4):
#         raise ValueError(
#             "Arg `coeff_tensor` should have dimension 2 or 4 but has dimension"
#             f" {len(coeff_tensor.shape)}."
#         )
#
#     if len(coeff_tensor.shape) == 4:
#         nso = coeff_tensor.shape[0]
#         fermion_op = of.FermionOperator()
#         for p, q, r, s in product(range(nso), repeat=4):
#             op = ((p, ladder_order[0]), (q, ladder_order[1]), (r, ladder_order[2]), (s, ladder_order[3]))
#             fop = of.FermionOperator(op, coefficient=coeff_tensor[p, q, r, s])
#             fermion_op += fop
#         return fermion_op
#
#     if len(coeff_tensor.shape) == 2:
#         nso = coeff_tensor.shape[0]
#         fermion_op = of.FermionOperator()
#         for p, q in product(range(nso), repeat=2):
#             op = ((p, 1), (q, 0))
#             fop = of.FermionOperator(op, coefficient=coeff_tensor[p, q])
#             fermion_op += fop
#         return fermion_op

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
        self.op_pool = []

    def singlet_t2(self):
        """
        Generate singlet rotations
        T_{ij}^{ab} = T^(v1a, v2b)_{o1a, o2b} + T^(v1b, v2a)_{o1b, o2a} +
                      T^(v1a, v2a)_{o1a, o2a} + T^(v1b, v2b)_{o1b, o2b}

        where v1,v2 are indices of the virtual obritals and o1, o2 are
        indices of the occupied orbitals with respect to the Hartree-Fock
        reference.
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
    def __init__(self, oei: np.ndarray, tei: np.ndarray, operator_pool,
                 n_alpha: int, n_beta: int,
                 iter_max=30, verbose=True, stopping_epsilon=1.0E-3
                 ):
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
        elec_hamil = RestrictedHamiltonian(
            (oei, np.einsum("ijlk", -0.5 * tei))
        )
        soei, stei = spinorb_from_spatial(oei, tei)
        astei = np.einsum('ijkl', stei) - np.einsum('ijlk', stei)
        molecular_hamiltonian = InteractionOperator(
            0, soei, 0.25 * astei)

        reduced_ham = make_reduced_hamiltonian(molecular_hamiltonian,
                                               n_alpha + n_beta)
        self.reduced_ham = reduced_ham
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

    def vbc(self, initial_wf: fqe.Wavefunction, update_rank=None,
            opt_method: str='L-BFGS-B'):
        """The variational Brillouin condition method

        Solve for the 2-body residual and then variationally determine
        the step size.  This exact simulation cannot be implemented without
        Trotterization.  A proxy for the approximate evolution is the update_
        rank pameter which limites the rank of the residual.

        Args:
            initial_wf: initial wavefunction
            update_rank: rank of residual to truncate via first factorization
                         of the residual matrix <[Gamma_{lk}^{ij}, H]>
            opt_method: scipy optimizer name
        """
        nso = 2 * self.sdim
        operator_pool = []
        operator_pool_fqe = []
        existing_parameters = []
        self.energies = []
        self.residuals = []
        iteration = 0
        while iteration < self.iter_max:
            # get current wavefunction
            wf = copy.deepcopy(initial_wf)
            for fqe_op, coeff in zip(operator_pool_fqe, existing_parameters):
                # fqe_op = build_hamiltonian(1j * op, self.sdim,
                #                            conserve_number=True)
                wf = wf.time_evolve(coeff, fqe_op)

            # calculate rdms for grad
            opdm, tpdm = wf.sector((self.nele, self.sz)).get_openfermion_rdms()
            d3 = wf.sector((self.nele, self.sz)).get_three_pdm()
            # get ACSE Residual and 2-RDM gradient
            acse_residual = two_rdo_commutator_symm(
                self.reduced_ham.two_body_tensor, tpdm,
                d3)


            if update_rank:
                if update_rank % 2 != 0:
                    raise ValueError("Update rank must be an even number")

                new_residual = np.zeros_like(acse_residual)
                for p, q, r, s in product(range(nso), repeat=4):
                    assert np.isclose(acse_residual[p, q, r, s], -acse_residual[p, q, s, r])
                    assert np.isclose(acse_residual[p, q, r, s], -acse_residual[q, p, r, s])
                    assert np.isclose(acse_residual[p, q, r, s], -acse_residual[s, r, q, p].conj())
                    new_residual[p, q, r, s] = acse_residual[p, q, r, s] - acse_residual[s, r, q, p]

                zeroed_residual = np.zeros_like(new_residual)
                zeroed_residual_symm = np.zeros_like(new_residual)
                zeroed_residual_symmd = np.zeros_like(new_residual)
                for p, q, r, s in product(range(nso), repeat=4):
                    if p < q and s < r and p * nso + q < s * nso + r:
                        if not np.allclose([new_residual[p, q, r, s], new_residual[q, p, s, r], new_residual[p, q, s, r], new_residual[q, p, r, s]], 0):
                            # print(new_residual[p, q, r, s].real, new_residual[q, p, s, r].real, new_residual[p, q, s, r].real, new_residual[q, p, r, s].real)
                            assert np.isclose(new_residual[p, q, r, s].real, new_residual[q, p, s, r].real)
                            assert np.isclose(new_residual[p, q, s, r].real, new_residual[q, p, r, s].real)
                            assert np.isclose(new_residual[p, q, r, s].real, -new_residual[s, r, q, p].real)
                            zeroed_residual[p, q, r, s] = new_residual[p, q, r, s]
                            zeroed_residual_symm[p, q, r, s] = new_residual[p, q, r, s]
                            zeroed_residual_symm[q, p, s, r] = new_residual[q, p, s, r]
                for p, q, r, s in product(range(nso), repeat=4):
                    if p * nso + q < s * nso + r:
                        zeroed_residual_symmd[p, q, r, s] = new_residual[p, q, r, s]
                        # zeroed_residual_symmd[q, p, s, r] = new_residual[q, p, s, r]

                sparse_new_residual = np.reshape(np.transpose(zeroed_residual, [0, 3, 1, 2]),
                                              (nso**2, nso**2))
                sparse_new_residual_symm = np.reshape(np.transpose(zeroed_residual_symm, [0, 3, 1, 2]),
                                              (nso**2, nso**2))
                sparse_new_residual_symmd = np.reshape(np.transpose(zeroed_residual_symmd, [0, 3, 1, 2]),
                                              (nso**2, nso**2))

                new_residual_mat = np.reshape(np.transpose(new_residual, [0, 3, 1, 2]),
                                              (nso**2, nso**2)).astype(np.float)
                assert of.is_hermitian(new_residual_mat)

                residual_mat = np.reshape(np.transpose(acse_residual, [0, 3, 1, 2]),
                                          (nso**2, nso**2))
                for p, q, r, s in product(range(nso), repeat=4):
                    assert np.isclose(acse_residual[p, q, r, s], residual_mat[p * nso + s, q * nso + r])
                    assert np.isclose(acse_residual[q, p, s, r], residual_mat[q * nso + r, p * nso + s])
                    assert np.isclose(new_residual[p, q, r, s], new_residual_mat[p * nso + s, q * nso + r])
                    assert np.isclose(new_residual_mat[p * nso + s, q * nso + r], -new_residual_mat[s * nso + p, r * nso + q])

                    # commented out because this shouldn't be there.  We only have the upper triangle piece. We
                    # want to test if this is reconstructed as V^T @ U^T
                    # assert np.isclose(sparse_new_residual[p * nso + s, q * nso + r], -sparse_new_residual[s * nso + p, r * nso + q])
                    # if p > q and s > r and p * nso + q < s * nso + r:
                    #     assert np.isclose(sparse_new_residual[p * nso + s, q * nso + r], 0)
                    #     if not np.isclose(new_residual_mat[p * nso + s, q * nso + r], 0):
                    #         print(p, q, r, s, new_residual_mat[p * nso + s, q * nso + r], sparse_new_residual[p * nso + s, q * nso + r].real)

                    # these should be zero.  We know A[p, q, r, s] = -A[s, r, q, p] = T[(s,p), (r,q)]
                    # but we also construct the matrix such that p < q, s < r, and (p, q) < (s, r)
                    # thus we need the (s, r) > (p, q) which we will relate tho the SVD of the sparse symmetric matrix
                    # we take.
                    if p < q and s < r and p * nso + q < s * nso + r:
                        assert np.isclose(sparse_new_residual_symm[s * nso + p, r * nso + q], 0)

                    if p * nso + q < s * nso + r:
                        assert np.isclose(sparse_new_residual_symmd[s * nso + p , r * nso + q,], 0)

                assert np.allclose(new_residual_mat, new_residual_mat.T)
                # assert np.allclose(sparse_new_residual_symmd, sparse_new_residual_symmd.T)

                # exit()

                # one_body_residual = -np.einsum('pqrq->pr', acse_residual)
                one_body_residual = -np.einsum('pqrq->pr', new_residual) # zeroed_residual_symm)
                test_fop = of.FermionOperator()
                for p, q in product(range(nso), repeat=2):
                    fop = ((p, 1), (q, 0))
                    test_fop += of.FermionOperator(fop,
                                                  coefficient=one_body_residual[p, q])

                # u, sigma, vh = np.linalg.svd(residual_mat.real)
                # s, v = np.linalg.eigh(residual_mat.real)
                # u, sigma, vh = np.linalg.svd(ut_new_residual_mat)
                assert np.allclose(sparse_new_residual_symm, sparse_new_residual_symm.T)
                assert not np.allclose(sparse_new_residual, sparse_new_residual.T)
                u, sigma, vh = np.linalg.svd(new_residual_mat) # sparse_new_residual_symm)
                assert np.isclose(np.linalg.norm(vh.imag), 0)
                assert np.isclose(np.linalg.norm(u.imag), 0)

                assert np.allclose(sparse_new_residual_symm, sparse_new_residual_symm.T)

                us = u @ np.diag(sigma**0.5)
                svh = np.diag(sigma**0.5) @ vh
                # assert np.allclose(us @ svh, sparse_new_residual)
                # assert np.allclose(us @ svh + svh.T @ us.T, sparse_new_residual_symm)
                assert np.allclose(us @ svh + svh.T @ us.T, 2 * new_residual_mat)

                ul = []
                ul_ops = []
                vl = []
                vl_ops = []
                vhl_ops = []
                for ll in range(len(sigma)):
                    ul.append(np.sqrt(sigma[ll]) * u[:, ll].reshape((nso, nso)))
                    ul_ops.append(get_fermion_op(np.sqrt(sigma[ll]) * u[:, ll].reshape((nso, nso))))
                    vl.append(np.sqrt(sigma[ll]) * vh[ll, :].reshape((nso, nso)))
                    vl_ops.append(get_fermion_op(np.sqrt(sigma[ll]) * vh[ll, :].reshape((nso, nso))))
                    vhl_ops.append(get_fermion_op(np.sqrt(sigma[ll]) * vh[ll, :].reshape((nso, nso)).T))
                    # print(ul_ops[-1] - vl_ops[-1] + of.hermitian_conjugated(ul_ops[-1] - vl_ops[-1]))
                    S = ul_ops[-1] + vl_ops[-1]
                    D = ul_ops[-1] - vl_ops[-1]
                    op1 = S + 1j * of.hermitian_conjugated(S)
                    op2 = S - 1j * of.hermitian_conjugated(S)
                    op3 = D + 1j * of.hermitian_conjugated(D)
                    op4 = D - 1j * of.hermitian_conjugated(D)
                    assert np.isclose(of.normal_ordered(of.commutator(op1, of.hermitian_conjugated(op1))).induced_norm(), 0)
                    assert np.isclose(of.normal_ordered(of.commutator(op2, of.hermitian_conjugated(op2))).induced_norm(), 0)
                    assert np.isclose(of.normal_ordered(of.commutator(op3, of.hermitian_conjugated(op3))).induced_norm(), 0)
                    assert np.isclose(of.normal_ordered(of.commutator(op4, of.hermitian_conjugated(op4))).induced_norm(), 0)

                    # print()

                # move l-sum outisde
                for ll in range(len(sigma)):
                    # for p, s, q, r in product(range(nso), repeat=4):
                    #     test_fop += of.FermionOperator(((p, 1), (s, 0)), coefficient=ul[ll][p, s]) * of.FermionOperator(((q, 1), (r, 0)), coefficient=vl[ll][q, r])
                    test_fop += 0.25 * ul_ops[ll] * vl_ops[ll]
                    test_fop += 0.25 * vl_ops[ll] * ul_ops[ll]
                    # test_fop += vl_ops[ll] * ul_ops[ll]
                    test_fop += -0.25 * of.hermitian_conjugated(vl_ops[ll]) * of.hermitian_conjugated(ul_ops[ll])
                    test_fop += -0.25 * of.hermitian_conjugated(ul_ops[ll]) * of.hermitian_conjugated(vl_ops[ll])

                low_rank_residual_mat = np.zeros_like(residual_mat)
                for p, s, q, r in product(range(nso), repeat=4):
                    test_val = 0j
                    test_nr = 0j
                    for ll in range(len(sigma)):
                        # test_val += u[p * nso + s, ll] * sigma[ll] * vh[ll, q * nso + r]
                        test_val += ul[ll][p, s] * vl[ll][q, r]
                        if p < q and s < r:#  and p * nso + q < s * nso + r:
                            test_nr += ul[ll][p, s] * vl[ll][q, r]
                            # test_fop += of.FermionOperator(((p, 1), (s, 0), (q, 1), (r, 0)), coefficient=ul[ll][p, s] * vl[ll][q, r])

                        assert np.isclose(np.sqrt(sigma[ll]) * u[p * nso + s, ll], ul[ll][p, s])
                        assert np.isclose(np.sqrt(sigma[ll]) * vh[ll, q * nso + r], vl[ll][q, r])

                    # assert np.isclose(sparse_new_residual[p * nso + s, q * nso + r],
                    #                   test_val
                    #                   )
                    assert np.isclose(new_residual_mat[p * nso + s, q * nso + r],
                                      test_val
                                      )
                    if p < q and s < r:#  and p * nso + q < s * nso + r:
                        assert np.isclose(new_residual_mat[p * nso + s, q * nso + r],
                                          test_nr
                                          )

                    lr_val = 0j
                    for ll in range(update_rank):
                        lr_val += u[p * nso + s, ll] * sigma[ll] * vh[ll, q * nso + r]
                    low_rank_residual_mat[p * nso + s, q * nso + r] = lr_val

                # true_fop = get_fermion_op(acse_residual)
                # true_fop = get_fermion_op(zeroed_residual_symm)
                true_fop = get_fermion_op(new_residual)
                print(of.normal_ordered(test_fop - true_fop).induced_norm())
                exit()

                low_rank_residual = np.reshape(low_rank_residual_mat, (nso, nso, nso, nso))
                tacse_residual = np.transpose(low_rank_residual, [0, 2, 3, 1])
                # assert np.allclose(tacse_residual, acse_residual)
                acse_residual = tacse_residual
                # for p, q, r, s in product(range(nso), repeat=4):
                for um, vm in zip(ul_ops, vhl_ops):
                    print(of.normal_ordered(um * vm + vm * um))
                    print()


                exit()

                # residual_mat = acse_residual.transpose(0, 1, 3, 2).reshape(
                #     (nso ** 2, nso ** 2))

                # hrm = 1j * residual_mat
                # w, v = np.linalg.eigh(hrm)
                # # sort by absolute algebraic size
                # idx = np.argsort(np.abs(w))[::-1]
                # w = w[idx]
                # v = v[:, idx]
                # residual_reconstructed = np.zeros_like(residual_mat)
                # for ii in range(update_rank):
                #     residual_reconstructed += w[ii] * v[:, [ii]] @ \
                #                               v[:, [ii]].conj().T
                # acse_residual = -1j * residual_reconstructed.reshape(
                #     (nso, nso, nso, nso)).transpose(0, 1, 3, 2)

            fop = get_fermion_op(acse_residual)
            operator_pool.append(fop)
            fqe_op = build_hamiltonian(1j * fop, self.sdim,
                                       conserve_number=True)
            operator_pool_fqe.append(fqe_op)
            existing_parameters.append(0)

            new_parameters, current_e = self.optimize_param(operator_pool_fqe,
                                                 existing_parameters,
                                                 initial_wf, opt_method)
            existing_parameters = new_parameters.tolist()
            if self.verbose:
                print(iteration, current_e, np.linalg.norm(acse_residual))
            self.energies.append(current_e)
            self.residuals.append(acse_residual)
            if np.linalg.norm(acse_residual) < self.stopping_eps:
                break
            iteration += 1

    def adapt_vqe(self, initial_wf: fqe.Wavefunction,
                  opt_method: str='L-BFGS-B'):
        """
        Run ADAPT-VQE using

        Args:
            initial_wf: Initial wavefunction at the start of the calculation
            opt_method: scipy optimizer to use
        """
        operator_pool = []
        operator_pool_fqe = []
        existing_parameters = []
        self.gradients = []
        self.energies = []
        iteration = 0
        while iteration < self.iter_max:
            # get current wavefunction
            wf = copy.deepcopy(initial_wf)
            for fqe_op, coeff in zip(operator_pool_fqe, existing_parameters):
                wf = wf.time_evolve(coeff, fqe_op)

            # calculate rdms for grad
            opdm, tpdm = wf.sector((self.nele, self.sz)).get_openfermion_rdms()
            d3 = wf.sector((self.nele, self.sz)).get_three_pdm()
            # get ACSE Residual and 2-RDM gradient
            acse_residual = two_rdo_commutator_symm(
                self.reduced_ham.two_body_tensor, tpdm,
                d3)
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

            max_grad_term_idx = np.argmax(np.abs(pool_grad))
            operator_pool.append(self.operator_pool.op_pool[max_grad_term_idx])
            fqe_op = build_hamiltonian(
                1j * self.operator_pool.op_pool[max_grad_term_idx], self.sdim,
                conserve_number=True)
            operator_pool_fqe.append(fqe_op)
            existing_parameters.append(0)

            new_parameters, current_e = self.optimize_param(operator_pool_fqe,
                                                 existing_parameters,
                                                 initial_wf, opt_method)
            existing_parameters = new_parameters.tolist()
            if self.verbose:
                print(iteration, current_e, max(np.abs(pool_grad)))
            self.energies.append(current_e)
            self.gradients.append(pool_grad)
            if max(np.abs(pool_grad)) < self.stopping_eps:
                break
            iteration += 1

    def optimize_param(self, pool: Union[
        List[of.FermionOperator], List[GeneralFQEHamiltonian]],
                       existing_params: Union[List, np.ndarray],
                       initial_wf: fqe.Wavefunction,
                       opt_method: str) -> fqe.wavefunction:
        """Optimize a wavefunction given a list of generators

        Args:
            pool: generators of rotation
            existing_params: parameters for the generators
            initial_wf: initial wavefunction
            opt_method: Scpy.optimize method
        """
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
                    fqe_op = build_hamiltonian(1j * op, self.sdim,
                                               conserve_number=True)
                wf = wf.time_evolve(coeff, fqe_op)

            # compute gradients
            grad_vec = np.zeros(len(params), dtype=np.complex128)
            for pidx, p in enumerate(params):
                # evolve e^{iG_{n-1}g_{n-1}}e^{iG_{n-2}g_{n-2}}G_{n-3}e^{-G_{n-3}g_{n-3}...|0>
                grad_wf = copy.deepcopy(initial_wf)
                for gidx, (op, coeff) in enumerate(zip(pool, params)):
                    if isinstance(op, ABCHamiltonian):
                        fqe_op = op
                    else:
                        fqe_op = build_hamiltonian(1j * op, self.sdim,
                                                   conserve_number=True)
                    grad_wf = grad_wf.time_evolve(coeff, fqe_op)
                    # if looking at the pth parameter then apply the operator
                    # to the state
                    if gidx == pidx:
                        grad_wf = grad_wf.apply(fqe_op)

                grad_val = grad_wf.expectationValue(self.elec_hamil, brawfn=wf)

                grad_vec[pidx] = -1j * grad_val + 1j * grad_val.conj()
                assert np.isclose(grad_vec[pidx].imag, 0)
            return wf.expectationValue(self.elec_hamil).real, np.array(
                grad_vec.real, order='F')

        res = sp.optimize.minimize(cost_func, existing_params,
                                   method=opt_method, jac=True)
        return res.x, res.fun
