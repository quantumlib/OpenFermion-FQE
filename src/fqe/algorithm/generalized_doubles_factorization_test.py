from itertools import product
import numpy as np
import scipy as sp
import copy

import fqe
from fqe.algorithm.generalized_doubles_factorization import (
    doubles_factorization, doubles_factorization2, takagi)
from fqe.algorithm.adapt_vqe import ADAPT
from fqe.algorithm.brillouin_calculator import two_rdo_commutator_symm
from fqe.algorithm.brillouin_calculator import get_fermion_op
from fqe.algorithm.low_rank import (evolve_fqe_givens_unrestricted,
                                    evolve_fqe_charge_charge_unrestricted, )
from fqe.unittest_data.build_lih_data import build_lih_data
from fqe.fqe_decorators import build_hamiltonian

from fqe.unittest_data.generate_openfermion_molecule import \
    build_lih_moleculardata
import openfermion as of

import time


def generate_antisymm_generator(nso):
    A = np.zeros(tuple([nso] * 4))
    for p, q, r, s in product(range(nso), repeat=4):
        if p < q and s < r  and p * nso + q < s * nso + r:
            A[p, q, r, s] = np.random.random()
            A[p, q, s, r] = -A[p, q, r, s]
            A[q, p, r, s] = -A[p, q, r, s]
            A[q, p, s, r] = A[p, q, r, s]
            A[s, r, q, p] = -A[p, q, r, s]
            A[r, s, q, p] = A[p, q, r, s]
            A[s, r, p, q] = A[p, q, r, s]
            A[r, s, p, q] = -A[p, q, r, s]
    return A


def test_generalized_doubles():
    generator = generate_antisymm_generator(6)
    nso = generator.shape[0]
    for p, q, r, s in product(range(nso), repeat=4):
        if p < q and s < r:
            assert np.isclose(generator[p, q, r, s], -generator[q, p, r, s])

    ul, vl, one_body_residual, ul_ops, vl_ops, one_body_op = \
        doubles_factorization(generator)

    generator_mat = np.reshape(np.transpose(generator, [0, 3, 1, 2]),
                               (nso ** 2, nso ** 2)).astype(np.float)
    one_body_residual_test = -np.einsum('pqrq->pr',
                                   generator)
    assert np.allclose(generator_mat, generator_mat.T)
    assert np.allclose(one_body_residual, one_body_residual_test)

    tgenerator_mat = np.zeros_like(generator_mat)
    for row_gem, col_gem in product(range(nso ** 2), repeat=2):
        p, s = row_gem // nso, row_gem % nso
        q, r = col_gem // nso, col_gem % nso
        tgenerator_mat[row_gem, col_gem] = generator[p, q, r, s]

    assert np.allclose(tgenerator_mat, generator_mat)

    u, sigma, vh = np.linalg.svd(generator_mat)

    fop = copy.deepcopy(one_body_op)
    fop2 = copy.deepcopy(one_body_op)
    fop3 = copy.deepcopy(one_body_op)
    fop4 = copy.deepcopy(one_body_op)
    for ll in range(len(sigma)):
        ul.append(np.sqrt(sigma[ll]) * u[:, ll].reshape((nso, nso)))
        ul_ops.append(
            get_fermion_op(np.sqrt(sigma[ll]) * u[:, ll].reshape((nso, nso))))
        vl.append(np.sqrt(sigma[ll]) * vh[ll, :].reshape((nso, nso)))
        vl_ops.append(
            get_fermion_op(np.sqrt(sigma[ll]) * vh[ll, :].reshape((nso, nso))))
        Smat = ul[ll] + vl[ll]
        Dmat = ul[ll] - vl[ll]

        S = ul_ops[ll] + vl_ops[ll]
        Sd = of.hermitian_conjugated(S)
        D = ul_ops[ll] - vl_ops[ll]
        Dd = of.hermitian_conjugated(D)
        op1 = S + 1j * of.hermitian_conjugated(S)
        op2 = S - 1j * of.hermitian_conjugated(S)
        op3 = D + 1j * of.hermitian_conjugated(D)
        op4 = D - 1j * of.hermitian_conjugated(D)
        assert np.isclose(of.normal_ordered(
            of.commutator(op1, of.hermitian_conjugated(op1))).induced_norm(), 0)
        assert np.isclose(of.normal_ordered(
            of.commutator(op2, of.hermitian_conjugated(op2))).induced_norm(), 0)
        assert np.isclose(of.normal_ordered(
            of.commutator(op3, of.hermitian_conjugated(op3))).induced_norm(), 0)
        assert np.isclose(of.normal_ordered(
            of.commutator(op4, of.hermitian_conjugated(op4))).induced_norm(), 0)

        fop3 += (1/8) * ((S**2 - Sd**2) - (D**2 - Dd**2))
        fop4 += (1/16) * ((op1**2 + op2**2) - (op3**2 + op4**2))

        op1mat = Smat + 1j * Smat.T
        op2mat = Smat - 1j * Smat.T
        op3mat = Dmat + 1j * Dmat.T
        op4mat = Dmat - 1j * Dmat.T

        assert np.allclose(of.commutator(op1mat, op1mat.conj().T), 0)
        assert np.allclose(of.commutator(op2mat, op2mat.conj().T), 0)
        assert np.allclose(of.commutator(op3mat, op3mat.conj().T), 0)
        assert np.allclose(of.commutator(op4mat, op4mat.conj().T), 0)

        # check that we have normal operators and that the outer product
        # of their eigenvalues is imaginary. Also check vv is unitary
        if not np.isclose(sigma[ll], 0):
            assert np.isclose(of.normal_ordered(get_fermion_op(op1mat) - op1).induced_norm(), 0)
            assert np.isclose(of.normal_ordered(get_fermion_op(op2mat) - op2).induced_norm(), 0)
            assert np.isclose(of.normal_ordered(get_fermion_op(op3mat) - op3).induced_norm(), 0)
            assert np.isclose(of.normal_ordered(get_fermion_op(op4mat) - op4).induced_norm(), 0)

            ww, vv = np.linalg.eig(op1mat)
            eye = np.eye(nso)
            assert np.allclose(np.outer(ww, ww).real, 0)
            assert np.allclose(vv.conj().T @ vv, eye)
            ww, vv = np.linalg.eig(op2mat)
            assert np.allclose(np.outer(ww, ww).real, 0)
            assert np.allclose(vv.conj().T @ vv, eye)
            ww, vv = np.linalg.eig(op3mat)
            assert np.allclose(np.outer(ww, ww).real, 0)
            assert np.allclose(vv.conj().T @ vv, eye)
            ww, vv = np.linalg.eig(op4mat)
            assert np.allclose(np.outer(ww, ww).real, 0)
            assert np.allclose(vv.conj().T @ vv, eye)

        fop2 += 0.25 * ul_ops[ll] * vl_ops[ll]
        fop2 += 0.25 * vl_ops[ll] * ul_ops[ll]
        fop2 += -0.25 * of.hermitian_conjugated(
            vl_ops[ll]) * of.hermitian_conjugated(ul_ops[ll])
        fop2 += -0.25 * of.hermitian_conjugated(
            ul_ops[ll]) * of.hermitian_conjugated(vl_ops[ll])

        fop += vl_ops[ll] * ul_ops[ll]

    true_fop = get_fermion_op(generator)
    assert np.isclose(of.normal_ordered(fop - true_fop).induced_norm(), 0)
    assert np.isclose(of.normal_ordered(fop2 - true_fop).induced_norm(), 0)
    assert np.isclose(of.normal_ordered(fop3 - true_fop).induced_norm(), 0)
    assert np.isclose(of.normal_ordered(fop4 - true_fop).induced_norm(), 0)


def test_random_evolution():
    sdim = 4
    nele = 4
    generator = generate_antisymm_generator(2 * sdim)
    nso = generator.shape[0]
    for p, q, r, s in product(range(nso), repeat=4):
        if p < q and s < r:
            assert np.isclose(generator[p, q, r, s], -generator[q, p, r, s])

    generator_mat = np.reshape(np.transpose(generator, [0, 3, 1, 2]),
                               (nso ** 2, nso ** 2)).astype(np.float)
    u, sigma, vh = np.linalg.svd(generator_mat)

    ul, vl, one_body_residual, ul_ops, vl_ops, one_body_op = \
        doubles_factorization(generator)

    rwf = fqe.get_number_conserving_wavefunction(nele, sdim)
    # rwf = fqe.Wavefunction([[nele, 0, sdim]])
    rwf.set_wfn(strategy='random')
    rwf.normalize()

    sigma_idx = np.where(sigma > 1.0E-13)[0]
    for ll in sigma_idx:
        Smat = ul[ll] + vl[ll]
        Dmat = ul[ll] - vl[ll]

        S = ul_ops[ll] + vl_ops[ll]
        D = ul_ops[ll] - vl_ops[ll]
        op1 = S + 1j * of.hermitian_conjugated(S)
        op2 = S - 1j * of.hermitian_conjugated(S)
        op3 = D + 1j * of.hermitian_conjugated(D)
        op4 = D - 1j * of.hermitian_conjugated(D)

        # the ops are normal
        assert np.isclose(of.normal_ordered(
            of.commutator(op1, of.hermitian_conjugated(op1))).induced_norm(), 0)
        assert np.isclose(of.normal_ordered(
            of.commutator(op2, of.hermitian_conjugated(op2))).induced_norm(), 0)
        assert np.isclose(of.normal_ordered(
            of.commutator(op3, of.hermitian_conjugated(op3))).induced_norm(), 0)
        assert np.isclose(of.normal_ordered(
            of.commutator(op4, of.hermitian_conjugated(op4))).induced_norm(), 0)

        # confirm that evolution under these Hermitian operators is the same
        # as the U^ D U form of the evolution
        assert of.is_hermitian(1j * op1**2)
        assert of.is_hermitian(1j * op2**2)
        assert of.is_hermitian(1j * op3**2)
        assert of.is_hermitian(1j * op4**2)

        op1mat = Smat + 1j * Smat.T
        op2mat = Smat - 1j * Smat.T
        op3mat = Dmat + 1j * Dmat.T
        op4mat = Dmat - 1j * Dmat.T

        o1_rwf = rwf.time_evolve(1/16, 1j * op1**2)
        ww, vv = np.linalg.eig(op1mat)
        assert np.allclose(vv @ np.diag(ww) @ vv.conj().T, op1mat)
        trwf = evolve_fqe_givens_unrestricted(rwf, vv.conj().T)
        v_pq = np.outer(ww, ww)
        for p, q in product(range(nso), repeat=2):
            fop = of.FermionOperator(((p, 1), (p, 0), (q, 1), (q, 0)),
                                     coefficient=-v_pq[p, q].imag)
            trwf = trwf.time_evolve(1/16, fop)
        trwf = evolve_fqe_givens_unrestricted(trwf, vv)
        assert np.isclose(fqe.vdot(o1_rwf, trwf), 1)

        o_rwf = rwf.time_evolve(1/16, 1j * op2**2)
        ww, vv = np.linalg.eig(op2mat)
        assert np.allclose(vv @ np.diag(ww) @ vv.conj().T, op2mat)
        trwf = evolve_fqe_givens_unrestricted(rwf, vv.conj().T)
        v_pq = np.outer(ww, ww)
        for p, q in product(range(nso), repeat=2):
            fop = of.FermionOperator(((p, 1), (p, 0), (q, 1), (q, 0)),
                                     coefficient=-v_pq[p, q].imag)
            trwf = trwf.time_evolve(1/16, fop)
        trwf = evolve_fqe_givens_unrestricted(trwf, vv)
        assert np.isclose(fqe.vdot(o_rwf, trwf), 1)

        o_rwf = rwf.time_evolve(-1/16, 1j * op3**2)
        ww, vv = np.linalg.eig(op3mat)
        assert np.allclose(vv @ np.diag(ww) @ vv.conj().T, op3mat)
        trwf = evolve_fqe_givens_unrestricted(rwf, vv.conj().T)
        v_pq = np.outer(ww, ww)
        for p, q in product(range(nso), repeat=2):
            fop = of.FermionOperator(((p, 1), (p, 0), (q, 1), (q, 0)),
                                     coefficient=-v_pq[p, q].imag)
            trwf = trwf.time_evolve(-1/16, fop)
        trwf = evolve_fqe_givens_unrestricted(trwf, vv)
        assert np.isclose(fqe.vdot(o_rwf, trwf), 1)

        o_rwf = rwf.time_evolve(-1/16, 1j * op4**2)
        ww, vv = np.linalg.eig(op4mat)
        assert np.allclose(vv @ np.diag(ww) @ vv.conj().T, op4mat)
        trwf = evolve_fqe_givens_unrestricted(rwf, vv.conj().T)
        v_pq = np.outer(ww, ww)
        for p, q in product(range(nso), repeat=2):
            fop = of.FermionOperator(((p, 1), (p, 0), (q, 1), (q, 0)),
                                     coefficient=-v_pq[p, q].imag)
            trwf = trwf.time_evolve(-1/16, fop)
        trwf = evolve_fqe_givens_unrestricted(trwf, vv)
        assert np.isclose(fqe.vdot(o_rwf, trwf), 1)




def test_trotter_error():
    sdim = 4
    nele = sdim
    generator = generate_antisymm_generator(2 * sdim)
    fop_generator = of.normal_ordered(get_fermion_op(generator))
    assert of.is_hermitian(1j * fop_generator)

    rwf = fqe.get_number_conserving_wavefunction(nele, sdim)
    # rwf = fqe.Wavefunction([[nele, 0, sdim]])
    rwf.set_wfn(strategy='random')
    rwf.normalize()
    # rwf.print_wfn()

    evolve_time = 0.1
    fqe_fop = build_hamiltonian(1j * fop_generator, norb=sdim,
                                conserve_number=True)
    start_time = time.time()
    final_rwf = rwf.apply_generated_unitary(evolve_time, 'taylor', fqe_fop,
                                            expansion=160)
    end_time = time.time()
    print("Exact evolution time ", end_time - start_time)

    nso = generator.shape[0]
    # for p, q, r, s in product(range(nso), repeat=4):
    #     if p < q and s < r:
    #         assert np.isclose(generator[p, q, r, s], -generator[q, p, r, s])

    generator_mat = np.reshape(np.transpose(generator, [0, 3, 1, 2]),
                               (nso ** 2, nso ** 2)).astype(np.float)
    start_time = time.time()
    u, sigma, vh = np.linalg.svd(generator_mat)
    end_time = time.time()
    print("svd generator_mat time ", end_time - start_time)

    start_time = time.time()
    ul, vl, one_body_residual, ul_ops, vl_ops, one_body_op = \
        doubles_factorization(generator)
    end_time = time.time()
    print("factorization time ", end_time - start_time)

    trotter_wf = copy.deepcopy(rwf)
    num_trotter_steps = 4
    uu = sp.linalg.expm(evolve_time * one_body_residual / num_trotter_steps)
    for _ in range(num_trotter_steps):
        # one-body evolution
        # ob_fop = build_hamiltonian(1j * one_body_op)
        # test_wf = rwf.time_evolve(evolve_time/num_trotter_steps, ob_fop)
        start_time = time.time()
        trotter_wf = evolve_fqe_givens_unrestricted(trotter_wf, uu)
        # assert np.isclose(fqe.vdot(trotter_wf, test_wf), 1)

        sigma_idx = np.where(sigma > 1.0E-13)[0]
        for ll in sigma_idx[:2]:
            Smat = ul[ll] + vl[ll]
            Dmat = ul[ll] - vl[ll]
            op1mat = Smat + 1j * Smat.T
            op2mat = Smat - 1j * Smat.T
            op3mat = Dmat + 1j * Dmat.T
            op4mat = Dmat - 1j * Dmat.T

            # now evolve by each operator
            ww, vv = np.linalg.eig(op1mat)
            assert np.allclose(vv @ np.diag(ww) @ vv.conj().T, op1mat)
            ss = time.time()
            trotter_wf = evolve_fqe_givens_unrestricted(trotter_wf, vv.conj().T)
            ee = time.time()
            print("givens ", ee - ss)
            v_pq = np.outer(ww, ww)
            print(v_pq)
            ss = time.time()
            trotter_wf = evolve_fqe_charge_charge_unrestricted(trotter_wf, -v_pq.imag,
                                                         evolve_time/(16 * num_trotter_steps))
            ee = time.time()
            print("charge charge ", ee - ss)
            ss = time.time()
            trotter_wf = evolve_fqe_givens_unrestricted(trotter_wf, vv)
            ee = time.time()
            print("givens ", ee - ss)
            print()

            # ww, vv = np.linalg.eig(op2mat)
            # assert np.allclose(vv @ np.diag(ww) @ vv.conj().T, op2mat)
            # trotter_wf = evolve_fqe_givens_unrestricted(trotter_wf, vv.conj().T)
            # v_pq = np.outer(ww, ww)
            # trotter_wf = evolve_fqe_charge_charge_unrestricted(trotter_wf, -v_pq.imag,
            #                                              evolve_time/(16 * num_trotter_steps))
            # trotter_wf = evolve_fqe_givens_unrestricted(trotter_wf, vv)

            ww, vv = np.linalg.eig(op3mat)
            assert np.allclose(vv @ np.diag(ww) @ vv.conj().T, op3mat)
            trotter_wf = evolve_fqe_givens_unrestricted(trotter_wf, vv.conj().T)
            v_pq = np.outer(ww, ww)
            trotter_wf = evolve_fqe_charge_charge_unrestricted(trotter_wf, -v_pq.imag, -evolve_time/(16 * num_trotter_steps))
            trotter_wf = evolve_fqe_givens_unrestricted(trotter_wf, vv)

            # ww, vv = np.linalg.eig(op4mat)
            # assert np.allclose(vv @ np.diag(ww) @ vv.conj().T, op4mat)
            # trotter_wf = evolve_fqe_givens_unrestricted(trotter_wf, vv.conj().T)
            # v_pq = np.outer(ww, ww)
            # trotter_wf = evolve_fqe_charge_charge_unrestricted(trotter_wf, -v_pq.imag, -evolve_time/(16 * num_trotter_steps))
            # trotter_wf = evolve_fqe_givens_unrestricted(trotter_wf, vv)

        end_time = time.time()
        print("Single Trotter step ", end_time - start_time)

    print(abs(fqe.vdot(trotter_wf, final_rwf)))


def test_reconstruction_error():
    sdim = 4
    nele = sdim
    generator = generate_antisymm_generator(2 * sdim)
    fop_generator = of.normal_ordered(get_fermion_op(generator))
    assert of.is_hermitian(1j * fop_generator)
    nso = generator.shape[0]
    generator_mat = np.reshape(np.transpose(generator, [0, 3, 1, 2]),
                               (nso ** 2, nso ** 2)).astype(np.float)
    start_time = time.time()
    u, sigma, vh = np.linalg.svd(generator_mat)
    end_time = time.time()
    print("svd generator_mat time ", end_time - start_time)

    start_time = time.time()
    ul, vl, one_body_residual, ul_ops, vl_ops, one_body_op = \
        doubles_factorization(generator)
    end_time = time.time()
    print("factorization time ", end_time - start_time)


    sigma_idx = np.where(sigma > 1.0E-13)[0]
    for ll in sigma_idx:
        Smat = ul[ll] + vl[ll]
        Dmat = ul[ll] - vl[ll]
        op1mat = Smat + 1j * Smat.T
        op2mat = Smat - 1j * Smat.T
        op3mat = Dmat + 1j * Dmat.T
        op4mat = Dmat - 1j * Dmat.T

        # now evolve by each operator
        ww, vv = np.linalg.eig(op1mat)
        assert np.allclose(vv @ np.diag(ww) @ vv.conj().T, op1mat)
        print(vv)
        exit()


def test_normal_op_tensor_reconstruction():
    sdim = 2
    generator = generate_antisymm_generator(2 * sdim)
    nso = generator.shape[0]
    for p, q, r, s in product(range(nso), repeat=4):
        if p < q and s < r:
            assert np.isclose(generator[p, q, r, s], -generator[q, p, r, s])

    generator_mat = np.reshape(np.transpose(generator, [0, 3, 1, 2]),
                               (nso ** 2, nso ** 2)).astype(np.float)
    u, sigma, vh = np.linalg.svd(generator_mat)

    ul, vl, one_body_residual, ul_ops, vl_ops, one_body_op = \
        doubles_factorization(generator)

    sigma_idx = np.where(sigma > 1.0E-13)[0]
    test_generator_mat = np.zeros_like(generator_mat)
    for p, q, r, s in product(range(nso), repeat=4):
        for ll in sigma_idx:
            Smat = ul[ll] + vl[ll]
            Dmat = ul[ll] - vl[ll]

            op1mat = Smat + 1j * Smat.T
            op2mat = Smat - 1j * Smat.T
            op3mat = Dmat + 1j * Dmat.T
            op4mat = Dmat - 1j * Dmat.T
            test_generator_mat[p * nso + s, q * nso + r] += (1 / 16) * \
                                                            (op1mat[p, s] *
                                                             op1mat[q, r] +
                                                             op2mat[p, s] *
                                                             op2mat[q, r] -
                                                             op3mat[p, s] *
                                                             op3mat[q, r] -
                                                             op4mat[p, s] *
                                                             op4mat[q, r])

    assert np.allclose(test_generator_mat, generator_mat)

    test_generator = np.zeros_like(generator).astype(np.complex128)
    for ll in sigma_idx:
        Smat = ul[ll] + vl[ll]
        Dmat = ul[ll] - vl[ll]

        op1mat = Smat + 1j * Smat.T
        op2mat = Smat - 1j * Smat.T
        op3mat = Dmat + 1j * Dmat.T
        op4mat = Dmat - 1j * Dmat.T

        test_op1 = np.zeros((nso, nso, nso, nso), dtype=np.complex128)
        test_op2 = np.zeros((nso, nso, nso, nso), dtype=np.complex128)
        test_op3 = np.zeros((nso, nso, nso, nso), dtype=np.complex128)
        test_op4 = np.zeros((nso, nso, nso, nso), dtype=np.complex128)
        for p, q, r, s in product(range(nso), repeat=4):
            test_op1[p, q, r, s] += op1mat[p, s] * op1mat[q, r]
            test_op2[p, q, r, s] += op2mat[p, s] * op2mat[q, r]
            test_op3[p, q, r, s] -= op3mat[p, s] * op3mat[q, r]
            test_op4[p, q, r, s] -= op4mat[p, s] * op4mat[q, r]
        test_op1 *= (1 / 16)
        test_op2 *= (1 / 16)
        test_op3 *= (1 / 16)
        test_op4 *= (1 / 16)
        assert of.is_hermitian(1j * test_op1)
        assert of.is_hermitian(1j * test_op2)
        assert of.is_hermitian(1j * test_op3)
        assert of.is_hermitian(1j * test_op4)
        test_generator += test_op1 + test_op2 + test_op3 + test_op4

    assert np.allclose(test_generator, generator)
    print("PASSED")

    test_generator = np.zeros_like(generator).astype(np.complex128)
    for ll in sigma_idx:
        Smat = ul[ll] + vl[ll]
        Dmat = ul[ll] - vl[ll]

        op1mat = Smat + 1j * Smat.T
        op2mat = Smat - 1j * Smat.T
        op3mat = Dmat + 1j * Dmat.T
        op4mat = Dmat - 1j * Dmat.T

        w1, v1 = np.linalg.eig(op1mat)
        assert np.allclose(v1 @ np.diag(w1) @ v1.conj().T, op1mat)
        v1c = v1.conj()
        for m, n in product(range(op1mat.shape[0]), repeat=2):
            assert np.isclose(op1mat[m, n],
                              v1[m, :].dot(np.diag(w1)).dot(v1c[n, :]))
        w2, v2 = np.linalg.eig(op2mat)
        assert np.allclose(v2 @ np.diag(w2) @ v2.conj().T, op2mat)
        v2c = v2.conj()
        for m, n in product(range(op2mat.shape[0]), repeat=2):
            assert np.isclose(op2mat[m, n],
                              np.einsum('j,j,j', v2[m, :], w2, v2c[n, :]))

        w3, v3 = np.linalg.eig(op3mat)
        v3c = v3.conj()
        assert np.allclose(v3 @ np.diag(w3) @ v3.conj().T, op3mat)
        w4, v4 = np.linalg.eig(op4mat)
        v4c = v4.conj()
        assert np.allclose(v4 @ np.diag(w4) @ v4.conj().T, op4mat)

        test_op1 = np.zeros((nso, nso, nso, nso), dtype=np.complex128)
        test_op2 = np.zeros((nso, nso, nso, nso), dtype=np.complex128)
        test_op3 = np.zeros((nso, nso, nso, nso), dtype=np.complex128)
        test_op4 = np.zeros((nso, nso, nso, nso), dtype=np.complex128)
        oww1 = np.outer(w1, w1)
        oww2 = np.outer(w2, w2)
        oww3 = np.outer(w3, w3)
        oww4 = np.outer(w4, w4)
        # print((-1j * oww1).real)
        # print((-1j * oww3).real)
        # print(v1)
        # print(v2)
        # print(v3)
        # print(v4)
        print(sp.linalg.logm(v1))
        print(sp.linalg.logm(v2))
        print(sp.linalg.logm(v3))
        print(sp.linalg.logm(v4))
        assert np.allclose(v1, v2.conj())
        assert np.allclose(v1, v3.conj())
        assert np.allclose(v1, v4)
        assert np.allclose(oww1, -oww2)
        assert np.allclose(oww3, -oww4)
        print()
        print(oww2)
        print(oww4)
        print()
        print()

        for p, q, r, s in product(range(nso), repeat=4):
            test_op1[p, q, r, s] += op1mat[p, s] * op1mat[q, r]
            assert np.isclose(op1mat[p, s] * op1mat[q, r],
                              np.einsum('i,i,ij,j,j', v1[p, :], v1c[s, :], oww1,
                                        v1[q, :], v1c[r, :]))
            test_op2[p, q, r, s] += op2mat[p, s] * op2mat[q, r]
            test_op3[p, q, r, s] -= op3mat[p, s] * op3mat[q, r]
            test_op4[p, q, r, s] -= op4mat[p, s] * op4mat[q, r]

        assert np.allclose(np.einsum('pi,si,ij,qj,rj->pqrs', v1, v1c, oww1, v1, v1c),
                           test_op1)
        assert np.allclose(
            np.einsum('pi,si,ij,qj,rj->pqrs', v2, v2c, oww2, v2, v2c),
            test_op2)
        assert np.allclose(
            np.einsum('pi,si,ij,qj,rj->pqrs', v3, v3c, -oww3, v3, v3c),
            test_op3)
        assert np.allclose(
            np.einsum('pi,si,ij,qj,rj->pqrs', v4, v4c, -oww4, v4, v4c),
            test_op4)


        test_op1 *= (1 / 16)
        test_op2 *= (1 / 16)
        test_op3 *= (1 / 16)
        test_op4 *= (1 / 16)
        assert of.is_hermitian(1j * test_op1)
        assert of.is_hermitian(1j * test_op2)
        assert of.is_hermitian(1j * test_op3)
        assert of.is_hermitian(1j * test_op4)
        test_generator += test_op1 + test_op2 + test_op3 + test_op4

    assert np.allclose(test_generator, generator)
    print("PASSED")


def test_generalized_doubles2():
    generator = generate_antisymm_generator(6)
    nso = generator.shape[0]
    for p, q, r, s in product(range(nso), repeat=4):
        if p < q and s < r:
            assert np.isclose(generator[p, q, r, s], -generator[q, p, r, s])

    ul, vl, one_body_residual, ul_ops, vl_ops, one_body_op = \
        doubles_factorization(generator)

    generator_mat = np.reshape(np.transpose(generator, [0, 3, 1, 2]),
                               (nso ** 2, nso ** 2)).astype(np.float)
    one_body_residual_test = -np.einsum('pqrq->pr',
                                   generator)
    assert np.allclose(generator_mat, generator_mat.T)
    assert np.allclose(one_body_residual, one_body_residual_test)

    tgenerator_mat = np.zeros_like(generator_mat)
    for row_gem, col_gem in product(range(nso ** 2), repeat=2):
        p, s = row_gem // nso, row_gem % nso
        q, r = col_gem // nso, col_gem % nso
        tgenerator_mat[row_gem, col_gem] = generator[p, q, r, s]

    assert np.allclose(tgenerator_mat, generator_mat)

    u, sigma, vh = np.linalg.svd(generator_mat)

    fop = copy.deepcopy(one_body_op)
    fop2 = copy.deepcopy(one_body_op)
    fop3 = copy.deepcopy(one_body_op)
    fop4 = copy.deepcopy(one_body_op)
    fop5 = copy.deepcopy(one_body_op)
    for ll in range(len(sigma)):
        ul.append(np.sqrt(sigma[ll]) * u[:, ll].reshape((nso, nso)))
        ul_ops.append(
            get_fermion_op(np.sqrt(sigma[ll]) * u[:, ll].reshape((nso, nso))))
        vl.append(np.sqrt(sigma[ll]) * vh[ll, :].reshape((nso, nso)))
        vl_ops.append(
            get_fermion_op(np.sqrt(sigma[ll]) * vh[ll, :].reshape((nso, nso))))
        Smat = ul[ll] + vl[ll]
        Dmat = ul[ll] - vl[ll]

        S = ul_ops[ll] + vl_ops[ll]
        Sd = of.hermitian_conjugated(S)
        D = ul_ops[ll] - vl_ops[ll]
        Dd = of.hermitian_conjugated(D)
        op1 = S + 1j * Sd
        op2 = S - 1j * Sd
        op3 = D + 1j * Dd
        op4 = D - 1j * Dd
        assert np.isclose(of.normal_ordered(
            of.commutator(op1, of.hermitian_conjugated(op1))).induced_norm(), 0)
        assert np.isclose(of.normal_ordered(
            of.commutator(op2, of.hermitian_conjugated(op2))).induced_norm(), 0)
        assert np.isclose(of.normal_ordered(
            of.commutator(op3, of.hermitian_conjugated(op3))).induced_norm(), 0)
        assert np.isclose(of.normal_ordered(
            of.commutator(op4, of.hermitian_conjugated(op4))).induced_norm(), 0)

        fop3 += (1/8) * ((S**2 - Sd**2) - (D**2 - Dd**2))
        # fop3 += (1/8) * (S**2 - Sd**2 + D**2 - Dd**2)
        # fop4 += (1/16) * (op1**2 + op2**2 + op3**2 + op4**2)
        # fop4 += (1/16) * ((op2**2 - op1**2) + (op4**2 - op3**2))


        op1mat = Smat + 1j * Smat.T
        op2mat = Smat - 1j * Smat.T
        op3mat = Dmat + 1j * Dmat.T
        op4mat = Dmat - 1j * Dmat.T

        assert np.allclose(of.commutator(op1mat, op1mat.conj().T), 0)
        assert np.allclose(of.commutator(op2mat, op2mat.conj().T), 0)
        assert np.allclose(of.commutator(op3mat, op3mat.conj().T), 0)
        assert np.allclose(of.commutator(op4mat, op4mat.conj().T), 0)

        # check that we have normal operators and that the outer product
        # of their eigenvalues is imaginary. Also check vv is unitary
        if not np.isclose(sigma[ll], 0):
            assert np.isclose(of.normal_ordered(get_fermion_op(op1mat) - op1).induced_norm(), 0)
            assert np.isclose(of.normal_ordered(get_fermion_op(op2mat) - op2).induced_norm(), 0)
            assert np.isclose(of.normal_ordered(get_fermion_op(op3mat) - op3).induced_norm(), 0)
            assert np.isclose(of.normal_ordered(get_fermion_op(op4mat) - op4).induced_norm(), 0)

            ww, vv = np.linalg.eig(op1mat)
            eye = np.eye(nso)
            assert np.allclose(np.outer(ww, ww).real, 0)
            assert np.allclose(vv.conj().T @ vv, eye)
            ww, vv = np.linalg.eig(op2mat)
            assert np.allclose(np.outer(ww, ww).real, 0)
            assert np.allclose(vv.conj().T @ vv, eye)
            ww, vv = np.linalg.eig(op3mat)
            assert np.allclose(np.outer(ww, ww).real, 0)
            assert np.allclose(vv.conj().T @ vv, eye)
            ww, vv = np.linalg.eig(op4mat)
            assert np.allclose(np.outer(ww, ww).real, 0)
            assert np.allclose(vv.conj().T @ vv, eye)

        # fop2 += 0.25 * ul_ops[ll] * vl_ops[ll]
        # fop2 += 0.25 * vl_ops[ll] * ul_ops[ll]
        # fop2 += -0.25 * of.hermitian_conjugated(
        #     vl_ops[ll]) * of.hermitian_conjugated(ul_ops[ll])
        # fop2 += -0.25 * of.hermitian_conjugated(
        #     ul_ops[ll]) * of.hermitian_conjugated(vl_ops[ll])
        fop2 += 0.25 * ul_ops[ll] * vl_ops[ll]
        fop2 += 0.25 * vl_ops[ll] * ul_ops[ll]
        fop2 += -0.25 * of.hermitian_conjugated(
            vl_ops[ll]) * of.hermitian_conjugated(ul_ops[ll])
        fop2 += -0.25 * of.hermitian_conjugated(
            ul_ops[ll]) * of.hermitian_conjugated(vl_ops[ll])

        assert np.allclose(of.normal_ordered(ul_ops[ll] * vl_ops[ll] + vl_ops[ll] * ul_ops[ll] - 0.5 * (S**2 - D**2)).induced_norm(), 0)
        # fop5 += (1/8) * (S**2 - D**2)
        # fop5 += (-1/8) * (Sd**2 - Dd**2)
        # fop5 += (1/8) * (S**2 - Sd**2)
        # fop5 += (1/8) * (Dd**2 - D**2)
        fop5 += (1/8) * (S**2 - Sd**2 + Dd**2 - D**2)
        fop4 += (1/16) * ((op1**2 + op2**2) - (op3**2 + op4**2))

        fop += vl_ops[ll] * ul_ops[ll]

    true_fop = get_fermion_op(generator)
    assert np.isclose(of.normal_ordered(fop - true_fop).induced_norm(), 0)
    assert np.isclose(of.normal_ordered(fop2 - true_fop).induced_norm(), 0)
    assert np.isclose(of.normal_ordered(fop3 - true_fop).induced_norm(), 0)
    assert np.isclose(of.normal_ordered(fop4 - true_fop).induced_norm(), 0)
    assert np.isclose(of.normal_ordered(fop5 - true_fop).induced_norm(), 0)


def test_normal_op_tensor_reconstruction2():
    sdim = 4
    generator = generate_antisymm_generator(2 * sdim)
    nso = generator.shape[0]
    for p, q, r, s in product(range(nso), repeat=4):
        if p < q and s < r:
            assert np.isclose(generator[p, q, r, s], -generator[q, p, r, s])

    generator_mat = np.reshape(np.transpose(generator, [0, 3, 1, 2]),
                               (nso ** 2, nso ** 2)).astype(np.float)
    u, sigma, vh = np.linalg.svd(generator_mat)

    ul, vl, one_body_residual, ul_ops, vl_ops, one_body_op = \
        doubles_factorization(generator)

    sigma_idx = np.where(sigma > 1.0E-13)[0]
    test_generator_mat = np.zeros_like(generator_mat)
    for p, q, r, s in product(range(nso), repeat=4):
        for ll in sigma_idx:
            Smat = ul[ll] + vl[ll]
            Dmat = ul[ll] - vl[ll]

            op1mat = Smat + 1j * Smat.T
            op2mat = Smat - 1j * Smat.T
            op3mat = Dmat + 1j * Dmat.T
            op4mat = Dmat - 1j * Dmat.T
            test_generator_mat[p * nso + s, q * nso + r] += (1 / 16) * \
                                                            (op1mat[p, s] *
                                                             op1mat[q, r] +
                                                             op2mat[p, s] *
                                                             op2mat[q, r] -
                                                             op3mat[p, s] *
                                                             op3mat[q, r] -
                                                             op4mat[p, s] *
                                                             op4mat[q, r])

    assert np.allclose(test_generator_mat, generator_mat)

    test_generator = np.zeros_like(generator).astype(np.complex128)
    for ll in sigma_idx:
        Smat = ul[ll] + vl[ll]
        Dmat = ul[ll] - vl[ll]

        op1mat = Smat + 1j * Smat.T
        op2mat = Smat - 1j * Smat.T
        op3mat = Dmat + 1j * Dmat.T
        op4mat = Dmat - 1j * Dmat.T

        test_op1 = np.zeros((nso, nso, nso, nso), dtype=np.complex128)
        test_op2 = np.zeros((nso, nso, nso, nso), dtype=np.complex128)
        test_op3 = np.zeros((nso, nso, nso, nso), dtype=np.complex128)
        test_op4 = np.zeros((nso, nso, nso, nso), dtype=np.complex128)
        for p, q, r, s in product(range(nso), repeat=4):
            test_op1[p, q, r, s] += op1mat[p, s] * op1mat[q, r]
            test_op2[p, q, r, s] += op2mat[p, s] * op2mat[q, r]
            test_op3[p, q, r, s] -= op3mat[p, s] * op3mat[q, r]
            test_op4[p, q, r, s] -= op4mat[p, s] * op4mat[q, r]
        test_op1 *= (1 / 16)
        test_op2 *= (1 / 16)
        test_op3 *= (1 / 16)
        test_op4 *= (1 / 16)
        assert of.is_hermitian(1j * test_op1)
        assert of.is_hermitian(1j * test_op2)
        assert of.is_hermitian(1j * test_op3)
        assert of.is_hermitian(1j * test_op4)
        test_generator += test_op1 + test_op2 + test_op3 + test_op4

    assert np.allclose(test_generator, generator)

    test_generator = np.zeros_like(generator).astype(np.complex128)
    for ll in sigma_idx:
        Smat = ul[ll] + vl[ll]
        Dmat = ul[ll] - vl[ll]

        op1mat = Smat + 1j * Smat.T
        op2mat = Smat - 1j * Smat.T
        op3mat = Dmat + 1j * Dmat.T
        op4mat = Dmat - 1j * Dmat.T

        w1, v1 = np.linalg.eig(op1mat)
        assert np.allclose(v1 @ np.diag(w1) @ v1.conj().T, op1mat)
        v1c = v1.conj()
        for m, n in product(range(op1mat.shape[0]), repeat=2):
            assert np.isclose(op1mat[m, n],
                              v1[m, :].dot(np.diag(w1)).dot(v1c[n, :]))
        w2, v2 = np.linalg.eig(op2mat)
        assert np.allclose(v2 @ np.diag(w2) @ v2.conj().T, op2mat)
        v2c = v2.conj()
        for m, n in product(range(op2mat.shape[0]), repeat=2):
            assert np.isclose(op2mat[m, n],
                              np.einsum('j,j,j', v2[m, :], w2, v2c[n, :]))

        w3, v3 = np.linalg.eig(op3mat)
        v3c = v3.conj()
        assert np.allclose(v3 @ np.diag(w3) @ v3.conj().T, op3mat)
        w4, v4 = np.linalg.eig(op4mat)
        v4c = v4.conj()
        assert np.allclose(v4 @ np.diag(w4) @ v4.conj().T, op4mat)

        test_op1 = np.zeros((nso, nso, nso, nso), dtype=np.complex128)
        test_op2 = np.zeros((nso, nso, nso, nso), dtype=np.complex128)
        test_op3 = np.zeros((nso, nso, nso, nso), dtype=np.complex128)
        test_op4 = np.zeros((nso, nso, nso, nso), dtype=np.complex128)
        oww1 = np.outer(w1, w1)
        oww2 = np.outer(w2, w2)
        oww3 = np.outer(w3, w3)
        oww4 = np.outer(w4, w4)
        assert np.allclose(v1, v2.conj())
        assert np.allclose(v1, v3.conj())
        assert np.allclose(v1, v4)
        assert np.allclose(oww1, -oww2)
        assert np.allclose(oww3, -oww4)

        for p, q, r, s in product(range(nso), repeat=4):
            test_op1[p, q, r, s] += op1mat[p, s] * op1mat[q, r]
            assert np.isclose(op1mat[p, s] * op1mat[q, r],
                              np.einsum('i,i,ij,j,j', v1[p, :], v1c[s, :], oww1,
                                        v1[q, :], v1c[r, :]))
            test_op2[p, q, r, s] += op2mat[p, s] * op2mat[q, r]
            test_op3[p, q, r, s] -= op3mat[p, s] * op3mat[q, r]
            test_op4[p, q, r, s] -= op4mat[p, s] * op4mat[q, r]

        assert np.allclose(np.einsum('pi,si,ij,qj,rj->pqrs', v1, v1c, oww1, v1, v1c),
                           test_op1)
        assert np.allclose(
            np.einsum('pi,si,ij,qj,rj->pqrs', v2, v2c, oww2, v2, v2c),
            test_op2)
        assert np.allclose(
            np.einsum('pi,si,ij,qj,rj->pqrs', v3, v3c, -oww3, v3, v3c),
            test_op3)
        assert np.allclose(
            np.einsum('pi,si,ij,qj,rj->pqrs', v4, v4c, -oww4, v4, v4c),
            test_op4)


        test_op1 *= (1 / 16)
        test_op2 *= (1 / 16)
        test_op3 *= (1 / 16)
        test_op4 *= (1 / 16)
        assert of.is_hermitian(1j * test_op1)
        assert of.is_hermitian(1j * test_op2)
        assert of.is_hermitian(1j * test_op3)
        assert of.is_hermitian(1j * test_op4)
        test_generator += test_op1 + test_op2 + test_op3 + test_op4

    assert np.allclose(test_generator, generator)
    print("PASSED")


def get_lih_molecule(bd):
    from openfermionpyscf import run_pyscf
    import os
    geometry = [['Li', [0, 0, 0]],
                ['H', [0, 0, bd]]]
    molecule = of.MolecularData(geometry=geometry, charge=0,
                                multiplicity=1,
                                basis='sto-3g')
    molecule.filename = os.path.join(os.getcwd(), molecule.name)
    molecule = run_pyscf(molecule, run_scf=True, run_fci=True)
    return molecule

def get_h4_molecule(bd):
    from openfermionpyscf import run_pyscf
    import os
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

def test_generalized_doubles_takagi():
    molecule = get_lih_molecule(1.7)
    molecule = build_lih_moleculardata()
    oei, tei = molecule.get_integrals()
    nele = 4
    nalpha = 2
    nbeta = 2
    sz = 0
    norbs = oei.shape[0]
    nso = 2 * norbs
    fqe_wf = fqe.Wavefunction(
        [[nele, sz, norbs]])
    fqe_wf.set_wfn(strategy='random')
    fqe_wf.normalize()
    opdm, tpdm = fqe_wf.sector((nele, sz)).get_openfermion_rdms()
    d3 = fqe_wf.sector((nele, sz)).get_three_pdm()

    adapt = ADAPT(oei, tei, None, nalpha, nbeta, iter_max=50)
    acse_residual = two_rdo_commutator_symm(
                adapt.reduced_ham.two_body_tensor, tpdm,
                d3)
    for p, q, r, s in product(range(nso), repeat=4):
        if p == q or r == s:
            continue
        assert np.isclose(acse_residual[p, q, r, s],
                          -acse_residual[s, r, q, p].conj())

    Zlp, Zlm, Zl, one_body_residual = doubles_factorization2(acse_residual)
    test_fop = get_fermion_op(one_body_residual)
    for ll in range(len(Zlp)):
        test_fop += 0.25 * get_fermion_op(Zlp[ll]) ** 2
        test_fop += 0.25 * get_fermion_op(Zlm[ll]) ** 2
    assert np.isclose(of.normal_ordered(
        test_fop - get_fermion_op(acse_residual)).induced_norm(), 0,
                      atol=1.0E-6)

def test_takagi():
    A = np.random.randn(36).reshape((6, 6))
    B = np.random.randn(36).reshape((6, 6))
    A = A + A.T
    B = B + B.T
    C = A + 1j * B
    assert np.allclose(C, C.T)
    T, Z = takagi(C)
    assert np.allclose(Z @ np.diag(T) @ Z.T, C)


if __name__ == "__main__":
    np.random.seed(10)
    np.set_printoptions(linewidth=500)
    # np.random.seed(10)
    # test_generalized_doubles2()
    # test_random_evolution()
    # test_normal_op_tensor_reconstruction2()
    # test_trotter_error()
    # test_reconstruction_error()
    # test_generalized_doubles_takagi()
    test_takagi()
