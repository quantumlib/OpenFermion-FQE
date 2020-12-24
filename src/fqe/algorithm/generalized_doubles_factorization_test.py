from itertools import product
import numpy as np
import scipy as sp
import copy

import fqe
from fqe.algorithm.generalized_doubles_factorization import doubles_factorization
from fqe.algorithm.brillouin_calculator import get_fermion_op
from fqe.algorithm.low_rank import (evolve_fqe_givens_unrestricted,
                                    evolve_fqe_charge_charge_unrestricted, )
from fqe.fqe_decorators import build_hamiltonian
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


def test_normal_op_tensor_reconstruction():
    sdim = 5
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
            test_generator_mat[p * nso + s, q * nso + r] += (1/16) * \
                (op1mat[p, s] * op1mat[q, r] + op2mat[p, s] * op2mat[q, r] -
                 op3mat[p, s] * op3mat[q, r] - op4mat[p, s] * op4mat[q, r])

    assert np.allclose(test_generator_mat,generator_mat)

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
        test_op1 *= (1/16)
        test_op2 *= (1/16)
        test_op3 *= (1/16)
        test_op4 *= (1/16)
        assert of.is_hermitian(1j * test_op1)
        assert of.is_hermitian(1j * test_op2)
        assert of.is_hermitian(1j * test_op3)
        assert of.is_hermitian(1j * test_op4)


def test_trotter_error():
    sdim = 8
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


if __name__ == "__main__":
    # np.random.seed(10)
    # test_generalized_doubles()
    # test_random_evolution()
    # test_normal_op_tensor_reconstruction()
    test_trotter_error()
