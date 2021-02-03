#   Copyright 2020 Google LLC

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
from itertools import product
import copy

import numpy as np
import scipy as sp

import openfermion as of
from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermion.chem import make_reduced_hamiltonian

import fqe
from fqe.algorithm.generalized_doubles_factorization import (
    doubles_factorization_svd, doubles_factorization_takagi, takagi)
from fqe.algorithm.brillouin_calculator import two_rdo_commutator_symm
from fqe.algorithm.brillouin_calculator import get_fermion_op
from fqe.algorithm.low_rank import (
    evolve_fqe_givens_unrestricted,)
from fqe.fqe_decorators import build_hamiltonian

from fqe.unittest_data.generate_openfermion_molecule import \
    build_lih_moleculardata


def generate_antisymm_generator(nso):
    A = np.zeros(tuple([nso] * 4))
    for p, q, r, s in product(range(nso), repeat=4):
        if p < q and s < r and p * nso + q < s * nso + r:
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
    generator = generate_antisymm_generator(2)
    nso = generator.shape[0]
    for p, q, r, s in product(range(nso), repeat=4):
        if p < q and s < r:
            assert np.isclose(generator[p, q, r, s], -generator[q, p, r, s])

    ul, vl, one_body_residual, ul_ops, vl_ops, one_body_op = \
        doubles_factorization_svd(generator)

    generator_mat = np.reshape(np.transpose(generator, [0, 3, 1, 2]),
                               (nso**2, nso**2)).astype(np.float)
    one_body_residual_test = -np.einsum('pqrq->pr', generator)
    assert np.allclose(generator_mat, generator_mat.T)
    assert np.allclose(one_body_residual, one_body_residual_test)

    tgenerator_mat = np.zeros_like(generator_mat)
    for row_gem, col_gem in product(range(nso**2), repeat=2):
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
        assert np.isclose(
            of.normal_ordered(of.commutator(
                op1, of.hermitian_conjugated(op1))).induced_norm(), 0)
        assert np.isclose(
            of.normal_ordered(of.commutator(
                op2, of.hermitian_conjugated(op2))).induced_norm(), 0)
        assert np.isclose(
            of.normal_ordered(of.commutator(
                op3, of.hermitian_conjugated(op3))).induced_norm(), 0)
        assert np.isclose(
            of.normal_ordered(of.commutator(
                op4, of.hermitian_conjugated(op4))).induced_norm(), 0)

        fop3 += (1 / 8) * ((S**2 - Sd**2) - (D**2 - Dd**2))
        fop4 += (1 / 16) * ((op1**2 + op2**2) - (op3**2 + op4**2))

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
            assert np.isclose(
                of.normal_ordered(get_fermion_op(op1mat) - op1).induced_norm(),
                0)
            assert np.isclose(
                of.normal_ordered(get_fermion_op(op2mat) - op2).induced_norm(),
                0)
            assert np.isclose(
                of.normal_ordered(get_fermion_op(op3mat) - op3).induced_norm(),
                0)
            assert np.isclose(
                of.normal_ordered(get_fermion_op(op4mat) - op4).induced_norm(),
                0)

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
    sdim = 2
    nele = 2
    generator = generate_antisymm_generator(2 * sdim)
    nso = generator.shape[0]
    for p, q, r, s in product(range(nso), repeat=4):
        if p < q and s < r:
            assert np.isclose(generator[p, q, r, s], -generator[q, p, r, s])

    generator_mat = np.reshape(np.transpose(generator, [0, 3, 1, 2]),
                               (nso**2, nso**2)).astype(np.float)
    _, sigma, _ = np.linalg.svd(generator_mat)

    ul, vl, _, ul_ops, vl_ops, _ = \
        doubles_factorization_svd(generator)

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

        op1mat = Smat + 1j * Smat.T
        op2mat = Smat - 1j * Smat.T
        op3mat = Dmat + 1j * Dmat.T
        op4mat = Dmat - 1j * Dmat.T

        o1_rwf = rwf.time_evolve(1 / 16, 1j * op1**2)
        ww, vv = np.linalg.eig(op1mat)
        assert np.allclose(vv @ np.diag(ww) @ vv.conj().T, op1mat)
        trwf = evolve_fqe_givens_unrestricted(rwf, vv.conj().T)
        v_pq = np.outer(ww, ww)
        for p, q in product(range(nso), repeat=2):
            fop = of.FermionOperator(((p, 1), (p, 0), (q, 1), (q, 0)),
                                     coefficient=-v_pq[p, q].imag)
            trwf = trwf.time_evolve(1 / 16, fop)
        trwf = evolve_fqe_givens_unrestricted(trwf, vv)
        assert np.isclose(fqe.vdot(o1_rwf, trwf), 1)

        o_rwf = rwf.time_evolve(1 / 16, 1j * op2**2)
        ww, vv = np.linalg.eig(op2mat)
        assert np.allclose(vv @ np.diag(ww) @ vv.conj().T, op2mat)
        trwf = evolve_fqe_givens_unrestricted(rwf, vv.conj().T)
        v_pq = np.outer(ww, ww)
        for p, q in product(range(nso), repeat=2):
            fop = of.FermionOperator(((p, 1), (p, 0), (q, 1), (q, 0)),
                                     coefficient=-v_pq[p, q].imag)
            trwf = trwf.time_evolve(1 / 16, fop)
        trwf = evolve_fqe_givens_unrestricted(trwf, vv)
        assert np.isclose(fqe.vdot(o_rwf, trwf), 1)

        o_rwf = rwf.time_evolve(-1 / 16, 1j * op3**2)
        ww, vv = np.linalg.eig(op3mat)
        assert np.allclose(vv @ np.diag(ww) @ vv.conj().T, op3mat)
        trwf = evolve_fqe_givens_unrestricted(rwf, vv.conj().T)
        v_pq = np.outer(ww, ww)
        for p, q in product(range(nso), repeat=2):
            fop = of.FermionOperator(((p, 1), (p, 0), (q, 1), (q, 0)),
                                     coefficient=-v_pq[p, q].imag)
            trwf = trwf.time_evolve(-1 / 16, fop)
        trwf = evolve_fqe_givens_unrestricted(trwf, vv)
        assert np.isclose(fqe.vdot(o_rwf, trwf), 1)

        o_rwf = rwf.time_evolve(-1 / 16, 1j * op4**2)
        ww, vv = np.linalg.eig(op4mat)
        assert np.allclose(vv @ np.diag(ww) @ vv.conj().T, op4mat)
        trwf = evolve_fqe_givens_unrestricted(rwf, vv.conj().T)
        v_pq = np.outer(ww, ww)
        for p, q in product(range(nso), repeat=2):
            fop = of.FermionOperator(((p, 1), (p, 0), (q, 1), (q, 0)),
                                     coefficient=-v_pq[p, q].imag)
            trwf = trwf.time_evolve(-1 / 16, fop)
        trwf = evolve_fqe_givens_unrestricted(trwf, vv)
        assert np.isclose(fqe.vdot(o_rwf, trwf), 1)


def test_normal_op_tensor_reconstruction():
    sdim = 2
    generator = generate_antisymm_generator(2 * sdim)
    nso = generator.shape[0]
    for p, q, r, s in product(range(nso), repeat=4):
        if p < q and s < r:
            assert np.isclose(generator[p, q, r, s], -generator[q, p, r, s])

    generator_mat = np.reshape(np.transpose(generator, [0, 3, 1, 2]),
                               (nso**2, nso**2)).astype(np.float)
    _, sigma, _ = np.linalg.svd(generator_mat)

    ul, vl, _, _, _, _ = \
        doubles_factorization_svd(generator)

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
                                                             op4mat[q, r]).real

    assert np.allclose(test_generator_mat, generator_mat)

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
            assert np.isclose(
                op1mat[p, s] * op1mat[q, r],
                np.einsum('i,i,ij,j,j', v1[p, :], v1c[s, :], oww1, v1[q, :],
                          v1c[r, :]))
            test_op2[p, q, r, s] += op2mat[p, s] * op2mat[q, r]
            test_op3[p, q, r, s] -= op3mat[p, s] * op3mat[q, r]
            test_op4[p, q, r, s] -= op4mat[p, s] * op4mat[q, r]

        assert np.allclose(
            np.einsum('pi,si,ij,qj,rj->pqrs', v1, v1c, oww1, v1, v1c), test_op1)
        assert np.allclose(
            np.einsum('pi,si,ij,qj,rj->pqrs', v2, v2c, oww2, v2, v2c), test_op2)
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


def test_generalized_doubles_takagi():
    molecule = build_lih_moleculardata()
    oei, tei = molecule.get_integrals()
    nele = 4
    nalpha = 2
    nbeta = 2
    sz = 0
    norbs = oei.shape[0]
    nso = 2 * norbs
    fqe_wf = fqe.Wavefunction([[nele, sz, norbs]])
    fqe_wf.set_wfn(strategy='hartree-fock')
    fqe_wf.normalize()
    _, tpdm = fqe_wf.sector((nele, sz)).get_openfermion_rdms()
    d3 = fqe_wf.sector((nele, sz)).get_three_pdm()

    soei, stei = spinorb_from_spatial(oei, tei)
    astei = np.einsum('ijkl', stei) - np.einsum('ijlk', stei)
    molecular_hamiltonian = of.InteractionOperator(0, soei, 0.25 * astei)
    reduced_ham = make_reduced_hamiltonian(molecular_hamiltonian,
                                           nalpha + nbeta)
    acse_residual = two_rdo_commutator_symm(reduced_ham.two_body_tensor, tpdm,
                                            d3)
    for p, q, r, s in product(range(nso), repeat=4):
        if p == q or r == s:
            continue
        assert np.isclose(acse_residual[p, q, r, s],
                          -acse_residual[s, r, q, p].conj())

    Zlp, Zlm, _, one_body_residual = doubles_factorization_takagi(acse_residual)
    test_fop = get_fermion_op(one_body_residual)
    # test the first four factors
    for ll in range(4):
        test_fop += 0.25 * get_fermion_op(Zlp[ll])**2
        test_fop += 0.25 * get_fermion_op(Zlm[ll])**2

        op1mat = Zlp[ll]
        op2mat = Zlm[ll]
        w1, v1 = sp.linalg.schur(op1mat)
        w1 = np.diagonal(w1)
        assert np.allclose(v1 @ np.diag(w1) @ v1.conj().T, op1mat)

        v1c = v1.conj()
        w2, v2 = sp.linalg.schur(op2mat)
        w2 = np.diagonal(w2)
        assert np.allclose(v2 @ np.diag(w2) @ v2.conj().T, op2mat)
        oww1 = np.outer(w1, w1)

        fqe_wf = fqe.Wavefunction([[nele, sz, norbs]])
        fqe_wf.set_wfn(strategy='hartree-fock')
        fqe_wf.normalize()
        nfqe_wf = fqe.get_number_conserving_wavefunction(nele, norbs)
        nfqe_wf.sector((nele, sz)).coeff = fqe_wf.sector((nele, sz)).coeff

        this_generatory = np.einsum('pi,si,ij,qj,rj->pqrs', v1, v1c, oww1, v1,
                                    v1c)
        fop = of.FermionOperator()
        for p, q, r, s in product(range(nso), repeat=4):
            op = ((p, 1), (s, 0), (q, 1), (r, 0))
            fop += of.FermionOperator(op,
                                      coefficient=this_generatory[p, q, r, s])

        fqe_fop = build_hamiltonian(1j * fop, norb=norbs, conserve_number=True)
        exact_wf = fqe.apply_generated_unitary(nfqe_wf, 1, 'taylor', fqe_fop)

        test_wf = fqe.algorithm.low_rank.evolve_fqe_givens_unrestricted(
            nfqe_wf,
            v1.conj().T)
        test_wf = fqe.algorithm.low_rank.evolve_fqe_charge_charge_unrestricted(
            test_wf, -oww1.imag)
        test_wf = fqe.algorithm.low_rank.evolve_fqe_givens_unrestricted(
            test_wf, v1)

        assert np.isclose(abs(fqe.vdot(test_wf, exact_wf))**2, 1)


def test_takagi():
    A = np.random.randn(36).reshape((6, 6))
    B = np.random.randn(36).reshape((6, 6))
    A = A + A.T
    B = B + B.T
    C = A + 1j * B
    assert np.allclose(C, C.T)
    T, Z = takagi(C)
    assert np.allclose(Z @ np.diag(T) @ Z.T, C)
