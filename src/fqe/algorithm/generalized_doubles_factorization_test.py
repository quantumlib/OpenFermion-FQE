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

import copy
from itertools import product

import numpy as np
import openfermion as of

from fqe.algorithm.generalized_doubles_factorization import doubles_factorization
from fqe.algorithm.brillouin_calculator import get_fermion_op


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
