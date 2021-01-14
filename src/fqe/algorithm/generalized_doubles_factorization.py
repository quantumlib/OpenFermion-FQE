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
"""
Generalized doubles factorization

This will go into OpenFermion.  Putting here until I write something up
or decide to just publish the code.
"""
from typing import List, Tuple
from itertools import product
import numpy as np
import openfermion as of
from fqe.algorithm.brillouin_calculator import get_fermion_op


def doubles_factorization(generator_tensor: np.ndarray, eig_cutoff=None):
    """
    Given an antisymmetric antihermitian tensor perform a double factorized
    low-rank decomposition.

    Given:

    A = sum_{pqrs}A^{pq}_{sr}p^ q^ r s

    with A^{pq}_{sr} = -A^{qp}_{sr} = -A^{pq}_{rs} = -A^{sr}_{pq}

    Rewrite A as a sum-of squares s.t

    A = sum_{l}Y_{l}^2

    where Y_{l} are normal operator one-body operators such that the spectral
    theorem holds and we can use the double factorization to implement an
    approximate evolution.
    """
    if not np.allclose(generator_tensor.imag, 0):
        raise TypeError("generator_tensor must be a real matrix")

    if eig_cutoff is not None:
        if eig_cutoff % 2 != 0:
            raise ValueError("eig_cutoff must be an even number")

    nso = generator_tensor.shape[0]
    generator_tensor = generator_tensor.astype(np.float)
    generator_mat = np.zeros((nso**2, nso**2))
    for row_gem, col_gem in product(range(nso**2), repeat=2):
        p, s = row_gem // nso, row_gem % nso
        q, r = col_gem // nso, col_gem % nso
        generator_mat[row_gem, col_gem] = generator_tensor[p, q, r, s]
    test_generator_mat = np.reshape(np.transpose(generator_tensor, [0, 3, 1, 2]),
                               (nso ** 2, nso ** 2)).astype(np.float)
    assert np.allclose(test_generator_mat, generator_mat)

    if not np.allclose(generator_mat, generator_mat.T):
        raise ValueError("generator tensor does not correspond to four-fold"
                         " antisymmetry")

    one_body_residual = -np.einsum('pqrq->pr',
                                   generator_tensor)
    u, sigma, vh = np.linalg.svd(generator_mat)

    ul = []
    ul_ops = []
    vl = []
    vl_ops = []
    if eig_cutoff is None:
        max_sigma = len(sigma)
    else:
        max_sigma = eig_cutoff

    for ll in range(max_sigma):
        ul.append(np.sqrt(sigma[ll]) * u[:, ll].reshape((nso, nso)))
        ul_ops.append(
            get_fermion_op(np.sqrt(sigma[ll]) * u[:, ll].reshape((nso, nso))))
        vl.append(np.sqrt(sigma[ll]) * vh[ll, :].reshape((nso, nso)))
        vl_ops.append(
            get_fermion_op(np.sqrt(sigma[ll]) * vh[ll, :].reshape((nso, nso))))
        S = ul_ops[ll] + vl_ops[ll]
        D = ul_ops[ll] - vl_ops[ll]
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

    one_body_op = of.FermionOperator()
    for p, q in product(range(nso), repeat=2):
        tfop = ((p, 1), (q, 0))
        one_body_op += of.FermionOperator(tfop,
                                          coefficient=one_body_residual[p, q])

    return ul, vl, one_body_residual, ul_ops, vl_ops, one_body_op

