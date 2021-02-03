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
from itertools import product, groupby
import numpy as np
import scipy as sp
from scipy.linalg import block_diag, sqrtm, polar, schur
import openfermion as of
from fqe.algorithm.brillouin_calculator import get_fermion_op


def doubles_factorization_svd(generator_tensor: np.ndarray, eig_cutoff=None):
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
    generator_tensor = generator_tensor.real
    generator_mat = np.zeros((nso**2, nso**2))
    for row_gem, col_gem in product(range(nso**2), repeat=2):
        p, s = row_gem // nso, row_gem % nso
        q, r = col_gem // nso, col_gem % nso
        generator_mat[row_gem, col_gem] = generator_tensor[p, q, r, s]
    test_generator_mat = np.reshape(
        np.transpose(generator_tensor, [0, 3, 1, 2]),
        (nso**2, nso**2)).astype(np.float)

    assert np.allclose(test_generator_mat, generator_mat)

    if not np.allclose(generator_mat, generator_mat.T):
        raise ValueError("generator tensor does not correspond to four-fold"
                         " antisymmetry")

    one_body_residual = -np.einsum('pqrq->pr', generator_tensor)
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

    one_body_op = of.FermionOperator()
    for p, q in product(range(nso), repeat=2):
        tfop = ((p, 1), (q, 0))
        one_body_op += of.FermionOperator(tfop,
                                          coefficient=one_body_residual[p, q])

    return ul, vl, one_body_residual, ul_ops, vl_ops, one_body_op


def takagi(N, tol=1e-13, rounding=13):
    r"""Autonne-Takagi decomposition of a complex symmetric (not Hermitian!) matrix.

    Note that singular values of N are considered equal if they are equal after np.round(values, tol).

    Taken from Strawberry Fields
    [https://github.com/XanaduAI/strawberryfields/blob/master/strawberryfields/decompositions.py#L28]

    Args:
        N (array[complex]): square, symmetric matrix N
        rounding (int): the number of decimal places to use when rounding the singular values of N
        tol (float): the tolerance used when checking if the input matrix is symmetric: :math:`|N-N^T| <` tol

    Returns:
        tuple[array, array]: (rl, U), where rl are the (rounded) singular values,
            and U is the Takagi unitary, such that :math:`N = U \diag(rl) U^T`.
    """
    (n, m) = N.shape
    if n != m:
        raise ValueError("The input matrix must be square")
    if np.linalg.norm(N - np.transpose(N)) >= tol:
        raise ValueError("The input matrix is not symmetric")

    N = np.real_if_close(N)

    if np.allclose(N, 0):
        return np.zeros(n), np.eye(n)

    if np.isrealobj(N):
        # If the matrix N is real one can be more clever and use its eigendecomposition
        l, U = np.linalg.eigh(N)
        vals = np.abs(l)  # These are the Takagi eigenvalues
        phases = np.sqrt(np.complex128([1 if i > 0 else -1 for i in l]))
        Uc = U @ np.diag(phases)  # One needs to readjust the phases
        list_vals = [(vals[i], i) for i in range(len(vals))]
        list_vals.sort(reverse=True)
        sorted_l, permutation = zip(*list_vals)
        permutation = np.array(permutation)
        Uc = Uc[:, permutation]
        # And also rearrange the unitary and values so that they are decreasingly ordered
        return np.array(sorted_l), Uc

    v, l, ws = np.linalg.svd(N)
    w = np.transpose(np.conjugate(ws))
    rl = np.round(l, rounding)

    # Generate list with degenerancies
    result = []
    for k, g in groupby(rl):
        result.append(list(g))

    # Generate lists containing the columns that correspond to degenerancies
    kk = 0
    for k in result:
        for ind, j in enumerate(k):  # pylint: disable=unused-variable
            k[ind] = kk
            kk = kk + 1

    # Generate the lists with the degenerate column subspaces
    vas = []
    was = []
    for i in result:
        vas.append(v[:, i])
        was.append(w[:, i])

    # Generate the matrices qs of the degenerate subspaces
    qs = []
    for i in range(len(result)):
        qs.append(sqrtm(np.transpose(vas[i]) @ was[i]))

    # Construct the Takagi unitary
    qb = block_diag(*qs)

    U = v @ np.conj(qb)
    return rl, U


def doubles_factorization_takagi(generator_tensor: np.ndarray, eig_cutoff=None):
    """
    Given an antisymmetric antihermitian tensor perform a double factorized
    low-rank decomposition.  This uses the Takagi decomposition of a complex
    symmetric matrix.  This reduces the number of tensor from 4 to 2 when
    compared against the SVD appraoch.

    Given:

    A = sum_{pqrs}A^{pq}_{sr}p^ q^ r s

    with A^{pq}_{sr} = -A^{qp}_{sr} = -A^{pq}_{rs} = -A^{sr}_{pq}

    Rewrite A as a sum-of squares s.t

    A = sum_{l}Y_{l}^2

    where Y_{l} are normal operator one-body operators such that the spectral
    theorem holds and we can use the double factorization to implement an
    approximate evolution.
    """
    if eig_cutoff is not None:
        if eig_cutoff % 2 != 0:
            raise ValueError("eig_cutoff must be an even number")

    nso = generator_tensor.shape[0]
    generator_mat = np.reshape(np.transpose(generator_tensor, [0, 3, 1, 2]),
                               (nso**2, nso**2))
    assert np.allclose(generator_mat, generator_mat.T)

    one_body_residual = -np.einsum('pqrq->pr', generator_tensor)

    # complex symmetric matrices give Q S Q^T with S diagonal and real
    # and Q is unitary.
    T, Z = takagi(generator_mat)

    nonzero_idx = np.where(T > 1.0E-12)[0]
    if eig_cutoff is None:
        max_sigma = len(nonzero_idx)
    else:
        max_sigma = eig_cutoff

    Zl = []
    Zlp = []
    Zlm = []
    for idx in nonzero_idx[:max_sigma]:
        Zl.append(np.sqrt(T[idx]) * Z[:, idx].reshape((nso, nso)))
        Zlp.append(Zl[-1] + 1j * Zl[-1].conj().T)
        Zlm.append(Zl[-1] - 1j * Zl[-1].conj().T)

    return Zlp, Zlm, Zl, one_body_residual
