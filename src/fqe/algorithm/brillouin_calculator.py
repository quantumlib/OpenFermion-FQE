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
"""Infrastructure for compute [rdo, A] with FQE. RDO is a 2-body operator
and A is a 2-body operator."""

import copy
from itertools import product

import numpy as np

import openfermion as of
import fqe
from fqe.wavefunction import Wavefunction
from fqe.hamiltonians.restricted_hamiltonian import RestrictedHamiltonian

try:
    from joblib import Parallel, delayed

    PARALLELIZABLE = True
except ImportError:
    PARALLELIZABLE = False


def get_fermion_op(coeff_tensor) -> of.FermionOperator:
    r"""Returns an openfermion.FermionOperator from the given coeff_tensor.

    Given A[i, j, k, l] of A = \sum_{ijkl}A[i, j, k, l]i^ j^ k^ l
    return the FermionOperator A.

    Args:
        coeff_tensor: Coefficients for 4-mode operator

    Returns:
        A FermionOperator object
    """
    if len(coeff_tensor.shape) == 4:
        nso = coeff_tensor.shape[0]
        fermion_op = of.FermionOperator()
        for p, q, r, s in product(range(nso), repeat=4):
            if p == q or r == s:
                continue
            op = ((p, 1), (q, 1), (r, 0), (s, 0))
            fop = of.FermionOperator(op, coefficient=coeff_tensor[p, q, r, s])
            fermion_op += fop
        return fermion_op

    elif len(coeff_tensor.shape) == 2:
        nso = coeff_tensor.shape[0]
        fermion_op = of.FermionOperator()
        for p, q in product(range(nso), repeat=2):
            oper = ((p, 1), (q, 0))
            fop = of.FermionOperator(oper, coefficient=coeff_tensor[p, q])
            fermion_op += fop
        return fermion_op

    else:
        raise ValueError(
            "Arg `coeff_tensor` should have dimension 2 or 4 but has dimension"
            f" {len(coeff_tensor.shape)}.")


def get_acse_residual_fqe(fqe_wf: Wavefunction, fqe_ham: RestrictedHamiltonian,
                          norbs: int) -> np.ndarray:
    """Get the ACSE block by using reduced density operators that are Sz spin
    adapted

    R^{ij}_{lk} = <psi | [i^ j^ k l, A] | psi>

    alpha-alpha, beta-beta, alpha-beta, and beta-alpha blocks

    we do not compression over alpha-alpha or beta-beta so these are still
    norbs**2 in linear dimension. In other words, we do computation on
    elements we know should be zero. This is for simplicity in the code.

    Args:
        fqe_wf:  fqe.Wavefunction object to calculate expectation value with
        fqe_ham: fqe.RestrictedHamiltonian operator corresponding to a chemical
                 Hamiltonian
        norbs: Number of orbitals. Number of spatial orbitals

    Returns:
        Gradient of the i^ j^ k l operator
    """
    acse_aa = np.zeros((norbs, norbs, norbs, norbs), dtype=np.complex128)
    acse_bb = np.zeros((norbs, norbs, norbs, norbs), dtype=np.complex128)
    acse_ab = np.zeros((norbs, norbs, norbs, norbs), dtype=np.complex128)

    fqe_appA = fqe_wf.apply(fqe_ham)

    for p, q, r, s in product(range(norbs), repeat=4):
        # alpha-alpha block real
        if p != q and r != s:
            rdo = ((2 * p, 1), (2 * q, 1), (2 * r, 0), (2 * s, 0))
            rdo = 1j * (of.FermionOperator(rdo) -
                        of.hermitian_conjugated(of.FermionOperator(rdo)))
            val1 = fqe.util.vdot(fqe_appA, fqe_wf.apply(rdo))
            val2 = np.conjugate(val1)
            acse_aa[p, q, r, s] = (val2 - val1) / 2j

            # alpha-alpha block imag
            rdo = ((2 * p, 1), (2 * q, 1), (2 * r, 0), (2 * s, 0))
            rdo = of.FermionOperator(rdo) + of.hermitian_conjugated(
                of.FermionOperator(rdo))
            val1 = fqe.util.vdot(fqe_appA, fqe_wf.apply(rdo))
            val2 = np.conjugate(val1)
            acse_aa[p, q, r, s] += (val2 - val1) / 2

            # beta-beta block real
            rdo = (
                (2 * p + 1, 1),
                (2 * q + 1, 1),
                (2 * r + 1, 0),
                (2 * s + 1, 0),
            )
            rdo = 1j * (of.FermionOperator(rdo) -
                        of.hermitian_conjugated(of.FermionOperator(rdo)))
            val1 = fqe.util.vdot(fqe_appA, fqe_wf.apply(rdo))
            val2 = np.conjugate(val1)
            acse_bb[p, q, r, s] += (val2 - val1) / 2j

            # beta-beta block imag
            rdo = (
                (2 * p + 1, 1),
                (2 * q + 1, 1),
                (2 * r + 1, 0),
                (2 * s + 1, 0),
            )
            rdo = of.FermionOperator(rdo) + of.hermitian_conjugated(
                of.FermionOperator(rdo))
            val1 = fqe.util.vdot(fqe_appA, fqe_wf.apply(rdo))
            val2 = np.conjugate(val1)
            acse_bb[p, q, r, s] += (val2 - val1) / 2

        # alpha-beta block real
        rdo = ((2 * p, 1), (2 * q + 1, 1), (2 * r + 1, 0), (2 * s, 0))
        rdo = 1j * (of.FermionOperator(rdo) -
                    of.hermitian_conjugated(of.FermionOperator(rdo)))
        val1 = fqe.util.vdot(fqe_appA, fqe_wf.apply(rdo))
        val2 = np.conjugate(val1)
        acse_ab[p, q, r, s] += (val2 - val1) / 2j

        # alpha-beta block imag
        rdo = ((2 * p, 1), (2 * q + 1, 1), (2 * r + 1, 0), (2 * s, 0))
        rdo = of.FermionOperator(rdo) + of.hermitian_conjugated(
            of.FermionOperator(rdo))
        val1 = fqe.util.vdot(fqe_appA, fqe_wf.apply(rdo))
        val2 = np.conjugate(val1)  # fqe.util.vdot(fqe_wf.apply(rdo), fqe_appA)
        acse_ab[p, q, r, s] += (val2 - val1) / 2

    # unroll residual blocks into full matrix
    acse_residual = np.zeros((2 * norbs, 2 * norbs, 2 * norbs, 2 * norbs),
                             dtype=np.complex128)
    acse_residual[::2, ::2, ::2, ::2] = acse_aa
    acse_residual[1::2, 1::2, 1::2, 1::2] = acse_bb
    acse_residual[::2, 1::2, 1::2, ::2] = acse_ab
    acse_residual[::2, 1::2, ::2, 1::2] = np.einsum("ijkl->ijlk", -acse_ab)
    acse_residual[1::2, ::2, ::2, 1::2] = np.einsum("ijkl->jilk", acse_ab)
    acse_residual[1::2, ::2, 1::2, ::2] = np.einsum(
        "ijkl->ijlk", -acse_residual[1::2, ::2, ::2, 1::2])

    return acse_residual


def get_tpdm_grad_fqe(fqe_wf, acse_res_tensor, norbs):
    r"""Compute the acse gradient  <psi [rdo, A] psi>

    alpha-alpha, beta-beta, alpha-beta, and beta-alpha blocks

    d D^{pq}_{rs} / d \lambda = <psi(lamba)|[p^ q^ s r, A]| psi(lambda)>

    Args:
        fqe_wf:  fqe.Wavefunction object to calculate expectation value with
        acse_res_tensor: fqe.RestrictedHamiltonian operator corresponding to a chemical
                         Hamiltonian
        norbs: Number of orbitals. Number of spatial orbitals

    Returns:
        Gradient of the i^ j^ k l operator
    """
    four_tensor_counter = np.zeros_like(acse_res_tensor)
    s_ops = []
    s_op_total = of.FermionOperator()
    for p, q, r, s in product(range(2 * norbs), repeat=4):
        if p * 2 * norbs + q >= s * 2 * norbs + r:
            if p * 2 * norbs + q != s * 2 * norbs + r:
                four_tensor_counter[p, q, r, s] += 1
                four_tensor_counter[s, r, q, p] += 1
                if abs(acse_res_tensor[p, q, r, s]) > 1.0e-12:
                    op = ((p, 1), (q, 1), (r, 0), (s, 0))
                    fop1 = of.FermionOperator(
                        op, coefficient=acse_res_tensor[p, q, r, s])
                    op = ((s, 1), (r, 1), (q, 0), (p, 0))
                    fop2 = of.FermionOperator(
                        op, coefficient=acse_res_tensor[s, r, q, p])
                    s_ops.append((fop1, fop2))
                    s_op_total += fop2
                    s_op_total += fop1
            else:
                four_tensor_counter[p, q, r, s] += 1
                if abs(acse_res_tensor[p, q, r, s]) > 1.0e-12:
                    op = ((p, 1), (q, 1), (r, 0), (s, 0))
                    fop1 = of.FermionOperator(
                        op, coefficient=acse_res_tensor[p, q, r, s])
                    s_ops.append((fop1, of.FermionOperator()))
                    s_op_total += fop1

    assert np.allclose(four_tensor_counter, 1)

    fqe_appS = copy.deepcopy(fqe_wf)
    fqe_appS.set_wfn("zero")
    for op1, op2 in s_ops:
        fqe_appS += fqe_wf.apply(1j * (op1 + op2))

    acse_aa = np.zeros((norbs, norbs, norbs, norbs), dtype=np.complex128)
    acse_bb = np.zeros((norbs, norbs, norbs, norbs), dtype=np.complex128)
    acse_ab = np.zeros((norbs, norbs, norbs, norbs), dtype=np.complex128)
    for p, q, r, s in product(range(norbs), repeat=4):
        # alpha-beta block real
        rdo = ((2 * p, 1), (2 * q + 1, 1), (2 * r + 1, 0), (2 * s, 0))
        rdo = of.FermionOperator(rdo) + of.hermitian_conjugated(
            of.FermionOperator(rdo))
        fqe_wf_rdo = fqe_wf.apply(rdo)
        val1 = fqe.util.vdot(fqe_appS, fqe_wf_rdo)
        val2 = fqe.util.vdot(fqe_wf_rdo, fqe_appS)
        acse_ab[p, q, r, s] += (val2 - val1) / 2j

        # alpha-beta block imag
        rdo = ((2 * p, 1), (2 * q + 1, 1), (2 * r + 1, 0), (2 * s, 0))
        rdo = 1j * (of.FermionOperator(rdo) -
                    of.hermitian_conjugated(of.FermionOperator(rdo)))
        fqe_wf_rdo = fqe_wf.apply(rdo)
        val3 = fqe.util.vdot(fqe_appS, fqe_wf_rdo)
        val4 = fqe.util.vdot(fqe_wf_rdo, fqe_appS)
        acse_ab[p, q, r, s] += (val4 - val3) / -2

        # alpha-alpha block real
        rdo = ((2 * p, 1), (2 * q, 1), (2 * r, 0), (2 * s, 0))
        rdo = of.FermionOperator(rdo) + of.hermitian_conjugated(
            of.FermionOperator(rdo))
        fqe_wf_rdo = fqe_wf.apply(rdo)
        val1 = fqe.util.vdot(fqe_appS, fqe_wf_rdo)
        val2 = fqe.util.vdot(fqe_wf_rdo, fqe_appS)
        acse_aa[p, q, r, s] += (val2 - val1) / 2j

        # alpha-alpha block imag
        rdo = ((2 * p, 1), (2 * q, 1), (2 * r, 0), (2 * s, 0))
        rdo = 1j * (of.FermionOperator(rdo) -
                    of.hermitian_conjugated(of.FermionOperator(rdo)))
        fqe_wf_rdo = fqe_wf.apply(rdo)
        val3 = fqe.util.vdot(fqe_appS, fqe_wf_rdo)
        val4 = fqe.util.vdot(fqe_wf_rdo, fqe_appS)
        acse_aa[p, q, r, s] += (val4 - val3) / -2

        # beta-beta block real
        rdo = ((2 * p + 1, 1), (2 * q + 1, 1), (2 * r + 1, 0), (2 * s + 1, 0))
        rdo = of.FermionOperator(rdo) + of.hermitian_conjugated(
            of.FermionOperator(rdo))

        fqe_wf_rdo = fqe_wf.apply(rdo)
        val1 = fqe.util.vdot(fqe_appS, fqe_wf_rdo)
        val2 = fqe.util.vdot(fqe_wf_rdo, fqe_appS)
        acse_bb[p, q, r, s] += (val2 - val1) / 2j

        # beta-beta block imag
        rdo = ((2 * p + 1, 1), (2 * q + 1, 1), (2 * r + 1, 0), (2 * s + 1, 0))
        rdo = 1j * (of.FermionOperator(rdo) -
                    of.hermitian_conjugated(of.FermionOperator(rdo)))
        fqe_wf_rdo = fqe_wf.apply(rdo)
        val3 = fqe.util.vdot(fqe_appS, fqe_wf_rdo)
        val4 = fqe.util.vdot(fqe_wf_rdo, fqe_appS)
        acse_bb[p, q, r, s] += (val4 - val3) / -2

    # unroll residual blocks into full matrix
    acse_residual = np.zeros((2 * norbs, 2 * norbs, 2 * norbs, 2 * norbs),
                             dtype=np.complex128)
    acse_residual[::2, ::2, ::2, ::2] = acse_aa
    acse_residual[1::2, 1::2, 1::2, 1::2] = acse_bb
    acse_residual[::2, 1::2, 1::2, ::2] = acse_ab
    acse_residual[::2, 1::2, ::2, 1::2] = np.einsum("ijkl->ijlk", -acse_ab)
    acse_residual[1::2, ::2, ::2, 1::2] = np.einsum("ijkl->jilk", acse_ab)
    acse_residual[1::2, ::2, 1::2, ::2] = np.einsum(
        "ijkl->ijlk", -acse_residual[1::2, ::2, ::2, 1::2])

    return acse_residual


def _acse_residual_atomic(p, q, r, s, fqe_appA, fqe_wf):
    """Internal function for comuting the residual"""
    rdo = ((2 * p, 1), (2 * q, 1), (2 * r, 0), (2 * s, 0))
    rdo = 1j * (of.FermionOperator(rdo) -
                of.hermitian_conjugated(of.FermionOperator(rdo)))
    val1 = fqe.util.vdot(fqe_appA, fqe_wf.apply(rdo))
    val2 = fqe.util.vdot(fqe_wf.apply(rdo), fqe_appA)
    acse_aa_i = (val2 - val1) / 2j

    # alpha-alpha block imag
    rdo = ((2 * p, 1), (2 * q, 1), (2 * r, 0), (2 * s, 0))
    rdo = of.FermionOperator(rdo) + of.hermitian_conjugated(
        of.FermionOperator(rdo))
    val1 = fqe.util.vdot(fqe_appA, fqe_wf.apply(rdo))
    val2 = fqe.util.vdot(fqe_wf.apply(rdo), fqe_appA)
    acse_aa_r = (val2 - val1) / 2

    # beta-beta block real
    rdo = ((2 * p + 1, 1), (2 * q + 1, 1), (2 * r + 1, 0), (2 * s + 1, 0))
    rdo = 1j * (of.FermionOperator(rdo) -
                of.hermitian_conjugated(of.FermionOperator(rdo)))
    val1 = fqe.util.vdot(fqe_appA, fqe_wf.apply(rdo))
    val2 = fqe.util.vdot(fqe_wf.apply(rdo), fqe_appA)
    acse_bb_i = (val2 - val1) / 2j

    # beta-beta block imag
    rdo = ((2 * p + 1, 1), (2 * q + 1, 1), (2 * r + 1, 0), (2 * s + 1, 0))
    rdo = of.FermionOperator(rdo) + of.hermitian_conjugated(
        of.FermionOperator(rdo))
    val1 = fqe.util.vdot(fqe_appA, fqe_wf.apply(rdo))
    val2 = fqe.util.vdot(fqe_wf.apply(rdo), fqe_appA)
    acse_bb_r = (val2 - val1) / 2

    # alpha-beta block real
    rdo = ((2 * p, 1), (2 * q + 1, 1), (2 * r + 1, 0), (2 * s, 0))
    rdo = 1j * (of.FermionOperator(rdo) -
                of.hermitian_conjugated(of.FermionOperator(rdo)))
    val1 = fqe.util.vdot(fqe_appA, fqe_wf.apply(rdo))
    val2 = fqe.util.vdot(fqe_wf.apply(rdo), fqe_appA)
    acse_ab_i = (val2 - val1) / 2j

    # alpha-beta block imag
    rdo = ((2 * p, 1), (2 * q + 1, 1), (2 * r + 1, 0), (2 * s, 0))
    rdo = of.FermionOperator(rdo) + of.hermitian_conjugated(
        of.FermionOperator(rdo))
    val1 = fqe.util.vdot(fqe_appA, fqe_wf.apply(rdo))
    val2 = fqe.util.vdot(fqe_wf.apply(rdo), fqe_appA)
    acse_ab_r = (val2 - val1) / 2
    return (
        p,
        q,
        r,
        s,
        acse_aa_i,
        acse_aa_r,
        acse_bb_i,
        acse_bb_r,
        acse_ab_i,
        acse_ab_r,
    )


def get_acse_residual_fqe_parallel(fqe_wf, fqe_ham, norbs):
    """Get the ACSE block by using reduced density operators that are Sz spin
    adapted

    R^{ij}_{lk} = <psi | [i^ j^ k l, A] | psi>

    alpha-alpha, beta-beta, alpha-beta, and beta-alpha blocks

    we do not compression over alpha-alpha or beta-beta so these are still
    norbs**2 in linear dimension. In other words, we do computation on
    elements we know should be zero. This is for simplicity in the code.

    Args:
        fqe_wf:  fqe.Wavefunction object to calculate expectation value with
        fqe_ham: fqe.RestrictedHamiltonian operator corresponding to a chemical
                 Hamiltonian
        norbs: Number of orbitals. Number of spatial orbitals

    Returns:
        Gradient of the i^ j^ k l operator
    """
    if not PARALLELIZABLE:
        raise ImportError("Joblib is not available")

    acse_aa = np.zeros((norbs, norbs, norbs, norbs), dtype=np.complex128)
    acse_bb = np.zeros((norbs, norbs, norbs, norbs), dtype=np.complex128)
    acse_ab = np.zeros((norbs, norbs, norbs, norbs), dtype=np.complex128)

    fqe_appA = fqe_wf.apply(fqe_ham)

    with Parallel(n_jobs=11, batch_size=norbs) as parallel:
        result = parallel(
            delayed(_acse_residual_atomic)(p, q, r, s, fqe_appA, fqe_wf)
            for p, q, r, s in product(range(norbs), repeat=4))

    for resval in result:
        p, q, r, s = resval[:4]
        acse_aa[p, q, r, s] = resval[4] + resval[5]
        acse_bb[p, q, r, s] = resval[6] + resval[7]
        acse_ab[p, q, r, s] = resval[8] + resval[9]
        # alpha-alpha block real

    # unroll residual blocks into full matrix
    acse_residual = np.zeros((2 * norbs, 2 * norbs, 2 * norbs, 2 * norbs),
                             dtype=np.complex128)
    acse_residual[::2, ::2, ::2, ::2] = acse_aa
    acse_residual[1::2, 1::2, 1::2, 1::2] = acse_bb
    acse_residual[::2, 1::2, 1::2, ::2] = acse_ab
    acse_residual[::2, 1::2, ::2, 1::2] = np.einsum("ijkl->ijlk", -acse_ab)
    acse_residual[1::2, ::2, ::2, 1::2] = np.einsum("ijkl->jilk", acse_ab)
    acse_residual[1::2, ::2, 1::2, ::2] = np.einsum(
        "ijkl->ijlk", -acse_residual[1::2, ::2, ::2, 1::2])

    return acse_residual


def _get_tpdm_grad_fqe_atomic(p, q, r, s, fqe_appS, fqe_wf):
    """Internal function for 2-RDM grad parallel"""
    # alpha-beta block real
    rdo = ((2 * p, 1), (2 * q + 1, 1), (2 * r + 1, 0), (2 * s, 0))
    rdo = of.FermionOperator(rdo) + of.hermitian_conjugated(
        of.FermionOperator(rdo))
    fqe_wf_rdo = fqe_wf.apply(rdo)
    val1 = fqe.util.vdot(fqe_appS, fqe_wf_rdo)
    val2 = fqe.util.vdot(fqe_wf_rdo, fqe_appS)
    acse_ab_i = (val2 - val1) / 2j

    # alpha-beta block imag
    rdo = ((2 * p, 1), (2 * q + 1, 1), (2 * r + 1, 0), (2 * s, 0))
    rdo = 1j * (of.FermionOperator(rdo) -
                of.hermitian_conjugated(of.FermionOperator(rdo)))
    fqe_wf_rdo = fqe_wf.apply(rdo)
    val3 = fqe.util.vdot(fqe_appS, fqe_wf_rdo)
    val4 = fqe.util.vdot(fqe_wf_rdo, fqe_appS)
    acse_ab_r = (val4 - val3) / -2

    # alpha-alpha block real
    rdo = ((2 * p, 1), (2 * q, 1), (2 * r, 0), (2 * s, 0))
    rdo = of.FermionOperator(rdo) + of.hermitian_conjugated(
        of.FermionOperator(rdo))
    fqe_wf_rdo = fqe_wf.apply(rdo)
    val1 = fqe.util.vdot(fqe_appS, fqe_wf_rdo)
    val2 = fqe.util.vdot(fqe_wf_rdo, fqe_appS)
    acse_aa_i = (val2 - val1) / 2j

    # alpha-alpha block imag
    rdo = ((2 * p, 1), (2 * q, 1), (2 * r, 0), (2 * s, 0))
    rdo = 1j * (of.FermionOperator(rdo) -
                of.hermitian_conjugated(of.FermionOperator(rdo)))
    fqe_wf_rdo = fqe_wf.apply(rdo)
    val3 = fqe.util.vdot(fqe_appS, fqe_wf_rdo)
    val4 = fqe.util.vdot(fqe_wf_rdo, fqe_appS)
    acse_aa_r = (val4 - val3) / -2

    # beta-beta block real
    rdo = ((2 * p + 1, 1), (2 * q + 1, 1), (2 * r + 1, 0), (2 * s + 1, 0))
    rdo = of.FermionOperator(rdo) + of.hermitian_conjugated(
        of.FermionOperator(rdo))

    fqe_wf_rdo = fqe_wf.apply(rdo)
    val1 = fqe.util.vdot(fqe_appS, fqe_wf_rdo)
    val2 = fqe.util.vdot(fqe_wf_rdo, fqe_appS)
    acse_bb_i = (val2 - val1) / 2j

    # beta-beta block imag
    rdo = ((2 * p + 1, 1), (2 * q + 1, 1), (2 * r + 1, 0), (2 * s + 1, 0))
    rdo = 1j * (of.FermionOperator(rdo) -
                of.hermitian_conjugated(of.FermionOperator(rdo)))
    fqe_wf_rdo = fqe_wf.apply(rdo)
    val3 = fqe.util.vdot(fqe_appS, fqe_wf_rdo)
    val4 = fqe.util.vdot(fqe_wf_rdo, fqe_appS)
    acse_bb_r = (val4 - val3) / -2
    return (
        p,
        q,
        r,
        s,
        acse_aa_r,
        acse_aa_i,
        acse_bb_r,
        acse_bb_i,
        acse_ab_r,
        acse_ab_i,
    )


def get_tpdm_grad_fqe_parallel(fqe_wf, acse_res_tensor, norbs):
    r"""Compute the acse gradient  <psi [rdo, A] psi>

    d D^{pq}_{rs} / d \lambda = <psi(lamba)|[p^ q^ s r, A]| psi(lambda)>

    Args:
        fqe_wf:  fqe.Wavefunction object to calculate expectation value with
        fqe_ham: fqe.RestrictedHamiltonian operator corresponding to a chemical
                 Hamiltonian
        norbs: Number of orbitals. Number of spatial orbitals

    Returns:
        Gradient of the i^ j^ k l operator
    """
    if not PARALLELIZABLE:
        raise ImportError("Joblib was not imported")
    four_tensor_counter = np.zeros_like(acse_res_tensor)
    s_ops = []
    # s_op_total = of.FermionOperator()
    for p, q, r, s in product(range(2 * norbs), repeat=4):
        if p * 2 * norbs + q >= s * 2 * norbs + r:
            if p * 2 * norbs + q != s * 2 * norbs + r:
                four_tensor_counter[p, q, r, s] += 1
                four_tensor_counter[s, r, q, p] += 1
                if abs(acse_res_tensor[p, q, r, s]) > 1.0e-12:
                    op = ((p, 1), (q, 1), (r, 0), (s, 0))
                    fop1 = of.FermionOperator(
                        op, coefficient=acse_res_tensor[p, q, r, s])
                    op = ((s, 1), (r, 1), (q, 0), (p, 0))
                    fop2 = of.FermionOperator(
                        op, coefficient=acse_res_tensor[s, r, q, p])
                    s_ops.append((fop1, fop2))
                    # s_op_total += fop2
                    # s_op_total += fop1
            else:
                four_tensor_counter[p, q, r, s] += 1
                if abs(acse_res_tensor[p, q, r, s]) > 1.0e-12:
                    op = ((p, 1), (q, 1), (r, 0), (s, 0))
                    fop1 = of.FermionOperator(
                        op, coefficient=acse_res_tensor[p, q, r, s])
                    s_ops.append((fop1, of.FermionOperator()))
                    # s_op_total += fop1

    assert np.allclose(four_tensor_counter, 1)

    fqe_appS = copy.deepcopy(fqe_wf)
    fqe_appS.set_wfn("zero")
    for op1, op2 in s_ops:
        fqe_appS += fqe_wf.apply(1j * (op1 + op2))

    acse_aa = np.zeros((norbs, norbs, norbs, norbs), dtype=np.complex128)
    acse_bb = np.zeros((norbs, norbs, norbs, norbs), dtype=np.complex128)
    acse_ab = np.zeros((norbs, norbs, norbs, norbs), dtype=np.complex128)
    with Parallel(n_jobs=-1) as parallel:
        result = parallel(
            delayed(_get_tpdm_grad_fqe_atomic)(p, q, r, s, fqe_appS, fqe_wf)
            for p, q, r, s in product(range(norbs), repeat=4))

    for resval in result:
        p, q, r, s = resval[:4]
        acse_aa[p, q, r, s] = resval[4] + resval[5]
        acse_bb[p, q, r, s] = resval[6] + resval[7]
        acse_ab[p, q, r, s] = resval[8] + resval[9]
        # alpha-alpha block real

    # unroll residual blocks into full matrix
    acse_residual = np.zeros((2 * norbs, 2 * norbs, 2 * norbs, 2 * norbs),
                             dtype=np.complex128)
    acse_residual[::2, ::2, ::2, ::2] = acse_aa
    acse_residual[1::2, 1::2, 1::2, 1::2] = acse_bb
    acse_residual[::2, 1::2, 1::2, ::2] = acse_ab
    acse_residual[::2, 1::2, ::2, 1::2] = np.einsum("ijkl->ijlk", -acse_ab)
    acse_residual[1::2, ::2, ::2, 1::2] = np.einsum("ijkl->jilk", acse_ab)
    acse_residual[1::2, ::2, 1::2, ::2] = np.einsum(
        "ijkl->ijlk", -acse_residual[1::2, ::2, ::2, 1::2])

    return acse_residual


def two_rdo_commutator(two_body_tensor: np.ndarray, tpdm: np.ndarray,
                       d3: np.ndarray) -> np.ndarray:
    r"""
    Calculate <psi | [p^ q^ r s, A] | psi>  where A two-body operator

    A = \sum_{ijkl}A^{ij}_{lk}i^ j^ k l

    where A^{ij}_{lk} is a 4-index tensor.  There is no restriction on the
    structure of A.

    Args:
        two_body_tensor: 4-tensor for the coefficients of A
        tpdm: spin-orbital two-RDM p^ q^ r s corresponding to (1'2'2 1)
        d3: spin-orbital three-RDM p^ q^ r^ s t u corresponding to (1'2'3'32 1)
    """
    dim = tpdm.shape[0]
    tensor_of_expectation = np.zeros(tuple([dim] * 4), dtype=tpdm.dtype)
    for p, q, r, s in product(range(dim), repeat=4):
        commutator_expectation = 0.
        #   (  -1.00000) kdelta(i,r) kdelta(j,s) cre(p) cre(q) des(k) des(l)
        commutator_expectation += -1.0 * np.einsum('kl,kl',
                                                   two_body_tensor[r, s, :, :],
                                                   tpdm[p, q, :, :],
                                                   optimize=True)

        #   (   1.00000) kdelta(i,s) kdelta(j,r) cre(p) cre(q) des(k) des(l)
        commutator_expectation += 1.0 * np.einsum('kl,kl',
                                                  two_body_tensor[s, r, :, :],
                                                  tpdm[p, q, :, :],
                                                  optimize=True)

        #   (   1.00000) kdelta(k,p) kdelta(l,q) cre(i) cre(j) des(r) des(s)
        commutator_expectation += 1.0 * np.einsum('ij,ij',
                                                  two_body_tensor[:, :, p, q],
                                                  tpdm[:, :, r, s],
                                                  optimize=True)

        #   (  -1.00000) kdelta(k,q) kdelta(l,p) cre(i) cre(j) des(r) des(s)
        commutator_expectation += -1.0 * np.einsum('ij,ij',
                                                   two_body_tensor[:, :, q, p],
                                                   tpdm[:, :, r, s],
                                                   optimize=True)

        #   (   1.00000) kdelta(i,r) cre(j) cre(p) cre(q) des(k) des(l) des(s)
        commutator_expectation += 1.0 * np.einsum('jkl,jkl',
                                                  two_body_tensor[r, :, :, :],
                                                  d3[:, p, q, :, :, s],
                                                  optimize=True)

        #   (  -1.00000) kdelta(i,s) cre(j) cre(p) cre(q) des(k) des(l) des(r)
        commutator_expectation += -1.0 * np.einsum('jkl,jkl',
                                                   two_body_tensor[s, :, :, :],
                                                   d3[:, p, q, :, :, r],
                                                   optimize=True)

        #   (  -1.00000) kdelta(j,r) cre(i) cre(p) cre(q) des(k) des(l) des(s)
        commutator_expectation += -1.0 * np.einsum('ikl,ikl',
                                                   two_body_tensor[:, r, :, :],
                                                   d3[:, p, q, :, :, s],
                                                   optimize=True)

        #   (   1.00000) kdelta(j,s) cre(i) cre(p) cre(q) des(k) des(l) des(r)
        commutator_expectation += 1.0 * np.einsum('ikl,ikl',
                                                  two_body_tensor[:, s, :, :],
                                                  d3[:, p, q, :, :, r],
                                                  optimize=True)

        #   (  -1.00000) kdelta(k,p) cre(i) cre(j) cre(q) des(l) des(r) des(s)
        commutator_expectation += -1.0 * np.einsum('ijl,ijl',
                                                   two_body_tensor[:, :, p, :],
                                                   d3[:, :, q, :, r, s],
                                                   optimize=True)

        #   (   1.00000) kdelta(k,q) cre(i) cre(j) cre(p) des(l) des(r) des(s)
        commutator_expectation += 1.0 * np.einsum('ijl,ijl',
                                                  two_body_tensor[:, :, q, :],
                                                  d3[:, :, p, :, r, s],
                                                  optimize=True)

        #   (   1.00000) kdelta(l,p) cre(i) cre(j) cre(q) des(k) des(r) des(s)
        commutator_expectation += 1.0 * np.einsum('ijk,ijk',
                                                  two_body_tensor[:, :, :, p],
                                                  d3[:, :, q, :, r, s],
                                                  optimize=True)

        #   (  -1.00000) kdelta(l,q) cre(i) cre(j) cre(p) des(k) des(r) des(s)
        commutator_expectation += -1.0 * np.einsum('ijk,ijk',
                                                   two_body_tensor[:, :, :, q],
                                                   d3[:, :, p, :, r, s],
                                                   optimize=True)
        tensor_of_expectation[p, q, r, s] = commutator_expectation

    return tensor_of_expectation


def two_rdo_commutator_symm(two_body_tensor: np.ndarray, tpdm: np.ndarray,
                            d3: np.ndarray) -> np.ndarray:
    r"""
    Calculate <psi | [p^ q^ r s, A] | psi>  where A two-body operator

    A = \sum_{ijkl}A^{ij}_{lk}i^ j^ k l

    where A^{ij}_{lk} is antisymmetric and hermitian

    Args:
        two_body_tensor: 4-tensor for the coefficients of A
        tpdm: spin-orbital two-RDM p^ q^ r s corresponding to (1'2'2 1)
        d3: spin-orbital three-RDM p^ q^ r^ s t u corresponding to (1'2'3'32 1)
    """
    dim = tpdm.shape[0]
    tensor_of_expectation = np.zeros(tuple([dim] * 4), dtype=tpdm.dtype)
    k2 = two_body_tensor.transpose(0, 1, 3, 2)
    for p, q, r, s in product(range(dim), repeat=4):
        commutator_expectation = 0.
        #   (  -2.00000) k2(p,q,a,b) cre(a) cre(b) des(r) des(s)
        commutator_expectation += -2. * np.einsum('ab,ab', k2[p, q, :, :],
                                                  tpdm[:, :, r, s])

        #   (   2.00000) k2(r,s,a,b) cre(p) cre(q) des(a) des(b)
        commutator_expectation += 2. * np.einsum('ab,ab', k2[r, s, :, :],
                                                 tpdm[p, q, :, :])

        #   (   2.00000) k2(p,a,b,c) cre(q) cre(b) cre(c) des(r) des(s) des(a)
        commutator_expectation += 2. * np.einsum('abc,bca', k2[p, :, :, :],
                                                 d3[q, :, :, r, s, :])

        #   (  -2.00000) k2(q,a,b,c) cre(p) cre(b) cre(c) des(r) des(s) des(a)
        commutator_expectation += -2. * np.einsum('abc,bca', k2[q, :, :, :],
                                                  d3[p, :, :, r, s, :])

        #   (  -2.00000) k2(r,a,b,c) cre(p) cre(q) cre(a) des(s) des(b) des(c)
        commutator_expectation += -2. * np.einsum('abc,abc', k2[r, :, :, :],
                                                  d3[p, q, :, s, :, :])

        #   (   2.00000) k2(s,a,b,c) cre(p) cre(q) cre(a) des(r) des(b) des(c)
        commutator_expectation += 2. * np.einsum('abc,abc', k2[s, :, :, :],
                                                 d3[p, q, :, r, :, :])

        tensor_of_expectation[p, q, r, s] = commutator_expectation
    return tensor_of_expectation


def two_rdo_commutator_antisymm(two_body_tensor: np.ndarray, tpdm: np.ndarray,
                                d3: np.ndarray) -> np.ndarray:
    r"""
    Calculate <psi | [p^ q^ r s, A] | psi>  where A two-body operator

    A = \sum_{ijkl}A^{ij}_{lk}i^ j^ k l

    where A^{ij}_{lk} is antisymmetric and antihermitian

    Args:
        two_body_tensor: 4-tensor for the coefficients of A
        tpdm: spin-orbital two-RDM p^ q^ r s corresponding to (1'2'2 1)
        d3: spin-orbital three-RDM p^ q^ r^ s t u corresponding to (1'2'3'32 1)
    """
    dim = tpdm.shape[0]
    tensor_of_expectation = np.zeros(tuple([dim] * 4), dtype=tpdm.dtype)
    k2 = two_body_tensor.transpose(0, 1, 3, 2)
    for p, q, r, s in product(range(dim), repeat=4):
        commutator_expectation = 0.
        #   (  2.00000) k2(p,q,a,b) cre(a) cre(b) des(r) des(s)
        commutator_expectation += 2. * np.einsum('ab,ab', k2[p, q, :, :],
                                                 tpdm[:, :, r, s])

        #   (   2.00000) k2(r,s,a,b) cre(p) cre(q) des(a) des(b)
        commutator_expectation += 2. * np.einsum('ab,ab', k2[r, s, :, :],
                                                 tpdm[p, q, :, :])

        #   (  -2.00000) k2(p,a,b,c) cre(q) cre(b) cre(c) des(r) des(s) des(a)
        commutator_expectation += -2. * np.einsum('abc,bca', k2[p, :, :, :],
                                                  d3[q, :, :, r, s, :])

        #   (  2.00000) k2(q,a,b,c) cre(p) cre(b) cre(c) des(r) des(s) des(a)
        commutator_expectation += 2. * np.einsum('abc,bca', k2[q, :, :, :],
                                                 d3[p, :, :, r, s, :])

        #   (  -2.00000) k2(r,a,b,c) cre(p) cre(q) cre(a) des(s) des(b) des(c)
        commutator_expectation += -2. * np.einsum('abc,abc', k2[r, :, :, :],
                                                  d3[p, q, :, s, :, :])

        #   (   2.00000) k2(s,a,b,c) cre(p) cre(q) cre(a) des(r) des(b) des(c)
        commutator_expectation += 2. * np.einsum('abc,abc', k2[s, :, :, :],
                                                 d3[p, q, :, r, :, :])
        tensor_of_expectation[p, q, r, s] = commutator_expectation
    return tensor_of_expectation


def one_rdo_commutator_symm(two_body_tensor: np.ndarray,
                            tpdm: np.ndarray) -> np.ndarray:
    r"""
    Calculate <psi | [p^ q, A] | psi> where A is a two-body operator
     A = \sum_{ijkl}A^{ij}_{lk}i^ j^ k l

    where A^{ij}_{lk} is antisymmetric and hermitian

    Args:
        two_body_tensor: 4-tensor for the coefficients of A
        tpdm: spin-orbital two-RDM p^ q^ r s corresponding to (1'2'2 1)
    """
    dim = tpdm.shape[0]
    tensor_of_expectation = np.zeros(tuple([dim] * 2), dtype=tpdm.dtype)
    k2 = two_body_tensor.transpose(0, 1, 3, 2)
    for p, q in product(range(dim), repeat=2):
        commutator_expectation = 0.
        #   (   2.00000) k2(p,a,b,c) cre(b) cre(c) des(q) des(a)
        commutator_expectation += 2.0 * np.einsum('abc,bca', k2[p, :, :, :],
                                                  tpdm[:, :, q, :])

        #   (  -2.00000) k2(q,a,b,c) cre(p) cre(a) des(b) des(c)
        commutator_expectation += -2.0 * np.einsum('abc,abc', k2[q, :, :, :],
                                                   tpdm[p, :, :, :])
        tensor_of_expectation[p, q] = commutator_expectation
    return tensor_of_expectation
