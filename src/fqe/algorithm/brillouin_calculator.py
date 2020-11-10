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
"""TODO: Add description of module."""

import copy
from itertools import product

import numpy as np

import openfermion as of
import fqe

try:
    from joblib import Parallel, delayed

    PARALLELIZABLE = True
except ImportError:
    PARALLELIZABLE = False


def get_fermion_op(coeff_tensor):
    """Returns an openfermion.FermionOperator from the given coeff_tensor.

    Args:
        coeff_tensor: TODO: Add description.
    """
    if len(coeff_tensor.shape) not in (2, 4):
        raise ValueError(
            "Arg `coeff_tensor` should have dimension 2 or 4 but has dimension"
            f" {len(coeff_tensor.shape)}."
        )

    if len(coeff_tensor.shape) == 4:
        nso = coeff_tensor.shape[0]
        fermion_op = of.FermionOperator()
        for p, q, r, s in product(range(nso), repeat=4):
            op = ((p, 1), (q, 1), (r, 0), (s, 0))
            fop = of.FermionOperator(op, coefficient=coeff_tensor[p, q, r, s])
            fermion_op += fop
        return fermion_op

    if len(coeff_tensor.shape) == 2:
        nso = coeff_tensor.shape[0]
        fermion_op = of.FermionOperator()
        for p, q in product(range(nso), repeat=2):
            op = ((p, 1), (q, 0))
            fop = of.FermionOperator(op, coefficient=coeff_tensor[p, q])
            fermion_op += fop
        return fermion_op


def get_acse_residual_fqe(fqe_wf, fqe_ham, norbs):
    """Get the ACSE block by using reduced density operators that are Sz spin
    adapted

    alpha-alpha, beta-beta, alpha-beta, and beta-alpha blocks

    we do not compression over alpha-alpha or beta-beta so these are still
    norbs**2 in linear dimension. In other words, we do computation on
    elements we know should be zero. This is for simplicity in the code.

    TODO: Document args and return value.
    Args:
        fqe_wf:
        fqe_ham:
        norbs: Number of orbitals.

    Returns:

    """
    acse_aa = np.zeros((norbs, norbs, norbs, norbs), dtype=np.complex128)
    acse_bb = np.zeros((norbs, norbs, norbs, norbs), dtype=np.complex128)
    acse_ab = np.zeros((norbs, norbs, norbs, norbs), dtype=np.complex128)

    fqe_appA = fqe_wf.apply(fqe_ham)

    for p, q, r, s in product(range(norbs), repeat=4):
        # alpha-alpha block real
        if p != q and r != s:
            rdo = ((2 * p, 1), (2 * q, 1), (2 * r, 0), (2 * s, 0))
            rdo = 1j * (
                of.FermionOperator(rdo)
                - of.hermitian_conjugated(of.FermionOperator(rdo))
            )
            val1 = fqe.util.vdot(fqe_appA, fqe_wf.apply(rdo))
            val2 = np.conjugate(val1)
            acse_aa[p, q, r, s] = (val2 - val1) / 2j

            # alpha-alpha block imag
            rdo = ((2 * p, 1), (2 * q, 1), (2 * r, 0), (2 * s, 0))
            rdo = of.FermionOperator(rdo) + of.hermitian_conjugated(
                of.FermionOperator(rdo)
            )
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
            rdo = 1j * (
                of.FermionOperator(rdo)
                - of.hermitian_conjugated(of.FermionOperator(rdo))
            )
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
                of.FermionOperator(rdo)
            )
            val1 = fqe.util.vdot(fqe_appA, fqe_wf.apply(rdo))
            val2 = np.conjugate(val1)
            acse_bb[p, q, r, s] += (val2 - val1) / 2

        # alpha-beta block real
        rdo = ((2 * p, 1), (2 * q + 1, 1), (2 * r + 1, 0), (2 * s, 0))
        rdo = 1j * (
            of.FermionOperator(rdo)
            - of.hermitian_conjugated(of.FermionOperator(rdo))
        )
        val1 = fqe.util.vdot(fqe_appA, fqe_wf.apply(rdo))
        val2 = np.conjugate(val1)
        acse_ab[p, q, r, s] += (val2 - val1) / 2j

        # alpha-beta block imag
        rdo = ((2 * p, 1), (2 * q + 1, 1), (2 * r + 1, 0), (2 * s, 0))
        rdo = of.FermionOperator(rdo) + of.hermitian_conjugated(
            of.FermionOperator(rdo)
        )
        val1 = fqe.util.vdot(fqe_appA, fqe_wf.apply(rdo))
        val2 = np.conjugate(val1)  # fqe.util.vdot(fqe_wf.apply(rdo), fqe_appA)
        acse_ab[p, q, r, s] += (val2 - val1) / 2

    # unroll residual blocks into full matrix
    acse_residual = np.zeros(
        (2 * norbs, 2 * norbs, 2 * norbs, 2 * norbs), dtype=np.complex128
    )
    acse_residual[::2, ::2, ::2, ::2] = acse_aa
    acse_residual[1::2, 1::2, 1::2, 1::2] = acse_bb
    acse_residual[::2, 1::2, 1::2, ::2] = acse_ab
    acse_residual[::2, 1::2, ::2, 1::2] = np.einsum("ijkl->ijlk", -acse_ab)
    acse_residual[1::2, ::2, ::2, 1::2] = np.einsum("ijkl->jilk", acse_ab)
    acse_residual[1::2, ::2, 1::2, ::2] = np.einsum(
        "ijkl->ijlk", -acse_residual[1::2, ::2, ::2, 1::2]
    )

    return acse_residual


def get_tpdm_grad_fqe(fqe_wf, acse_res_tensor, norbs):
    """Compute the acse gradient  <psi [rdo, A] psi>

    alpha-alpha, beta-beta, alpha-beta, and beta-alpha blocks

    we do not compression over alpha-alpha or beta-beta so these are still
    norbs**2 in linear dimension. In other words, we do computation on elements
    we know should be zero. This is for simplicity in the code.

    TODO: Document args and return value.
    Args:
        fqe_wf:
        acse_res_tensor:
        norbs: Number of orbitals.

    Returns:

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
                        op, coefficient=acse_res_tensor[p, q, r, s]
                    )
                    op = ((s, 1), (r, 1), (q, 0), (p, 0))
                    fop2 = of.FermionOperator(
                        op, coefficient=acse_res_tensor[s, r, q, p]
                    )
                    s_ops.append((fop1, fop2))
                    s_op_total += fop2
                    s_op_total += fop1
            else:
                four_tensor_counter[p, q, r, s] += 1
                if abs(acse_res_tensor[p, q, r, s]) > 1.0e-12:
                    op = ((p, 1), (q, 1), (r, 0), (s, 0))
                    fop1 = of.FermionOperator(
                        op, coefficient=acse_res_tensor[p, q, r, s]
                    )
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
            of.FermionOperator(rdo)
        )
        fqe_wf_rdo = fqe_wf.apply(rdo)
        val1 = fqe.util.vdot(fqe_appS, fqe_wf_rdo)
        val2 = fqe.util.vdot(fqe_wf_rdo, fqe_appS)
        acse_ab[p, q, r, s] += (val2 - val1) / 2j

        # alpha-beta block imag
        rdo = ((2 * p, 1), (2 * q + 1, 1), (2 * r + 1, 0), (2 * s, 0))
        rdo = 1j * (
            of.FermionOperator(rdo)
            - of.hermitian_conjugated(of.FermionOperator(rdo))
        )
        fqe_wf_rdo = fqe_wf.apply(rdo)
        val3 = fqe.util.vdot(fqe_appS, fqe_wf_rdo)
        val4 = fqe.util.vdot(fqe_wf_rdo, fqe_appS)
        acse_ab[p, q, r, s] += (val4 - val3) / -2

        # alpha-alpha block real
        rdo = ((2 * p, 1), (2 * q, 1), (2 * r, 0), (2 * s, 0))
        rdo = of.FermionOperator(rdo) + of.hermitian_conjugated(
            of.FermionOperator(rdo)
        )
        fqe_wf_rdo = fqe_wf.apply(rdo)
        val1 = fqe.util.vdot(fqe_appS, fqe_wf_rdo)
        val2 = fqe.util.vdot(fqe_wf_rdo, fqe_appS)
        acse_aa[p, q, r, s] += (val2 - val1) / 2j

        # alpha-alpha block imag
        rdo = ((2 * p, 1), (2 * q, 1), (2 * r, 0), (2 * s, 0))
        rdo = 1j * (
            of.FermionOperator(rdo)
            - of.hermitian_conjugated(of.FermionOperator(rdo))
        )
        fqe_wf_rdo = fqe_wf.apply(rdo)
        val3 = fqe.util.vdot(fqe_appS, fqe_wf_rdo)
        val4 = fqe.util.vdot(fqe_wf_rdo, fqe_appS)
        acse_aa[p, q, r, s] += (val4 - val3) / -2

        # beta-beta block real
        rdo = ((2 * p + 1, 1), (2 * q + 1, 1), (2 * r + 1, 0), (2 * s + 1, 0))
        rdo = of.FermionOperator(rdo) + of.hermitian_conjugated(
            of.FermionOperator(rdo)
        )

        fqe_wf_rdo = fqe_wf.apply(rdo)
        val1 = fqe.util.vdot(fqe_appS, fqe_wf_rdo)
        val2 = fqe.util.vdot(fqe_wf_rdo, fqe_appS)
        acse_bb[p, q, r, s] += (val2 - val1) / 2j

        # beta-beta block imag
        rdo = ((2 * p + 1, 1), (2 * q + 1, 1), (2 * r + 1, 0), (2 * s + 1, 0))
        rdo = 1j * (
            of.FermionOperator(rdo)
            - of.hermitian_conjugated(of.FermionOperator(rdo))
        )
        fqe_wf_rdo = fqe_wf.apply(rdo)
        val3 = fqe.util.vdot(fqe_appS, fqe_wf_rdo)
        val4 = fqe.util.vdot(fqe_wf_rdo, fqe_appS)
        acse_bb[p, q, r, s] += (val4 - val3) / -2

    # unroll residual blocks into full matrix
    acse_residual = np.zeros(
        (2 * norbs, 2 * norbs, 2 * norbs, 2 * norbs), dtype=np.complex128
    )
    acse_residual[::2, ::2, ::2, ::2] = acse_aa
    acse_residual[1::2, 1::2, 1::2, 1::2] = acse_bb
    acse_residual[::2, 1::2, 1::2, ::2] = acse_ab
    acse_residual[::2, 1::2, ::2, 1::2] = np.einsum("ijkl->ijlk", -acse_ab)
    acse_residual[1::2, ::2, ::2, 1::2] = np.einsum("ijkl->jilk", acse_ab)
    acse_residual[1::2, ::2, 1::2, ::2] = np.einsum(
        "ijkl->ijlk", -acse_residual[1::2, ::2, ::2, 1::2]
    )

    return acse_residual


def acse_residual_atomic(p, q, r, s, fqe_appA, fqe_wf):
    """TODO: Add docstring."""
    rdo = ((2 * p, 1), (2 * q, 1), (2 * r, 0), (2 * s, 0))
    rdo = 1j * (
        of.FermionOperator(rdo)
        - of.hermitian_conjugated(of.FermionOperator(rdo))
    )
    val1 = fqe.util.vdot(fqe_appA, fqe_wf.apply(rdo))
    val2 = fqe.util.vdot(fqe_wf.apply(rdo), fqe_appA)
    acse_aa_i = (val2 - val1) / 2j

    # alpha-alpha block imag
    rdo = ((2 * p, 1), (2 * q, 1), (2 * r, 0), (2 * s, 0))
    rdo = of.FermionOperator(rdo) + of.hermitian_conjugated(
        of.FermionOperator(rdo)
    )
    val1 = fqe.util.vdot(fqe_appA, fqe_wf.apply(rdo))
    val2 = fqe.util.vdot(fqe_wf.apply(rdo), fqe_appA)
    acse_aa_r = (val2 - val1) / 2

    # beta-beta block real
    rdo = ((2 * p + 1, 1), (2 * q + 1, 1), (2 * r + 1, 0), (2 * s + 1, 0))
    rdo = 1j * (
        of.FermionOperator(rdo)
        - of.hermitian_conjugated(of.FermionOperator(rdo))
    )
    val1 = fqe.util.vdot(fqe_appA, fqe_wf.apply(rdo))
    val2 = fqe.util.vdot(fqe_wf.apply(rdo), fqe_appA)
    acse_bb_i = (val2 - val1) / 2j

    # beta-beta block imag
    rdo = ((2 * p + 1, 1), (2 * q + 1, 1), (2 * r + 1, 0), (2 * s + 1, 0))
    rdo = of.FermionOperator(rdo) + of.hermitian_conjugated(
        of.FermionOperator(rdo)
    )
    val1 = fqe.util.vdot(fqe_appA, fqe_wf.apply(rdo))
    val2 = fqe.util.vdot(fqe_wf.apply(rdo), fqe_appA)
    acse_bb_r = (val2 - val1) / 2

    # alpha-beta block real
    rdo = ((2 * p, 1), (2 * q + 1, 1), (2 * r + 1, 0), (2 * s, 0))
    rdo = 1j * (
        of.FermionOperator(rdo)
        - of.hermitian_conjugated(of.FermionOperator(rdo))
    )
    val1 = fqe.util.vdot(fqe_appA, fqe_wf.apply(rdo))
    val2 = fqe.util.vdot(fqe_wf.apply(rdo), fqe_appA)
    acse_ab_i = (val2 - val1) / 2j

    # alpha-beta block imag
    rdo = ((2 * p, 1), (2 * q + 1, 1), (2 * r + 1, 0), (2 * s, 0))
    rdo = of.FermionOperator(rdo) + of.hermitian_conjugated(
        of.FermionOperator(rdo)
    )
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

    alpha-alpha, beta-beta, alpha-beta, and beta-alpha blocks

    we do not compression over alpha-alpha or beta-beta so these are still
    norbs**2 in linear dimension. In other words, we do computation on elements
    we know should be zero. This is for simplicity in the code.
    """
    # TODO: Check if parallel has been imported? e.g.
    #  if not PARALLELIZABLE:
    #      raise SomeError
    acse_aa = np.zeros((norbs, norbs, norbs, norbs), dtype=np.complex128)
    acse_bb = np.zeros((norbs, norbs, norbs, norbs), dtype=np.complex128)
    acse_ab = np.zeros((norbs, norbs, norbs, norbs), dtype=np.complex128)

    fqe_appA = fqe_wf.apply(fqe_ham)

    with Parallel(n_jobs=11, batch_size=norbs) as parallel:
        result = parallel(
            delayed(acse_residual_atomic)(p, q, r, s, fqe_appA, fqe_wf)
            for p, q, r, s in product(range(norbs), repeat=4)
        )

    for resval in result:
        p, q, r, s = resval[:4]
        acse_aa[p, q, r, s] = resval[4] + resval[5]
        acse_bb[p, q, r, s] = resval[6] + resval[7]
        acse_ab[p, q, r, s] = resval[8] + resval[9]
        # alpha-alpha block real

    # unroll residual blocks into full matrix
    acse_residual = np.zeros(
        (2 * norbs, 2 * norbs, 2 * norbs, 2 * norbs), dtype=np.complex128
    )
    acse_residual[::2, ::2, ::2, ::2] = acse_aa
    acse_residual[1::2, 1::2, 1::2, 1::2] = acse_bb
    acse_residual[::2, 1::2, 1::2, ::2] = acse_ab
    acse_residual[::2, 1::2, ::2, 1::2] = np.einsum("ijkl->ijlk", -acse_ab)
    acse_residual[1::2, ::2, ::2, 1::2] = np.einsum("ijkl->jilk", acse_ab)
    acse_residual[1::2, ::2, 1::2, ::2] = np.einsum(
        "ijkl->ijlk", -acse_residual[1::2, ::2, ::2, 1::2]
    )

    return acse_residual


def get_tpdm_grad_fqe_atomic(p, q, r, s, fqe_appS, fqe_wf):
    """TODO: Add docstring."""
    # alpha-beta block real
    rdo = ((2 * p, 1), (2 * q + 1, 1), (2 * r + 1, 0), (2 * s, 0))
    rdo = of.FermionOperator(rdo) + of.hermitian_conjugated(
        of.FermionOperator(rdo)
    )
    fqe_wf_rdo = fqe_wf.apply(rdo)
    val1 = fqe.util.vdot(fqe_appS, fqe_wf_rdo)
    val2 = fqe.util.vdot(fqe_wf_rdo, fqe_appS)
    acse_ab_i = (val2 - val1) / 2j

    # alpha-beta block imag
    rdo = ((2 * p, 1), (2 * q + 1, 1), (2 * r + 1, 0), (2 * s, 0))
    rdo = 1j * (
        of.FermionOperator(rdo)
        - of.hermitian_conjugated(of.FermionOperator(rdo))
    )
    fqe_wf_rdo = fqe_wf.apply(rdo)
    val3 = fqe.util.vdot(fqe_appS, fqe_wf_rdo)
    val4 = fqe.util.vdot(fqe_wf_rdo, fqe_appS)
    acse_ab_r = (val4 - val3) / -2

    # alpha-alpha block real
    rdo = ((2 * p, 1), (2 * q, 1), (2 * r, 0), (2 * s, 0))
    rdo = of.FermionOperator(rdo) + of.hermitian_conjugated(
        of.FermionOperator(rdo)
    )
    fqe_wf_rdo = fqe_wf.apply(rdo)
    val1 = fqe.util.vdot(fqe_appS, fqe_wf_rdo)
    val2 = fqe.util.vdot(fqe_wf_rdo, fqe_appS)
    acse_aa_i = (val2 - val1) / 2j

    # alpha-alpha block imag
    rdo = ((2 * p, 1), (2 * q, 1), (2 * r, 0), (2 * s, 0))
    rdo = 1j * (
        of.FermionOperator(rdo)
        - of.hermitian_conjugated(of.FermionOperator(rdo))
    )
    fqe_wf_rdo = fqe_wf.apply(rdo)
    val3 = fqe.util.vdot(fqe_appS, fqe_wf_rdo)
    val4 = fqe.util.vdot(fqe_wf_rdo, fqe_appS)
    acse_aa_r = (val4 - val3) / -2

    # beta-beta block real
    rdo = ((2 * p + 1, 1), (2 * q + 1, 1), (2 * r + 1, 0), (2 * s + 1, 0))
    rdo = of.FermionOperator(rdo) + of.hermitian_conjugated(
        of.FermionOperator(rdo)
    )

    fqe_wf_rdo = fqe_wf.apply(rdo)
    val1 = fqe.util.vdot(fqe_appS, fqe_wf_rdo)
    val2 = fqe.util.vdot(fqe_wf_rdo, fqe_appS)
    acse_bb_i = (val2 - val1) / 2j

    # beta-beta block imag
    rdo = ((2 * p + 1, 1), (2 * q + 1, 1), (2 * r + 1, 0), (2 * s + 1, 0))
    rdo = 1j * (
        of.FermionOperator(rdo)
        - of.hermitian_conjugated(of.FermionOperator(rdo))
    )
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
    """Compute the acse gradient  <psi [rdo, A] psi>

    alpha-alpha, beta-beta, alpha-beta, and beta-alpha blocks

    we do not compression over alpha-alpha or beta-beta so these are still
    norbs**2 in linear dimension. In other words, we do computation on elements
    we know should be zero. This is for simplicity in the code.
    """
    # TODO: Check if parallel has been imported? e.g.
    #  if not PARALLELIZABLE:
    #      raise SomeError
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
                        op, coefficient=acse_res_tensor[p, q, r, s]
                    )
                    op = ((s, 1), (r, 1), (q, 0), (p, 0))
                    fop2 = of.FermionOperator(
                        op, coefficient=acse_res_tensor[s, r, q, p]
                    )
                    s_ops.append((fop1, fop2))
                    # s_op_total += fop2
                    # s_op_total += fop1
            else:
                four_tensor_counter[p, q, r, s] += 1
                if abs(acse_res_tensor[p, q, r, s]) > 1.0e-12:
                    op = ((p, 1), (q, 1), (r, 0), (s, 0))
                    fop1 = of.FermionOperator(
                        op, coefficient=acse_res_tensor[p, q, r, s]
                    )
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
            delayed(get_tpdm_grad_fqe_atomic)(p, q, r, s, fqe_appS, fqe_wf)
            for p, q, r, s in product(range(norbs), repeat=4)
        )

    for resval in result:
        p, q, r, s = resval[:4]
        acse_aa[p, q, r, s] = resval[4] + resval[5]
        acse_bb[p, q, r, s] = resval[6] + resval[7]
        acse_ab[p, q, r, s] = resval[8] + resval[9]
        # alpha-alpha block real

    # unroll residual blocks into full matrix
    acse_residual = np.zeros(
        (2 * norbs, 2 * norbs, 2 * norbs, 2 * norbs), dtype=np.complex128
    )
    acse_residual[::2, ::2, ::2, ::2] = acse_aa
    acse_residual[1::2, 1::2, 1::2, 1::2] = acse_bb
    acse_residual[::2, 1::2, 1::2, ::2] = acse_ab
    acse_residual[::2, 1::2, ::2, 1::2] = np.einsum("ijkl->ijlk", -acse_ab)
    acse_residual[1::2, ::2, ::2, 1::2] = np.einsum("ijkl->jilk", acse_ab)
    acse_residual[1::2, ::2, 1::2, ::2] = np.einsum(
        "ijkl->ijlk", -acse_residual[1::2, ::2, ::2, 1::2]
    )

    return acse_residual
