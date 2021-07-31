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
"""Build Hamiltonian is a convenience routine for initializing unittest data.
"""
# pylint: disable=line-too-long
# pylint: disable=missing-docstring
# pylint: disable=invalid-name
# pylint: disable=too-many-nested-blocks
from typing import Tuple

import numpy as np

from openfermion import FermionOperator
from openfermion.utils import hermitian_conjugated


def number_nonconserving_fop(rank: int, norb: int) -> FermionOperator:
    """Returns a FermionOperator Hamiltonian that breaks the number symmetry

    Args:
        rank (int): rank of the Hamiltonian

        norb (int): number of orbitals

    Returns:
        (FermionOperator): resulting FermionOperator object
    """
    hamil = FermionOperator()

    if rank >= 0:
        hamil += FermionOperator("", 6.0)

    if rank >= 2:
        for i in range(0, 2 * norb, 2):
            for j in range(0, 2 * norb, 2):
                opstring = str(i) + " " + str(j + 1)
                hamil += FermionOperator(
                    opstring,
                    (i + 1 + j * 2) * 0.1 - (i + 1 + 2 * (j + 1)) * 0.1j,
                )
                opstring = str(i) + "^ " + str(j + 1) + "^ "
                hamil += FermionOperator(opstring,
                                         (i + 1 + j) * 0.1 + (i + 1 + j) * 0.1j)
                opstring = str(i + 1) + " " + str(j)
                hamil += FermionOperator(opstring,
                                         (i + 1 + j) * 0.1 - (i + 1 + j) * 0.1j)
                opstring = str(i + 1) + "^ " + str(j) + "^ "
                hamil += FermionOperator(
                    opstring,
                    (i + 1 + j * 2) * 0.1 + (i + 1 + 2 * (j + 1)) * 0.1j,
                )

    return (hamil + hermitian_conjugated(hamil)) / 2.0


def to_spin1(h1r: np.ndarray, asymmetric: bool = False) -> np.ndarray:
    """
    Converts the 1-body Hamiltonian term from spatial to spin-orbital basis

    Args:
        h1r (np.ndarray): Hamiltonian in spatial orbital basis

        asymmetric (bool): introduce asymmetry between alpha and beta terms, \
            has no effect if full=False

    Returns:
        (np.ndarray): resulting array
    """
    norb = h1r.shape[0]
    assert(h1r.shape[1] == norb)
    n = 2 * norb
    h1e = np.zeros((n, n))

    h1e[:norb, :norb] = h1r
    prefactor = 2.0 if asymmetric else 1.0
    h1e[norb:, norb:] = prefactor * h1r
    return h1e


def to_spin2(h2r: np.ndarray, asymmetric: bool = False) -> np.ndarray:
    """
    Converts the 2-body Hamiltonian term from spatial to spin-orbital basis

    Args:
        h2r (np.ndarray): Hamiltonian in spatial orbital basis

        asymmetric (bool): introduce asymmetry between alpha and beta terms, \
            has no effect if full=False

    Returns:
        (np.ndarray): resulting array
    """
    norb = h2r.shape[0]
    assert(h2r.shape[1] == norb)
    assert(h2r.shape[2] == norb)
    assert(h2r.shape[3] == norb)
    n = 2 * norb
    h2e = np.zeros((n, n, n, n))
    h2e[:norb, :norb, :norb, :norb] = h2r

    prefactor = 2.0 if asymmetric else 1.0
    h2e[:norb, norb:, :norb, norb:] = prefactor * h2r
    h2e[norb:, :norb, norb:, :norb] = prefactor * h2r

    prefactor = 4.0 if asymmetric else 1.0
    h2e[norb:, norb:, norb:, norb:] = prefactor * h2r

    return h2e

def to_spin3(h3r: np.ndarray, asymmetric: bool = False) -> np.ndarray:
    """
    Converts the 3-body Hamiltonian term from spatial to spin-orbital basis

    Args:
        h3r (np.ndarray): Hamiltonian in spatial orbital basis

        asymmetric (bool): introduce asymmetry between alpha and beta terms, \
            has no effect if full=False

    Returns:
        (np.ndarray): resulting array
    """
    norb = h3r.shape[0]
    assert(h3r.shape[1] == norb)
    assert(h3r.shape[2] == norb)
    assert(h3r.shape[3] == norb)
    assert(h3r.shape[4] == norb)
    assert(h3r.shape[5] == norb)
    n = 2 * norb
    h3e = np.zeros((n, n, n, n, n, n))

    h3e[:norb, :norb, :norb, :norb, :norb, :norb] = h3r

    prefactor = 2.0 if asymmetric else 1.0
    h3e[:norb, norb:, :norb, :norb, norb:, :norb] = prefactor * h3r
    h3e[norb:, :norb, :norb, norb:, :norb, :norb] = prefactor * h3r
    h3e[:norb, :norb, norb:, :norb, :norb, norb:] = prefactor * h3r

    prefactor = 4.0 if asymmetric else 1.0
    h3e[norb:, norb:, :norb, norb:, norb:, :norb] = prefactor * h3r
    h3e[:norb, norb:, norb:, :norb, norb:, norb:] = prefactor * h3r
    h3e[norb:, :norb, norb:, norb:, :norb, norb:] = prefactor * h3r

    prefactor = 6.0 if asymmetric else 1.0
    h3e[norb:, norb:, norb:, norb:, norb:, norb:] = prefactor * h3r

    return h3e


def to_spin4(h4r: np.ndarray, asymmetric: bool = False) -> np.ndarray:
    """
    Converts the 4-body Hamiltonian term from spatial to spin-orbital basis

    Args:
        h4r (np.ndarray): Hamiltonian in spatial orbital basis

        asymmetric (bool): introduce asymmetry between alpha and beta terms, \
            has no effect if full=False

    Returns:
        (np.ndarray): resulting array
    """

    norb = h4r.shape[0]
    assert(h4r.shape[1] == norb)
    assert(h4r.shape[2] == norb)
    assert(h4r.shape[3] == norb)
    assert(h4r.shape[4] == norb)
    assert(h4r.shape[5] == norb)
    assert(h4r.shape[6] == norb)
    assert(h4r.shape[7] == norb)
    n = 2 * norb
    h4e = np.zeros((n, n, n, n, n, n, n, n))

    h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb] = h4r

    prefactor = 2.0 if asymmetric else 1.0
    h4e[:norb, norb:, :norb, :norb, :norb, norb:, :norb, :norb] = prefactor*h4r
    h4e[norb:, :norb, :norb, :norb, norb:, :norb, :norb, :norb] = prefactor*h4r
    h4e[:norb, :norb, norb:, :norb, :norb, :norb, norb:, :norb] = prefactor*h4r
    h4e[:norb, :norb, :norb, norb:, :norb, :norb, :norb, norb:] = prefactor*h4r

    prefactor = 4.0 if asymmetric else 1.0
    h4e[norb:, norb:, :norb, :norb, norb:, norb:, :norb, :norb] = prefactor*h4r
    h4e[:norb, norb:, norb:, :norb, :norb, norb:, norb:, :norb] = prefactor*h4r
    h4e[norb:, :norb, norb:, :norb, norb:, :norb, norb:, :norb] = prefactor*h4r
    h4e[:norb, norb:, :norb, norb:, :norb, norb:, :norb, norb:] = prefactor*h4r
    h4e[norb:, :norb, :norb, norb:, norb:, :norb, :norb, norb:] = prefactor*h4r
    h4e[:norb, :norb, norb:, norb:, :norb, :norb, norb:, norb:] = prefactor*h4r

    prefactor = 6.0 if asymmetric else 1.0
    h4e[norb:, norb:, norb:, :norb, norb:, norb:, norb:, :norb] = prefactor*h4r
    h4e[norb:, norb:, :norb, norb:, norb:, norb:, :norb, norb:] = prefactor*h4r
    h4e[:norb, norb:, norb:, norb:, :norb, norb:, norb:, norb:] = prefactor*h4r
    h4e[norb:, :norb, norb:, norb:, norb:, :norb, norb:, norb:] = prefactor*h4r

    prefactor = 8.0 if asymmetric else 1.0
    h4e[norb:, norb:, norb:, norb:, norb:, norb:, norb:, norb:] = prefactor*h4r

    return h4e


def build_H1(norb: int, full: bool = False, asymmetric: bool = False) \
    -> np.ndarray:
    """Build Hamiltonian array for 1-body interactions.

    Args:
        norb (int): number of orbitals

        full (bool): represent in the full spin-orbital basis

        asymmetric (bool): introduce asymmetry between alpha and beta terms, \
            has no effect if full=False

    Returns:
        (np.ndarray): resulting array
    """

    h1e = np.zeros((norb,) * 2)

    for i in range(norb):
        h1e[i, i] += i * 2.0
        for j in range(norb):
            h1e[i, j] += (i + j) * 0.02

    if full:
        return to_spin1(h1e, asymmetric)
    else:
        return h1e


def build_H2(norb: int, full: bool = False, asymmetric: bool = False) \
    -> np.ndarray:
    """Build Hamiltonian array for 2-body interactions.

    Args:
        norb (int): number of orbitals

        full (bool): represent in the full spin-orbital basis

        asymmetric (bool): introduce asymmetry between alpha and beta terms, \
            has no effect if full=False

    Returns:
        (np.ndarray): resulting array
    """

    h2e = np.zeros((norb,) * 4)

    for i in range(norb):
        for j in range(norb):
            for k in range(norb):
                for l in range(norb):
                    h2e[i, j, k, l] += (i + k) * (j + l) * 0.02

    if full:
        return to_spin2(h2e, asymmetric)
    else:
        return h2e

def build_H3(norb: int, full: bool = False, asymmetric: bool = False) \
    -> np.ndarray:
    """Build Hamiltonian array for 3-body interactions.

    Args:
        norb (int): number of orbitals

        full (bool): represent in the full spin-orbital basis

        asymmetric (bool): introduce asymmetry between alpha and beta terms, \
            has no effect if full=False

    Returns:
        (np.ndarray): resulting array
    """

    h3e = np.zeros((norb,) * 6)

    for i in range(norb):
        for j in range(norb):
            for k in range(norb):
                for l in range(norb):
                    for m in range(norb):
                        for n in range(norb):
                            h3e[i, j, k, l, m, n] += ((i + l) * (j + m) *
                                                      (k + n) * 0.002)

    if full:
        return to_spin3(h3e, asymmetric)
    else:
        return h3e


def build_H4(norb: int, full: bool = False, asymmetric: bool = False) \
    -> np.ndarray:
    """Build Hamiltonian array for 4-body interactions.


    Args:
        norb (int): number of orbitals

        full (bool): represent in the full spin-orbital basis

        asymmetric (bool): introduce asymmetry between alpha and beta terms, \
            has no effect if full=False

    Returns:
        (np.ndarray): resulting array
    """
    h4e = np.zeros((norb,) * 8)

    for i in range(norb):
        for j in range(norb):
            for k in range(norb):
                for l in range(norb):
                    for m in range(norb):
                        for n in range(norb):
                            for o in range(norb):
                                for p in range(norb):
                                    h4e[i, j, k, l, m, n, o, p] += ((i + m) *
                                                                    (j + n) *
                                                                    (k + o) *
                                                                    (l + p) *
                                                                    0.001)

    if full:
        return to_spin4(h4e, asymmetric)
    else:
        return h4e


def build_restricted(norb: int, full: bool = False, asymmetric: bool = False) \
    -> Tuple[np.ndarray, ...]:
    """ Build a test Hamiltonian for 1,2,3- and 4-body interactions with \
    same alpha and beta interactions

    Args:
        norb (int): number of spatial orbitals

        full (bool): whether Hamiltonian is generated in the full spin-orbital \
            basis (True) or spatial orbital basis (False)

        asymmetric (bool): introduce asymmetry between alpha and beta terms

    Returns:
        Tuple[np.ndarray, ...]: resulting set of arrays
    """
    h1e = build_H1(norb, full, asymmetric)
    h2e = build_H2(norb, full, asymmetric)
    h3e = build_H3(norb, full, asymmetric)
    h4e = build_H4(norb, full, asymmetric)

    return h1e, h2e, h3e, h4e


def build_gso(norb: int) -> Tuple[np.ndarray, ...]:
    """ Build a test Hamiltonian that mimics a GSO Hamiltonian structure.

    Args:
        norb (int): number of spatial orbitals

    Returns:
        Tuple[np.ndarray, ...]: resulting set of arrays
    """

    return build_restricted(2 * norb, full=False)

def build_sso(norb: int) -> Tuple[np.ndarray, ...]:
    """ Build a test Hamiltonian that mimics a SSO Hamiltonian structure.

    Args:
        norb (int): number of (spatial or spin-) orbitals

    Returns:
        Tuple[np.ndarray, ...]: resulting set of arrays
    """

    return build_restricted(norb, full=True, asymmetric=True)
