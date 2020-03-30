#   Copyright 2020 Google

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
#pylint: disable=line-too-long
#pylint: disable=missing-docstring
#pylint: disable=invalid-name
#pylint: disable=too-many-nested-blocks
import numpy

from openfermion import FermionOperator
from openfermion.utils import hermitian_conjugated


def number_nonconserving_fop(rank, norb):
    """Return a FermionOperator Hamiltonian
    """
    hamil = FermionOperator()

    if rank >= 0:
        hamil += FermionOperator('', 6.0)

    if rank >= 2:
        for i in range(0, 2*norb, 2):
            for j in range(0, 2*norb, 2):
                opstring = str(i) + ' ' + str(j + 1)
                hamil += FermionOperator(opstring, (i+1 + j*2)*0.1 - (i+1 + 2*(j + 1))*0.1j)
                opstring = str(i) + '^ ' + str(j + 1) + '^ '
                hamil += FermionOperator(opstring, (i+1 + j)*0.1 + (i+1 + j)*0.1j)
                opstring = str(i + 1) + ' ' + str(j)
                hamil += FermionOperator(opstring, (i+1 + j)*0.1 - (i+1 + j)*0.1j)
                opstring = str(i + 1) + '^ ' + str(j) + '^ '
                hamil += FermionOperator(opstring, (i+1 + j*2)*0.1 + (i+1 + 2*(j + 1))*0.1j)

    return (hamil + hermitian_conjugated(hamil))/2.0


def build_restricted(norb, full=False):
    """Build data structures for evolution tests to avoid large amount of data
    being saved in remote repository.
    """
    if full:
        orb_use = 2*norb
    else:
        orb_use = norb

    h1e = numpy.zeros((orb_use, orb_use), dtype=numpy.complex128)
    h2e = numpy.zeros((orb_use, orb_use, orb_use, orb_use), dtype=numpy.complex128)
    h3e = numpy.zeros((orb_use, orb_use, orb_use, orb_use, orb_use, orb_use), dtype=numpy.complex128)
    h4e = numpy.zeros((orb_use, orb_use, orb_use, orb_use, orb_use, orb_use, orb_use, orb_use), dtype=numpy.complex128)

    for i in range(norb):
        for j in range(norb):
            h1e[i, j] += (i+j) * 0.02
            for k in range(norb):
                for l in range(norb):
                    h2e[i, j, k, l] += (i+k)*(j+l)*0.02
                    for m in range(norb):
                        for n in range(norb):
                            h3e[i, j, k, l, m, n] += (i+l)*(j+m)*(k+n)*0.002
                            for o in range(norb):
                                for p in range(norb):
                                    h4e[i, j, k, l, m, n, o, p] += (i+m)*(j+n)*(k+o)*(l+p)*0.001
        h1e[i, i] += i * 2.0

    if full:
        h1e[norb:, norb:] = h1e[:norb, :norb]

        h2e[:norb, norb:, :norb, norb:] = h2e[:norb, :norb, :norb, :norb]
        h2e[norb:, :norb, norb:, :norb] = h2e[:norb, :norb, :norb, :norb]
        h2e[norb:, norb:, norb:, norb:] = h2e[:norb, :norb, :norb, :norb]

        h3e[:norb, norb:, :norb, :norb, norb:, :norb] = h3e[:norb, :norb, :norb, :norb, :norb, :norb]
        h3e[norb:, :norb, :norb, norb:, :norb, :norb] = h3e[:norb, :norb, :norb, :norb, :norb, :norb]
        h3e[:norb, :norb, norb:, :norb, :norb, norb:] = h3e[:norb, :norb, :norb, :norb, :norb, :norb]
        h3e[norb:, norb:, :norb, norb:, norb:, :norb] = h3e[:norb, :norb, :norb, :norb, :norb, :norb]
        h3e[:norb, norb:, norb:, :norb, norb:, norb:] = h3e[:norb, :norb, :norb, :norb, :norb, :norb]
        h3e[norb:, :norb, norb:, norb:, :norb, norb:] = h3e[:norb, :norb, :norb, :norb, :norb, :norb]
        h3e[norb:, norb:, norb:, norb:, norb:, norb:] = h3e[:norb, :norb, :norb, :norb, :norb, :norb]

        h4e[:norb, norb:, :norb, :norb, :norb, norb:, :norb, :norb] = h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]
        h4e[norb:, :norb, :norb, :norb, norb:, :norb, :norb, :norb] = h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]
        h4e[:norb, :norb, norb:, :norb, :norb, :norb, norb:, :norb] = h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]
        h4e[:norb, :norb, :norb, norb:, :norb, :norb, :norb, norb:] = h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]
        h4e[norb:, norb:, :norb, :norb, norb:, norb:, :norb, :norb] = h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]
        h4e[:norb, norb:, norb:, :norb, :norb, norb:, norb:, :norb] = h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]
        h4e[norb:, :norb, norb:, :norb, norb:, :norb, norb:, :norb] = h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]
        h4e[:norb, norb:, :norb, norb:, :norb, norb:, :norb, norb:] = h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]
        h4e[norb:, :norb, :norb, norb:, norb:, :norb, :norb, norb:] = h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]
        h4e[:norb, :norb, norb:, norb:, :norb, :norb, norb:, norb:] = h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]
        h4e[norb:, norb:, norb:, :norb, norb:, norb:, norb:, :norb] = h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]
        h4e[norb:, norb:, :norb, norb:, norb:, norb:, :norb, norb:] = h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]
        h4e[:norb, norb:, norb:, norb:, :norb, norb:, norb:, norb:] = h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]
        h4e[norb:, :norb, norb:, norb:, norb:, :norb, norb:, norb:] = h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]
        h4e[norb:, norb:, norb:, norb:, norb:, norb:, norb:, norb:] = h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]

    return h1e, h2e, h3e, h4e


def build_gso(norb):
    h1e = numpy.zeros((norb*2, norb*2), dtype=numpy.complex128)
    h2e = numpy.zeros((norb*2, norb*2, norb*2, norb*2), dtype=numpy.complex128)
    h3e = numpy.zeros((norb*2, norb*2, norb*2, norb*2, norb*2, norb*2), dtype=numpy.complex128)
    h4e = numpy.zeros((norb*2, norb*2, norb*2, norb*2, norb*2, norb*2, norb*2, norb*2), dtype=numpy.complex128)
    for i in range(norb*2):
        for j in range(norb*2):
            h1e[i, j] += (i+j) * 0.02
            for k in range(norb*2):
                for l in range(norb*2):
                    h2e[i, j, k, l] += (i+k)*(j+l)*0.02
                    for m in range(norb*2):
                        for n in range(norb*2):
                            h3e[i, j, k, l, m, n] += (i+l)*(j+m)*(k+n)*0.002
                            for o in range(norb*2):
                                for p in range(norb*2):
                                    h4e[i, j, k, l, m, n, o, p] += (i+m)*(j+n)*(k+o)*(l+p)*0.001
        h1e[i, i] += i * 2.0
    return h1e, h2e, h3e, h4e


def build_sso(norb):
    """Build data structures for evolution tests to avoid large amount of data
    being saved in remote repository.
    """
    h1e = numpy.zeros((norb*2, norb*2), dtype=numpy.complex128)
    h2e = numpy.zeros((norb*2, norb*2, norb*2, norb*2), dtype=numpy.complex128)
    h3e = numpy.zeros((norb*2, norb*2, norb*2, norb*2, norb*2, norb*2), dtype=numpy.complex128)
    h4e = numpy.zeros((norb*2, norb*2, norb*2, norb*2, norb*2, norb*2, norb*2, norb*2), dtype=numpy.complex128)
    for i in range(norb):
        for j in range(norb):
            h1e[i, j] += (i+j) * 0.02
            for k in range(norb):
                for l in range(norb):
                    h2e[i, j, k, l] += (i+k)*(j+l)*0.02
                    for m in range(norb):
                        for n in range(norb):
                            h3e[i, j, k, l, m, n] += (i+l)*(j+m)*(k+n)*0.002
                            for o in range(norb):
                                for p in range(norb):
                                    h4e[i, j, k, l, m, n, o, p] += (i+m)*(j+n)*(k+o)*(l+p)*0.001
        h1e[i, i] += i * 2.0

    h1e[norb:, norb:] = 2.0*h1e[:norb, :norb]

    h2e[:norb, norb:, :norb, norb:] = 2.0*h2e[:norb, :norb, :norb, :norb]
    h2e[norb:, :norb, norb:, :norb] = 2.0*h2e[:norb, :norb, :norb, :norb]

    h2e[norb:, norb:, norb:, norb:] = 4.0*h2e[:norb, :norb, :norb, :norb]


    h3e[:norb, norb:, :norb, :norb, norb:, :norb] = 2.0*h3e[:norb, :norb, :norb, :norb, :norb, :norb]
    h3e[norb:, :norb, :norb, norb:, :norb, :norb] = 2.0*h3e[:norb, :norb, :norb, :norb, :norb, :norb]
    h3e[:norb, :norb, norb:, :norb, :norb, norb:] = 2.0*h3e[:norb, :norb, :norb, :norb, :norb, :norb]

    h3e[norb:, norb:, :norb, norb:, norb:, :norb] = 4.0*h3e[:norb, :norb, :norb, :norb, :norb, :norb]
    h3e[:norb, norb:, norb:, :norb, norb:, norb:] = 4.0*h3e[:norb, :norb, :norb, :norb, :norb, :norb]
    h3e[norb:, :norb, norb:, norb:, :norb, norb:] = 4.0*h3e[:norb, :norb, :norb, :norb, :norb, :norb]

    h3e[norb:, norb:, norb:, norb:, norb:, norb:] = 6.0*h3e[:norb, :norb, :norb, :norb, :norb, :norb]


    h4e[:norb, norb:, :norb, :norb, :norb, norb:, :norb, :norb] = 2.0*h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]
    h4e[norb:, :norb, :norb, :norb, norb:, :norb, :norb, :norb] = 2.0*h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]
    h4e[:norb, :norb, norb:, :norb, :norb, :norb, norb:, :norb] = 2.0*h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]
    h4e[:norb, :norb, :norb, norb:, :norb, :norb, :norb, norb:] = 2.0*h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]

    h4e[norb:, norb:, :norb, :norb, norb:, norb:, :norb, :norb] = 4.0*h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]
    h4e[:norb, norb:, norb:, :norb, :norb, norb:, norb:, :norb] = 4.0*h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]
    h4e[norb:, :norb, norb:, :norb, norb:, :norb, norb:, :norb] = 4.0*h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]
    h4e[:norb, norb:, :norb, norb:, :norb, norb:, :norb, norb:] = 4.0*h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]
    h4e[norb:, :norb, :norb, norb:, norb:, :norb, :norb, norb:] = 4.0*h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]
    h4e[:norb, :norb, norb:, norb:, :norb, :norb, norb:, norb:] = 4.0*h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]

    h4e[norb:, norb:, norb:, :norb, norb:, norb:, norb:, :norb] = 6.0*h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]
    h4e[norb:, norb:, :norb, norb:, norb:, norb:, :norb, norb:] = 6.0*h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]
    h4e[:norb, norb:, norb:, norb:, :norb, norb:, norb:, norb:] = 6.0*h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]
    h4e[norb:, :norb, norb:, norb:, norb:, :norb, norb:, norb:] = 6.0*h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]

    h4e[norb:, norb:, norb:, norb:, norb:, norb:, norb:, norb:] = 8.0*h4e[:norb, :norb, :norb, :norb, :norb, :norb, :norb, :norb]

    return h1e, h2e, h3e, h4e
