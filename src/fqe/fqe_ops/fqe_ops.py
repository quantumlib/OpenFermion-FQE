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
"""Defines the number operator, S^2 operator, Sz operator, and time-reveral
operator."""

import copy
from typing import TYPE_CHECKING

import numpy as np

from fqe.util import alpha_beta_electrons, vdot
from fqe.fqe_ops import fqe_operator

if TYPE_CHECKING:
    from fqe.wavefunction import Wavefunction


class NumberOperator(fqe_operator.FqeOperator):
    """The number operator."""

    def contract(self, brastate: "Wavefunction",
                 ketstate: "Wavefunction") -> complex:
        """Given two wavefunctions, generate the expectation value of the
        operator according to its representation.

        Args:
            brastate: Wavefunction on the bra side.
            ketstate: Wavefunction on the ket side.
        """
        out = copy.deepcopy(ketstate)
        for _, sector in out._civec.items():
            sector.scale(sector.nalpha() + sector.nbeta())
        return vdot(brastate, out)

    def representation(self):
        """Returns the representation of the number operator, which is 'N'."""
        return "N"

    def rank(self):
        """Returns the rank of the number operator."""
        return 2


class S2Operator(fqe_operator.FqeOperator):
    r"""The :math:`S^2` operator."""

    def contract(self, brastate: "Wavefunction",
                 ketstate: "Wavefunction") -> complex:
        """Given two wavefunctions, generate the expectation value of the
        operator according to its representation.

        Args:
            brastate: Wavefunction on the bra side.
            ketstate: Wavefunction on the ket side.
        """
        out = copy.deepcopy(ketstate)
        for _, sector in out._civec.items():
            sector.apply_inplace_s2()
        return vdot(brastate, out)

    def representation(self):
        """Returns the representation of the operator."""
        return "s_2"

    def rank(self):
        """Returns rank of the operator."""
        return 2


class SzOperator(fqe_operator.FqeOperator):
    r"""The :math:`S_z` operator."""

    def contract(self, brastate: "Wavefunction",
                 ketstate: "Wavefunction") -> complex:
        """Given two wavefunctions, generate the expectation value of the
        operator according to its representation.

        Args:
            brastate: Wavefunction on the bra side.
            ketstate: Wavefunction on the ket side.
        """
        out = copy.deepcopy(ketstate)
        for _, sector in out._civec.items():
            sector.scale((sector.nalpha() - sector.nbeta()) * 0.5)
        return vdot(brastate, out)

    def representation(self):
        """Returns the representation of the Sz operator."""
        return "s_z"

    def rank(self):
        """Returns the rank of the Sz operator."""
        return 2


class TimeReversalOp(fqe_operator.FqeOperator):
    """The time-reversal operator.

    The program assumes the Kramers-paired storage for the wavefunction.
    """

    def contract(self, brastate: "Wavefunction",
                 ketstate: "Wavefunction") -> complex:
        """Given two wavefunctions, generate the expectation value of the
        operator according to its representation.

        Args:
            brastate: Wavefunction on the bra side.
            ketstate: Wavefunction on the ket side.
        """
        out = copy.deepcopy(ketstate)
        for (nele, nab), sector in out._civec.items():
            nalpha, nbeta = alpha_beta_electrons(nele, nab)
            if nalpha < nbeta:
                if not (nele, nbeta - nalpha) in out._civec.keys():
                    raise ValueError(
                        "The wavefunction space is not closed under "
                        "time reversal.")
                sector2 = out._civec[(nele, nbeta - nalpha)]
                tmp = np.copy(sector.coeff)
                phase = (-1)**(nbeta * (nalpha + 1))
                phase2 = (-1)**(nalpha * (nbeta + 1))
                sector.coeff = sector2.coeff.T.conj() * phase2
                sector2.coeff = tmp.T.conj() * phase
            elif nalpha > nbeta:
                if not (nele, nbeta - nalpha) in out._civec.keys():
                    raise ValueError(
                        "The wavefunction space is not closed under "
                        "time reversal.")
            elif nalpha == nbeta:
                sector.coeff = sector.coeff.T.conj()
        return vdot(brastate, out)

    def representation(self):
        """Returns the representation of the operator."""
        return "T"

    def rank(self):
        """Returns the rank of the operator."""
        return 2
