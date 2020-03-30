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

"""Implements the Sz operator
"""

import copy

from typing import TYPE_CHECKING

from fqe.util import vdot
from fqe.fqe_ops import fqe_operator

if TYPE_CHECKING:
    from fqe.wavefunction import Wavefunction


class SzOperator(fqe_operator.FqeOperator):
    """:math:`S_z` operator as a specialization of FqeOperator
    """

    def contract(self,
                 brastate: 'Wavefunction',
                 ketstate: 'Wavefunction') -> complex:
        """Given two wavefunctions, generate the expectation value of the
        operator according to its representation.

        Args:
            brastate (Wavefunction) - wave function on the bra side

            ketstate (Wavefunction) - wave function on the ket side
        """
        out = copy.deepcopy(ketstate)
        for _, sector in out._civec.items():
            sector.scale((sector.nalpha() - sector.nbeta()) * 0.5)
        return vdot(brastate, out)


    def representation(self):
        """Representation of the Sz operator
        """
        return 's_z'


    def rank(self):
        """Rank of the Sz operator, which is 2
        """
        return 2
