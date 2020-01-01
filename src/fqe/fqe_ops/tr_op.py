#   Copyright 2019 Quantum Simulation Technologies Inc.
#
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

from _fqe_control import vdot
from fqe.fqe_ops import fqe_operator


class TimeReversalOp(fqe_operator.FqeOperator):


    def contract(self, brastate, ketstate):
        """
        """
        out = copy.deepcopy(ketstate)
        for (nele, nab), sector in out._civec.items():
            nalpha, nbeta = alpha_beta_electrons(nele, nab)
            if nalpha < nbeta:
                if not (nele, nbeta-nalpha) in out._civec.keys():
                    raise Exception('The wave function space is not closed under time reversal') 
                sector2 = out._civec[(nele, nbeta-nalpha)]
                tmp = numpy.copy(sector.coeff)
                phase = (-1)**(nbeta*(nalpha+1))
                phase2 = (-1)**(nalpha*(nbeta+1))
                sector.coeff = sector2.coeff.T.conj() * phase2
                sector2.coeff = tmp.T.conj() * phase
            elif nalpha == nbeta:
                sector.coeff = sector.coeff.T.conj()
        return vdot(brastate, out)


    def representation(self):
        """
        """
        return 's_z'


    def rank(self):
        """
        """
        return 2
