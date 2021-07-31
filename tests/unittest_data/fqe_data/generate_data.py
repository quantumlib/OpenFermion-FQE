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
"""Generate reference data for fqe_data tests
"""
import os
import numpy
from fqe.fqe_data import FqeData
from fqe import get_diagonalcoulomb_hamiltonian

import sys
sys.path.append(r'../')
from build_hamiltonian import build_H1, build_H2, build_H3, build_H4


def generate_data(nalpha, nbeta, norb, diag=True, maxn=2, maxrdm=0,
                  daga=[], undaga=[], dagb=[], undagb=[]):
    # make random coeffs
    nstr = "{:02d}{:02d}{:02d}".format(nalpha, nbeta, norb)
    test = FqeData(nalpha, nbeta, norb)

    crfile = "cr" + nstr + ".npy"
    cifile = "ci" + nstr + ".npy"

    # use a fixed random seed (different for different test cases)
    seed = 432782 + norb*1000 + nalpha*100 + nbeta
    rng = numpy.random.default_rng(seed)

    cr = rng.uniform(-0.5, 0.5, size=test.coeff.shape)
    ci = rng.uniform(-0.5, 0.5, size=test.coeff.shape)

    cr.tofile(crfile)
    ci.tofile(cifile)

    test.coeff = cr + 1.j*ci

    if diag:
        # make diagonal coulomb (repulsive) matrix
        dmat = 8*rng.uniform(0, 1, size=(norb, norb))
        dmat.tofile("dmat" + nstr + ".npy")

        # make diagonal coulomb reference
        hamil = get_diagonalcoulomb_hamiltonian(dmat)

        diag, vij = hamil.iht(0.1)
        data = test.diagonal_coulomb(diag, vij)

        data.real.tofile("cr" + nstr + "_dc.npy")
        data.imag.tofile("ci" + nstr + "_dc.npy")

    # make dense Hamiltonian arrays
    if maxn > 0:  # 1-particle
        h1 = build_H1(norb)
        h1.tofile("h1" + nstr + ".npy")

        out = test.apply((h1,))
        out.coeff.real.tofile("cr" + nstr + "_1.npy")
        out.coeff.imag.tofile("ci" + nstr + "_1.npy")
    if maxn > 1:  # 2-particle
        h2 = build_H2(norb)
        h2.tofile("h2" + nstr + ".npy")
        h1z = numpy.zeros(h1.shape)

        out = test.apply((h1z, h2))
        out.coeff.real.tofile("cr" + nstr + "_2.npy")
        out.coeff.imag.tofile("ci" + nstr + "_2.npy")

        out = test.apply((h1, h2))
        out.coeff.real.tofile("cr" + nstr + "_12.npy")
        out.coeff.imag.tofile("ci" + nstr + "_12.npy")
    if maxn > 2:  # 3-particle
        h3 = build_H3(norb)
        h3.tofile("h3" + nstr + ".npy")
        h2z = numpy.zeros(h2.shape)

        out = test.apply((h1z, h2z, h3))
        out.coeff.real.tofile("cr" + nstr + "_3.npy")
        out.coeff.imag.tofile("ci" + nstr + "_3.npy")

        out = test.apply((h1, h2, h3))
        out.coeff.real.tofile("cr" + nstr + "_123.npy")
        out.coeff.imag.tofile("ci" + nstr + "_123.npy")
    if maxn > 3:  # 4-particle
        h4 = build_H4(norb)
        h4.tofile("h4" + nstr + ".npy")
        h3z = numpy.zeros(h3.shape)

        out = test.apply((h1z, h2z, h3z, h4))
        out.coeff.real.tofile("cr" + nstr + "_4.npy")
        out.coeff.imag.tofile("ci" + nstr + "_4.npy")

        out = test.apply((h1, h2, h3, h4))
        out.coeff.real.tofile("cr" + nstr + "_1234.npy")
        out.coeff.imag.tofile("ci" + nstr + "_1234.npy")

    if maxrdm == 1:
        d1 = test.rdm1()
        d1.real.tofile("dr1" + nstr + ".npy")
        d1.imag.tofile("di1" + nstr + ".npy")
    elif maxrdm == 2:
        d1, d2 = test.rdm12()
        d1.real.tofile("dr1" + nstr + ".npy")
        d1.imag.tofile("di1" + nstr + ".npy")
        d2.real.tofile("dr2" + nstr + ".npy")
        d2.imag.tofile("di2" + nstr + ".npy")
    elif maxrdm == 3:
        d1, d2, d3 = test.rdm123()
        d1.real.tofile("dr1" + nstr + ".npy")
        d1.imag.tofile("di1" + nstr + ".npy")
        d2.real.tofile("dr2" + nstr + ".npy")
        d2.imag.tofile("di2" + nstr + ".npy")
        d3.real.tofile("dr3" + nstr + ".npy")
        d3.imag.tofile("di3" + nstr + ".npy")
    elif maxrdm == 4:
        d1, d2, d3, d4 = test.rdm1234()
        d1.real.tofile("dr1" + nstr + ".npy")
        d1.imag.tofile("di1" + nstr + ".npy")
        d2.real.tofile("dr2" + nstr + ".npy")
        d2.imag.tofile("di2" + nstr + ".npy")
        d3.real.tofile("dr3" + nstr + ".npy")
        d3.imag.tofile("di3" + nstr + ".npy")
        d4.real.tofile("dr4" + nstr + ".npy")
        d4.imag.tofile("di4" + nstr + ".npy")
    if maxrdm > 4:
        raise Exception("Higher that 4-particle RDMs not available")


    if daga or undaga or dagb or undagb:
        out = test.apply_individual_nbody(complex(1), daga, undaga, dagb, undagb)
        sdaga = ''.join([str(da) for da in daga])
        sundaga = ''.join([str(uda) for uda in undaga])
        sdagb = ''.join([str(db) for db in dagb])
        sundagb = ''.join([str(udb) for udb in undagb])
        out.coeff.real.tofile("cr" + nstr + "indv" + sdaga + "_" + sundaga
                              + "_" + sdagb + "_" + sundagb + ".npy")
        out.coeff.imag.tofile("ci" + nstr + "indv" + sdaga + "_" + sundaga
                              + "_" + sdagb + "_" + sundagb + ".npy")

def regenerate_reference_data():
    """ Regenerates all reference data for fqe_data_test.
        Data obtained with it should NOT be merged into the main branch
        unless they are generated from a stable branch!
    """
    generate_data(2, 3, 6, daga=[1], undaga=[2], dagb=[], undagb=[])
    generate_data(2, 1, 4, maxn=4, maxrdm=4)
    generate_data(1, 1, 2, diag=False, maxn=4, maxrdm=0)  # test for spin orbs

if __name__ == "__main__":
    regenerate_reference_data()
