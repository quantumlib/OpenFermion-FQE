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
"""Generate reference data for fqe_data_set tests
"""
import os
import numpy
import sys
from fqe.fqe_data import FqeData
from fqe.fqe_data_set import FqeDataSet
sys.path.append(r'../')
from build_hamiltonian import build_H1, build_H2, build_H3, build_H4


def _write_fqe_data(nelec, norb, suffix, abspath, data):
    for nbeta in range(0, nelec + 1):
        nalpha = nelec - nbeta
        nstr = "{:02d}{:02d}{:02d}".format(nalpha, nbeta, norb)
        sector = data._data[(nalpha, nbeta)]
        crfile = os.path.join(abspath, "cr" + nstr + suffix + ".npy")
        cifile = os.path.join(abspath, "ci" + nstr + suffix + ".npy")

        sector.coeff.real.tofile(crfile)
        sector.coeff.imag.tofile(cifile)


def generate_data(nelec,
                  norb,
                  diag=True,
                  maxn=2,
                  maxrdm=0,
                  daga=[],
                  undaga=[],
                  dagb=[],
                  undagb=[]):
    dstr = "{:02d}{:02d}".format(nelec, norb)
    abspath = os.path.abspath(__file__)
    abspath = abspath.replace("generate_data.py", "set" + dstr)
    if os.path.isdir(abspath):
        print("Directory already exists")
    else:
        os.mkdir(abspath)

    # use a fixed random seed
    seed = 134570 + norb * 100 + nelec
    rng = numpy.random.default_rng(seed)

    _data = dict()
    for nbeta in range(0, nelec + 1):
        nalpha = nelec - nbeta
        nstr = "{:02d}{:02d}{:02d}".format(nalpha, nbeta, norb)
        sector = FqeData(nalpha, nbeta, norb)
        crfile = os.path.join(abspath, "cr" + nstr + ".npy")
        cifile = os.path.join(abspath, "ci" + nstr + ".npy")

        cr = rng.uniform(-0.5, 0.5, size=sector.coeff.shape)
        ci = rng.uniform(-0.5, 0.5, size=sector.coeff.shape)

        cr.tofile(crfile)
        ci.tofile(cifile)
        sector.coeff = cr + 1.j * ci
        _data[(nalpha, nbeta)] = sector
    test = FqeDataSet(nelec, norb, _data)

    # apply arrays
    if maxn > 0:  # 1-particle
        h1 = build_H1(norb, full=True)

        out = test.apply((h1,))
        _write_fqe_data(nelec, norb, "_1", abspath, out)
    if maxn > 1:  # 2-particle
        h2 = build_H2(norb, full=True)
        h1z = numpy.zeros(h1.shape)

        out = test.apply((h1z, h2))
        _write_fqe_data(nelec, norb, "_2", abspath, out)

        out = test.apply((h1, h2))
        _write_fqe_data(nelec, norb, "_12", abspath, out)
    if maxn > 2:  # 3-particle
        h3 = build_H3(norb, full=True)
        h2z = numpy.zeros(h2.shape)

        out = test.apply((h1z, h2z, h3))
        _write_fqe_data(nelec, norb, "_3", abspath, out)

        out = test.apply((h1, h2, h3))
        _write_fqe_data(nelec, norb, "_123", abspath, out)
    if maxn > 3:  # 4-particle
        h4 = build_H4(norb, full=True)
        h3z = numpy.zeros(h3.shape)

        out = test.apply((h1z, h2z, h3z, h4))
        _write_fqe_data(nelec, norb, "_4", abspath, out)

        out = test.apply((h1, h2, h3, h4))
        _write_fqe_data(nelec, norb, "_1234", abspath, out)

    # RDMs
    if maxrdm == 1:
        d1 = test.rdm1()
        d1.real.tofile(os.path.join(abspath, "dr1.npy"))
        d1.imag.tofile(os.path.join(abspath, "di1.npy"))
    elif maxrdm == 2:
        d1, d2 = test.rdm12()
        d1.real.tofile(os.path.join(abspath, "dr1.npy"))
        d1.imag.tofile(os.path.join(abspath, "di1.npy"))
        d2.real.tofile(os.path.join(abspath, "dr2.npy"))
        d2.imag.tofile(os.path.join(abspath, "di2.npy"))
    elif maxrdm == 3:
        d1, d2, d3 = test.rdm123()
        d1.real.tofile(os.path.join(abspath, "dr1.npy"))
        d1.imag.tofile(os.path.join(abspath, "di1.npy"))
        d2.real.tofile(os.path.join(abspath, "dr2.npy"))
        d2.imag.tofile(os.path.join(abspath, "di2.npy"))
        d3.real.tofile(os.path.join(abspath, "dr3.npy"))
        d3.imag.tofile(os.path.join(abspath, "di3.npy"))
    elif maxrdm == 4:
        d1, d2, d3, d4 = test.rdm1234()
        d1.real.tofile(os.path.join(abspath, "dr1.npy"))
        d1.imag.tofile(os.path.join(abspath, "di1.npy"))
        d2.real.tofile(os.path.join(abspath, "dr2.npy"))
        d2.imag.tofile(os.path.join(abspath, "di2.npy"))
        d3.real.tofile(os.path.join(abspath, "dr3.npy"))
        d3.imag.tofile(os.path.join(abspath, "di3.npy"))
        d4.real.tofile(os.path.join(abspath, "dr4.npy"))
        d4.imag.tofile(os.path.join(abspath, "di4.npy"))
    if maxrdm > 4:
        raise Exception("Higher that 4-particle RDMs not available")

    if daga or undaga or dagb or undagb:
        # apply individual N-body operator
        out = test.apply_individual_nbody(complex(1), daga, undaga, dagb,
                                          undagb)
        sdaga = "".join([str(da) for da in daga])
        sundaga = "".join([str(uda) for uda in undaga])
        sdagb = "".join([str(db) for db in dagb])
        sundagb = "".join([str(udb) for udb in undagb])
        suffix = "indv" + sdaga + "_" + sundaga + "_" + sdagb + "_" + sundagb
        _write_fqe_data(nelec, norb, suffix, abspath, out)

        # evolve with individual N-body operator
        test.evolve_inplace_individual_nbody(0.1, complex(1), daga, undaga,
                                             dagb, undagb)
        suffix = "ievo" + sdaga + "_" + sundaga + "_" + sdagb + "_" + sundagb
        _write_fqe_data(nelec, norb, suffix, abspath, test)


def regenerate_reference_data():
    """ Regenerates the reference data
    """
    generate_data(2,
                  2,
                  maxn=4,
                  maxrdm=4,
                  daga=[0],
                  undaga=[1],
                  dagb=[0],
                  undagb=[0])


if __name__ == "__main__":
    regenerate_reference_data()
