import os
import copy

import numpy
from fqe.fqe_data import FqeData
from fqe import get_diagonalcoulomb_hamiltonian
import tests.unittest_data as fud


class FqeDataLoader:
    """ FqeDataLoader provides an interface to load reference data for
    unit testing.
    """
    def __init__(self, nalpha, nbeta, norb):
        """
        Args:
            nalpha (int) - Number of alpha electrons
            nbeta (int) - Number of beta electrons
            norb (int) - Number of spatial orbitals
        """
        self.nalpha = nalpha
        self.nbeta = nbeta
        self.norb = norb
        self.nstr = "{:02d}{:02d}{:02d}".format(nalpha, nbeta, norb)
        self.file_path = fud.__file__.replace("__init__.py", "fqe_data")

        self.data = FqeData(self.nalpha, self.nbeta, self.norb)
        fr = os.path.join(self.file_path, "cr" + self.nstr + ".npy")
        fi = os.path.join(self.file_path, "ci" + self.nstr + ".npy")
        cr = numpy.fromfile(fr).reshape(self.data.coeff.shape)
        ci = numpy.fromfile(fi).reshape(self.data.coeff.shape)
        self.data.coeff = cr + 1.j*ci

    def get_fqe_data(self):
        """Return a copy of the FqeData object"""
        return copy.deepcopy(self.data)

    def get_diagonal_coulomb(self):
        """Return the diagonal coulomb operator"""
        filename = os.path.join(self.file_path, "dmat" + self.nstr + ".npy")
        dmat = numpy.fromfile(filename).reshape(self.norb, self.norb)
        hamil = get_diagonalcoulomb_hamiltonian(dmat)
        return hamil

    def get_diagonal_coulomb_ref(self):
        """Return the reference coefficients for diagonal_coulomb method"""
        fr = os.path.join(self.file_path, "cr" + self.nstr + "_dc.npy")
        fi = os.path.join(self.file_path, "ci" + self.nstr + "_dc.npy")
        cr = numpy.fromfile(fr).reshape(self.data.coeff.shape)
        ci = numpy.fromfile(fi).reshape(self.data.coeff.shape)
        return cr + 1.j*ci

    def get_harray(self, order):
        filename = os.path.join(self.file_path,
                                "h" + str(order) + self.nstr + ".npy")
        shape = tuple([self.norb] * order * 2)
        return numpy.fromfile(filename).reshape(shape)

    def get_href(self, hstr):
        fr = os.path.join(self.file_path,
                          "cr" + self.nstr + "_" + hstr + ".npy")
        fi = os.path.join(self.file_path,
                          "ci" + self.nstr + "_" + hstr + ".npy")
        cr = numpy.fromfile(fr).reshape(self.data.coeff.shape)
        ci = numpy.fromfile(fi).reshape(self.data.coeff.shape)
        return cr + 1.j*ci

    def get_indv_ref(self, daga, undaga, dagb, undagb):
        nstr = self.nstr
        sdaga = ''.join([str(da) for da in daga])
        sundaga = ''.join([str(uda) for uda in undaga])
        sdagb = ''.join([str(db) for db in dagb])
        sundagb = ''.join([str(udb) for udb in undagb])
        fr = os.path.join(self.file_path,
            "cr" + nstr + "indv" + sdaga + "_" + sundaga
            + "_" + sdagb + "_" + sundagb + ".npy")
        fi = os.path.join(self.file_path,
            "ci" + nstr + "indv" + sdaga + "_" + sundaga
            + "_" + sdagb + "_" + sundagb + ".npy")
        cr = numpy.fromfile(fr).reshape(self.data.coeff.shape)
        ci = numpy.fromfile(fi).reshape(self.data.coeff.shape)
        return cr + 1.j*ci

    def get_rdm(self, order):
        fr = os.path.join(self.file_path,
                          "dr" + str(order) + self.nstr + ".npy")
        fi = os.path.join(self.file_path,
                          "di" + str(order) + self.nstr + ".npy")
        shape = tuple([self.norb] * order * 2)
        dr = numpy.fromfile(fr).reshape(shape)
        di = numpy.fromfile(fi).reshape(shape)
        return dr + 1.j*di
