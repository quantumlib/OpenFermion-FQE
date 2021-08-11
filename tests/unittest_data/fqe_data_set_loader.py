import os
import copy

import numpy
from fqe.fqe_data import FqeData
from fqe.fqe_data_set import FqeDataSet
import tests.unittest_data as fud


class FqeDataSetLoader:
    """ FqeDataSetLoader provides an interface to load reference data for
    unit testing.
    """

    def __init__(self, nelec, norb):
        """
        Args:
            nelec (int) - Number of total electrons
            norb (int) - Number of spatial orbitals
        """
        self.nelec = nelec
        self.norb = norb
        self.dstr = "set{:02d}{:02d}".format(nelec, norb)
        ipath = fud.__file__.replace("__init__.py", "fqe_data_set")
        self.file_path = os.path.join(ipath, self.dstr)
        _data = dict()
        for nbeta in range(0, nelec + 1):
            nalpha = nelec - nbeta
            sector = FqeData(nalpha, nbeta, norb)
            nstr = "{:02d}{:02d}{:02d}".format(nalpha, nbeta, norb)
            fr = os.path.join(self.file_path, "cr" + nstr + ".npy")
            fi = os.path.join(self.file_path, "ci" + nstr + ".npy")
            cr = numpy.fromfile(fr).reshape(sector.coeff.shape)
            ci = numpy.fromfile(fi).reshape(sector.coeff.shape)
            sector.coeff = cr + 1.j * ci
            _data[(nalpha, nbeta)] = sector
        self.data = FqeDataSet(self.nelec, self.norb, _data)

    def get_fqe_data_set(self):
        """Return a copy of the FqeDataset object"""
        return copy.deepcopy(self.data)

    def get_href(self, hstr):
        """Get reference coefficients after application of an operator
        """
        out = self._read_fqe_data('_' + hstr)
        return out

    def get_rdm(self, order):
        """Get the RDM of the specified order."""
        fr = os.path.join(self.file_path, "dr" + str(order) + ".npy")
        fi = os.path.join(self.file_path, "di" + str(order) + ".npy")
        shape = tuple([2 * self.norb] * order * 2)
        dr = numpy.fromfile(fr).reshape(shape)
        di = numpy.fromfile(fi).reshape(shape)
        return dr + 1.j * di

    def get_indv_ref(self, daga, undaga, dagb, undagb):
        """Get reference coefficients for applications of an
        individual N-body operator."""
        sdaga = "".join([str(da) for da in daga])
        sundaga = "".join([str(uda) for uda in undaga])
        sdagb = "".join([str(db) for db in dagb])
        sundagb = "".join([str(udb) for udb in undagb])
        suffix = "indv" + sdaga + "_" + sundaga + "_" + sdagb + "_" + sundagb
        return self._read_fqe_data(suffix)

    def get_ievo_ref(self, daga, undaga, dagb, undagb):
        """Get reference coefficients for applications of an
        individual N-body operator."""
        sdaga = "".join([str(da) for da in daga])
        sundaga = "".join([str(uda) for uda in undaga])
        sdagb = "".join([str(db) for db in dagb])
        sundagb = "".join([str(udb) for udb in undagb])
        suffix = "ievo" + sdaga + "_" + sundaga + "_" + sdagb + "_" + sundagb
        return self._read_fqe_data(suffix)

    def _read_fqe_data(self, suffix):
        _data = dict()
        for nbeta in range(0, self.nelec + 1):
            nalpha = self.nelec - nbeta
            nstr = "{:02d}{:02d}{:02d}".format(nalpha, nbeta, self.norb)
            sector = FqeData(nalpha, nbeta, self.norb)
            crfile = os.path.join(self.file_path, "cr" + nstr + suffix + ".npy")
            cifile = os.path.join(self.file_path, "ci" + nstr + suffix + ".npy")

            cr = numpy.fromfile(crfile).reshape(sector.coeff.shape)
            ci = numpy.fromfile(cifile).reshape(sector.coeff.shape)

            sector.coeff = cr + 1.j * ci
            _data[(nalpha, nbeta)] = sector
        return FqeDataSet(self.nelec, self.norb, _data)
