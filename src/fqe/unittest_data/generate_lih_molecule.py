import os
from openfermion import MolecularData
import fqe.unittest_data as fud


def build_lih_moleculardata():  # pragma: no cover
    # Set up molecule.
    geometry = [('Li', (0., 0., 0.)), ('H', (0., 0., 1.45))]
    basis = 'sto-3g'
    multiplicity = 1
    filename = os.path.join(fud.__file__.replace('__init__.py', ''),
                            'H1-Li1_sto-3g_singlet_1.45.hdf5')
    molecule = MolecularData(geometry,
                             basis,
                             multiplicity,
                             filename=filename)
    molecule.load()
    return molecule
