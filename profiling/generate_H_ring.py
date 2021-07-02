import os
from openfermion.chem import make_atomic_ring


def get_H_ring_data(amount_H):
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            f'Hring_{amount_H}.hdf5')
    molecule = make_atomic_ring(amount_H, 1.0, 'sto-3g',
                                atom_type='H', charge=0, filename=filename)

    if (os.path.isfile(filename)):
        molecule.load()

    if molecule.hf_energy is None:
        molecule = generate_H_ring_data(molecule)
    return molecule


def generate_H_ring_data(molecule):
    from openfermionpyscf import run_pyscf

    print("calculating")
    molecule = run_pyscf(molecule, run_scf=True)
    molecule.save()
    return molecule
