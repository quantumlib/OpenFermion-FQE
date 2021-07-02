import cProfile
import pstats
import fqe
import numpy
from generate_H_ring import get_H_ring_data

amount_H = 12
molecule = get_H_ring_data(amount_H)

print('Hartree-Fock energy of {} Hartree.'.format(molecule.hf_energy))

nele = molecule.n_electrons
sz = molecule.multiplicity - 1
norbs = molecule.n_orbitals
h1, h2 = molecule.get_integrals()

fqe_wf = fqe.Wavefunction([[nele, sz, norbs]])
fqe_wf.set_wfn(strategy='hartree-fock')
fqe_wf.normalize()

hamiltonian = fqe.get_restricted_hamiltonian(
    (h1, numpy.einsum("ijlk", -0.5 * h2)),
    e_0=molecule.nuclear_repulsion
)
initial_energy = fqe_wf.expectationValue(hamiltonian)

print(f'Initial Energy: {initial_energy}')
time = 0.1
cProfile.run('evolved = fqe_wf.time_evolve(time, hamiltonian)',
             'fqe_H_ring.profile')
final_energy = evolved.expectationValue(hamiltonian)
print(f'Final Energy: {final_energy}')

profile = pstats.Stats('fqe_H_ring.profile')
profile.sort_stats('cumtime')
profile.print_stats(30)
