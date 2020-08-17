import numpy as np
import cirq
from openfermion.circuits.primitives import optimal_givens_decomposition
from openfermion import givens_decomposition_square
from openfermion import random_quadratic_hamiltonian
import openfermion as of
from scipy.linalg import expm
import fqe
from fqe.algorithm.low_rank import evolve_fqe_givens
import time


def evolve_cirq_givens(initial_wf: np.ndarray, u: np.ndarray):
    n_qubits = int(np.log2(initial_wf.shape[0]))
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit(optimal_givens_decomposition(qubits, u.copy()))
    final_state = circuit.final_wavefunction(initial_state=initial_wf.flatten())
    return final_state


def time_cirq(initial_wf: np.ndarray, u: np.ndarray, trials=1):
    times = []
    for i in range(trials):
        start_time = time.time()
        _ = evolve_cirq_givens(initial_wf, u)
        times.append(time.time() - start_time)
    return times


def time_fqe_givens(initial_wf: fqe.Wavefunction, u: np.ndarray, trials=1):
    times = []
    for i in range(trials):
        start_time = time.time()
        _ = evolve_fqe_givens(initial_wf, u)
        times.append(time.time() - start_time)
    return times


def time_fqe_hamiltonian(initial_wf: fqe.Wavefunction,
                         ham: fqe.restricted_hamiltonian.RestrictedHamiltonian,
                         trials=1):
    times = []
    for i in range(trials):
        start_time = time.time()
        _ = initial_wf.time_evolve(1., ham)
        times.append(time.time() - start_time)
    return times


if __name__ == "__main__":
    import cProfile
    import numpy as np
    import cirq
    from openfermion.circuits.primitives import optimal_givens_decomposition
    from openfermion import random_quadratic_hamiltonian
    import openfermion as of
    from scipy.linalg import expm
    import fqe
    from fqe.algorithm.low_rank import evolve_fqe_givens
    import time
    norbs = 12
    sz = 0
    nelec = norbs
    start_time = time.time()
    initial_wfn = fqe.Wavefunction([[nelec, sz, norbs]])
    print("Wavefunction Initialization ", time.time() - start_time)
    graph = initial_wfn.sector((nelec, sz)).get_fcigraph()
    hf_wf = np.zeros((graph.lena(), graph.lenb()), dtype=np.complex128)
    hf_wf[0, 0] = 1
    start_time = time.time()
    cirq_wf = of.jw_hartree_fock_state(nelec, 2 * norbs)
    print("Cirq wf initialization time ", time.time() - start_time)
    initial_wfn.set_wfn(strategy='from_data',
                        raw_data={(nelec, sz): hf_wf})

    # set up Hamiltonian
    ikappa = random_quadratic_hamiltonian(norbs, conserves_particle_number=True,
                                          real=True, expand_spin=False, seed=5)
    ikappa_matrix = ikappa.n_body_tensors[1, 0]

    # Evolution time and unitaries
    dt = 0.275
    u = expm(-1j * dt * ikappa_matrix)
    fqe_ham = fqe.restricted_hamiltonian.RestrictedHamiltonian(
        (ikappa_matrix * dt,))

    # evolve_fqe_givens(initial_wfn, u)
    # evolve_fqe_of_givens(initial_wfn, u)
    # cProfile.run('evolve_cirq_givens(cirq_wf, np.kron(u, np.eye(2)))', 'fqe_givens_profile')
    cProfile.run('evolve_fqe_givens(initial_wfn, u)', 'fqe_givens_profile')
    # cProfile.run('initial_wfn.time_evolve(1., fqe_ham)', 'fqe_givens_profile')

    import pstats
    profile = pstats.Stats('fqe_givens_profile')
    profile.sort_stats('cumtime')
    profile.print_stats(30)
