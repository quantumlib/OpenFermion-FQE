import cProfile
import numpy as np
import fqe
import time


if __name__ == "__main__":
    norbs = 20
    sz = 0
    nelec = norbs
    nelec = 10
    start_time = time.time()
    initial_wfn = fqe.Wavefunction([[nelec, sz, norbs]])
    # cProfile.run('initial_wfn = fqe.Wavefunction([[nelec, sz, norbs]])',
    #              'fqe_wavef.profile')
    graph = initial_wfn.sector((nelec, sz)).get_fcigraph()
    print("Wavefunction Initialization ", time.time() - start_time)
    hf_wf = np.zeros((graph.lena(), graph.lenb()), dtype=np.complex128)
    initial_wfn.set_wfn(strategy='from_data', raw_data={(nelec, sz): hf_wf})
    exit(0)

    import pstats
    profile = pstats.Stats('fqe_wavef.profile')
    profile.sort_stats('cumtime')
    profile.print_stats(30)
