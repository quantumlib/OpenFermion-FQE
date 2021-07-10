import fqe
import numpy
import openfermion
import cProfile

if __name__ == "__main__":
    norbs = 12
    nel = 12
    sz = 0

    fqe_wfn = fqe.Wavefunction(param=[[nel, sz, norbs]])
    fqe_wfn.set_wfn(strategy='random')
    # binarycode = None
    binarycode = openfermion.transforms.bravyi_kitaev_code(norbs * 2)

    cProfile.run('cirq_wfn = fqe.to_cirq(fqe_wfn, binarycode=binarycode)',
                 'fqe_to_cirq.profile')
    cProfile.run('fqe_wfn_from_cirq = fqe.from_cirq(cirq_wfn, thresh=1e-7, '
                 'binarycode=binarycode)', 'fqe_from_cirq.profile')
    # fqe_wfn_from_cirq.print_wfn()
    # fqe_wfn.print_wfn()
    # print(cirq_wfn)
    # print(cirq_wfn[numpy.nonzero(cirq_wfn)].T)
    assert(numpy.allclose(fqe.to_cirq(fqe_wfn_from_cirq,
                                      binarycode=binarycode), cirq_wfn))

    import pstats
    profile = pstats.Stats('fqe_to_cirq.profile')
    profile.sort_stats('cumtime')
    profile.print_stats(30)
    profile = pstats.Stats('fqe_from_cirq.profile')
    profile.sort_stats('cumtime')
    profile.print_stats(30)
