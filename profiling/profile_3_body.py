import numpy
from fqe import Wavefunction
import cProfile

if __name__ == '__main__':
    norb = 10
    nel = norb
    sz = 0
    h1e_spa = numpy.zeros((norb, norb), dtype=numpy.complex128)
    h2e_spa = numpy.zeros((norb, norb, norb, norb), dtype=numpy.complex128)
    h3e_spa = numpy.zeros((norb, norb, norb, norb, norb, norb),
                          dtype=numpy.complex128)

    for i in range(norb):
        for j in range(norb):
            for k in range(norb):
                for l in range(norb):
                    for m in range(norb):
                        for n in range(norb):
                            h3e_spa[i, j, k, l, m, n] += (i + l) * (j + m) * (
                                k + n) * 0.002

    wfn = Wavefunction([[nel, sz, norb]])
    wfn.set_wfn(strategy='random')

    cProfile.run('test = wfn.apply(tuple([h1e_spa, h2e_spa, h3e_spa]))',
                 '3body.profile')
    # test = wfn.apply(tuple([h1e_spa, h2e_spa, h3e_spa]))

    # rdm3 = wfn.rdm123(wfn)
    # energy = numpy.tensordot(h3e_spa, rdm3[2],
    #                          axes=([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]))
    # print(energy)
    import pstats
    profile = pstats.Stats('3body.profile')
    profile.sort_stats('cumtime')
    profile.print_stats(30)
