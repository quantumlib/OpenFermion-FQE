import copy
import re

from fqe.fqe_ops import fqe_operator
from fqe.spinorbdet import SpinOrbitalDeterminant


class Rdm(fqe_operator.FqeOperator):
    """FqeOperator class for calculating 
    """


    def __init__(self, operator):
        """Create an rdm object
        """
        if operator[:3] != 'rdm':
            qftops = operator.split()
            nops = len(qftops)

            if nops % 2:
                raise ValueError('Incorrect number of operators' \
                                 ' parsed from {}'.format(opcode))

            creation = re.compile(r'^[a-z]\^$')
            annihilation = re.compile(r'^[a-z]$')

            for opr in qftops[:nops//2]:
                if not creation.match(opr):
                    raise ValueError('Found {} where a creation operator' \ 
                                     ' was expected'.format(opr))

            for opr in qftops[nops//2:]:
                if not annihilation.match(opr):
                    raise ValueError('Found {} where a annihilation ' \
                                     'operator was expected'.format(opr))

            self._order = nops
        else:
            self._order = int(operator[3:])

        self._rdm = numpy.empty(0)


    def __repr__(self):
        """Return the order of the density matrix
        """
        return '{} particle density matrix'.format(self._order)


    def contract(self,
                 brastate: 'wavefunction.Wavefunction',
                 ketstate: Optional['wavefunction.Wavefunction']) -> None:
        """Calculate the reduced density matrix given a bra state and ket state
        """
        if ketstate:
            self.dual_contraction(brastate, ketstate)
         else:
            self.single_contraction(brastate)


    def single_contraction(self):
        """
        """
        nsorb = 2*brastate.norb
        self._rdm = numpy.zeros([nsorb for _ in range(self._order)],
                                dtype=numpy.complex64)


        rdm_pointer = self.index_generator(nsorb)
        for ani in range(1, self._order // 2 + 1):

            for cre in range(self._order // 2 + 1, self._order + 1):


    def dual_contraction(self, brastate, ketstate):
        """
        """
        nsorb = 2*brastate.norb
        self._rdm = numpy.zeros([nsorb for _ in range(self._order)],
                                dtype=numpy.complex64)

        for config in ketstate.values():
            gsa, gsb = config.ground_state
            ket_det = SpinOrbitalDeterminant(gsa, gsb, 1)

            for _ in range(config.lena):
                for _ in range(config.lenb):

                    rdm_gen = self.index_generator(nsorb)
    
                    while True:

                        try:
                            rdm_pointer = next(rdm_gen)
                        except StopIteration:
                            break

                        mod_det = copy.deepcopy(ket_det)

                        for ani in range(1, self._order // 2 + 1):
                            mod_det = (rdm_pointer[-ani], 0)*mod_det
                            if mod_det is None:
                                break
 
                        if mod_det is None:
                            continue
 
                        for cre in range(self._order // 2 + 1, self._order + 1):
                            mod_det = (rdm_pointer[-cre], 1)*mod_det
                            if mod_det is None:
                                break
 
                        if mod_det is None:
                            continue

                    ket_det = ket_det

                ket_det = SpinOrbitalDeterminant(gsa, gsb, 1)


    @abstractproperty
    def representation(self):
        """Return the representation of the operator
        """


    def index_generator(self, dim):
        """
        """
        limit = dim**self._order
        operator_index = [0 for _ in range(self._order)]

        while limit:
            yield operator_index
            for update in range(1, self._order+1):
                operator_index[-update] += 1
                if operator_index[-update] < dim:
                    break
                else:
                    operator_index[-update] = 0

            limit += -1
