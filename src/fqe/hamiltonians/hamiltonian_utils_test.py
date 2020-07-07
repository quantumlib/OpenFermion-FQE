import numpy as np
import openfermion as of
from fqe.hamiltonians.hamiltonian_utils import gather_nbody_spin_sectors


def test_nbody_spin_sectors():
    """
    Parse an openfermion op into alpha-components and beta components
    return the parity for converting into the partitioned ordering.
    """
    op = of.FermionOperator(((3, 1), (4, 1), (2, 0), (0, 0)),
                            coefficient=1.0 + 0.5j)
    # op += of.hermitian_conjugated(op)
    coefficient, parity, alpha_sub_ops, beta_sub_ops = \
        gather_nbody_spin_sectors(op)
    assert np.isclose(coefficient.real, 1.0)
    assert np.isclose(coefficient.imag, 0.5)
    assert np.isclose(parity, -1)
    assert tuple(map(tuple, alpha_sub_ops)) == ((4, 1), (2, 0), (0, 0))
    assert tuple(map(tuple, beta_sub_ops)) == ((3, 1),)
