Examples
********

Time evolution with dense Hamiltonians
======================================

Most basic example
------------------

Here, we consider time evolution of wave functions for LiH.
This also shows an example as to how to construct the Hamiltonian;
note that 'restricted' means that the Hamiltonian conserves both the spin and number symmetries.

.. code-block::

    from openfermion import FermionOperator
    import numpy
    import fqe
    from fqe.unittest_data import build_lih_data, build_hamiltonian

    h1e, h2e, wfn = build_lih_data.build_lih_data('energy')
    lih_hamiltonian = fqe.get_restricted_hamiltonian(tuple([h1e, h2e]))
    lihwfn = fqe.Wavefunction([[4, 0, 6]])
    lihwfn.set_wfn(strategy='from_data', raw_data={(4, 0): wfn})

This one- and two-body Hamiltonian can be used to time-evolve the wave function

.. code-block::

    time = 0.01
    evovled = lihwfn.time_evolve(time, lih_hamiltonian)

One can also perform the same operation using a wrapper function,

.. code-block::

    evolved = fqe.time_evolve(lihwfn, time, lih_hamiltonian)

Diagonal Hamiltonians
---------------------

There is a specialized code for diagonal Hamiltonians.

.. code-block::

    wfn = fqe.Wavefunction([[4, 0, 4]])
    wfn.set_wfn(strategy='random')

    diagonal = FermionOperator('0^ 0', -2.0) + \
               FermionOperator('1^ 1', -2.0) + \
               FermionOperator('2^ 2', -0.7) + \
               FermionOperator('3^ 3', -0.7) + \
               FermionOperator('4^ 4', -0.1) + \
               FermionOperator('5^ 5', -0.1) + \
               FermionOperator('6^ 6', 0.5) + \
               FermionOperator('7^ 7', 0.5)

    evolved = wfn.time_evolve(time, diagonal)

Quadratic Hamiltonians
----------------------

By taking advantage of the fact that quadratic Hamiltonians can be easily brought into a diagonal form by means of a unitary transformation of
oirbitals, we time-evolve wave functions exactly with quadratic Hamiltonians.

An example when the spin and number symmetries are both conserved:

.. code-block::

    there has to be an example

An example when the spin symmetry is broken (but the number symmetry is conserved):

.. code-block::

    norb = 4
    h1e = numpy.zeros((2*norb, 2*norb), dtype=numpy.complex128)
    for i in range(2*norb):
        for j in range(2*norb):
            h1e[i, j] += (i+j) * 0.02
        h1e[i, i] += i * 2.0

    hamil = fqe.get_gso_hamiltonian(tuple([h1e]))
    wfn = fqe.get_number_conserving_wavefunction(4, norb)
    wfn.set_wfn(strategy='random')
    evolved = wfn.time_evolve(time, hamil)

An example when the number symmetry is broken (but the spin symmetry is conserved) is as follows.
Note that the Hamiltonian in this example is initialized from the FermionOperator object in build_hamiltonian.number_nonconserving_fop.

.. code-block::

    norb = 4
    wfn = fqe.get_spin_conserving_wavefunction(2, norb)
    hamil = build_hamiltonian.number_nonconserving_fop(2, norb)
    wfn.set_wfn(strategy='random')
    evolved = wfn.time_evolve(time, hamil)


Diagonal Coulomb Operators
--------------------------

Time evolution with diagonal Coulomb operators can be performed exactly. This has been implemented as a specialization as well.

.. code-block::

    norb = 4
    wfn = fqe.Wavefunction([[5, 1, norb]])
    vij = numpy.zeros((norb, norb, norb, norb), dtype=numpy.complex128)
    for i in range(norb):
        for j in range(norb):
            vij[i, j] += 4*(i % norb + 1)*(j % norb + 1)*0.21

    wfn.set_wfn(strategy='random')
    hamil = fqe.get_diagonalcoulomb_hamiltonian(vij)
    evolved = wfn.time_evolve(time, hamil)

Individual N-body Generators
----------------------------

If the generator only consists of one term and its Hermitian conjugate, the time evolution can be performed exactly, of which our code internally takes advantage.

.. code-block::

    norb = 3
    nele = 4
    ops = FermionOperator('5^ 1^ 2 0', 3.0 - 1.j)
    ops += FermionOperator('0^ 2^ 1 5', 3.0 + 1.j)
    wfn = fqe.get_number_conserving_wavefunction(nele, norb)
    wfn.set_wfn(strategy='random')
    wfn.normalize()
    evolved = wfn.time_evolve(time, ops)

Dense Hamiltonians
------------------

Time evolution with dense Hamiltonians are performed using polynomial expansions of the exponential. The Taylor and Chebyshev expansions are available.
The time evolution of a wave function using the LiH Hamiltonian above is

.. code-block::

    taylor_wfn = lihwfn.apply_generated_unitary(time, 'taylor', lih_hamiltonian, accuracy=1.e-8)

An example for those with the Chebyshev expansion is as follows. Note that one has to specify the lower and upper bounds of the spectrum:

.. code-block::

    norb = 2
    nalpha = 1
    nbeta = 1
    nele = nalpha + nbeta
    time = 0.05
    h1e = numpy.zeros((norb*2, norb*2), dtype=numpy.complex128)
    for i in range(2*norb):
        for j in range(2*norb):
            h1e[i, j] += (i+j) * 0.02
        h1e[i, i] += i * 2.0
    hamil = fqe.get_general_hamiltonian(tuple([h1e]))
    spec_lim = [4.074913702385936, 8.165086297614062]
    wfn = fqe.Wavefunction([[nele, nalpha - nbeta, norb]])
    wfn.set_wfn(strategy='random')
    evol_wfn = wfn.apply_generated_unitary(time, 'chebyshev', hamil, spec_lim=spec_lim)

RDMs and Expectation Values
===========================

The RDMs and expectation values can be computed from wavefunctions using the following syntax.
The wrapped APIs allow for computation of transition RDMs and their elements.

.. code-block::

    rdm1 = lihwfn.expectationValue('i^ j')
    val = lihwfn.expectationValue('5^ 3')

    trdm1 = fqe.expectationValue(lihwfn2, 'i^ j', lihwfn)
    val = fqe.expectationValue(lihwfn2, '5^ 3', lihwfn)

The higher-rank RDMs can be obtained in the same manner. Note that, in the following example,
we are calculating the hole RDMs.

.. code-block::

    hrdm2 = lihwfn.expectationValue('i j k^ l^')
    hrdm2 = fqe.expectationValue(lihwfn, 'i j k^ l^', lihwfn)

Other expectation values can be obtained as

.. code-block::

    li_h_energy = lihwfn.expectationValue(lih_hamiltonian)
    li_h_energy = fqe.expectationValue(lihwfn, lih_hamiltonian, lihwfn)

In addition, there are specializations of Operator that allows computation of the expectation values of the symmetry operators.

.. code-block::

    op = fqe.get_s2_operator()
    print(lihwfn.expectationValue(op))

    op = fqe.get_sz_operator()
    print(lihwfn.expectationValue(op))

    op = fqe.get_time_reversal_operator()
    print(lihwfn.expectationValue(op))

    op = fqe.get_number_operator()
    print(lihwfn.expectationValue(op))
