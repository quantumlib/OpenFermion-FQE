import numpy
from fqe.fci_graph import Spinmap, FciGraph
from fqe.fqe_data import FqeData
from fqe.fqe_data_set import FqeDataSet
from fqe.wavefunction import Wavefunction


def compare_Spinmap(A: Spinmap, B: Spinmap) -> bool:
    """Compares two Spinmaps

    Args:
        A: First spinmap
        B: Second spinmap

    Returns:
        (bool) - A == B
    """
    assert A.keys() == B.keys()
    for k in A.keys():
        assert numpy.array_equal(A[k], B[k])
    return True


def compareFciGraph(graph1: 'FciGraph', graph2: 'FciGraph'):
    """Compares the equality between two FciGraph objects.

    Args:
        graph1: The FciGraph object to compare to.
        graph2: The FciGraph object to compare to.
    """
    assert graph1._norb == graph2._norb
    assert graph1._nalpha == graph2._nalpha
    assert graph1._nbeta == graph2._nbeta
    assert graph1._lena == graph2._lena
    assert graph1._lenb == graph2._lenb
    assert numpy.array_equal(graph1._astr, graph2._astr)
    assert numpy.array_equal(graph1._bstr, graph2._bstr)
    assert numpy.array_equal(graph1._aind, graph2._aind)
    assert numpy.array_equal(graph1._bind, graph2._bind)
    assert compare_Spinmap(graph1._alpha_map, graph2._alpha_map)
    assert compare_Spinmap(graph1._beta_map, graph2._beta_map)
    assert numpy.array_equal(graph1._dexca, graph2._dexca)
    assert numpy.array_equal(graph1._dexcb, graph2._dexcb)
    return True


def FqeData_isclose(data1: 'FqeData', data2: 'FqeData', **kwargs) -> bool:
    """Compares two FqeData's end compare their closeness.

    Args:
        data1 (FqeData) - The set to compare to.
        data2 (FqeData) - The set to compare to.

    Kwargs:
        rtol (float) - The relative tolerance parameter (as in numpy.isclose).
        atol (float) - The absolute tolerance parameter (as in numpy.isclose).

    Returns:
        (bool) - if closeness is satisfied
    """
    assert data1._nele == data2._nele
    assert data1._m_s == data2._m_s
    assert compareFciGraph(data1._core, data2._core)
    assert data1._dtype == data2._dtype
    assert numpy.allclose(data1.coeff, data2.coeff, **kwargs)
    return True


def FqeDataSet_isclose(dataset1: 'FqeDataSet', dataset2: 'FqeDataSet',
                       **kwargs) -> bool:
    """Compares two FqeDataSet's end compare their closeness.

    Args:
        dataset1 (FqeDataSet) - The set to compare to.
        dataset2 (FqeDataSet) - The set to compare to.

    Kwargs:
        rtol (float) - The relative tolerance parameter (as in numpy.isclose).
        atol (float) - The absolute tolerance parameter (as in numpy.isclose).

    Returns:
        (bool) - if closeness is satisfied
    """
    assert dataset1._nele == dataset2._nele
    assert dataset1._norb == dataset2._norb
    assert dataset1._data.keys() == dataset2._data.keys()

    for k in dataset1._data:
        assert FqeData_isclose(dataset1._data[k], dataset2._data[k], **kwargs)
    return True


def Wavefunction_isclose(wfn1: 'Wavefunction', wfn2: 'Wavefunction',
                         **kwargs) -> bool:
    """Compares two Wavefunction's end compare their closeness.

    Args:
        wfn1 (Wavefunction) - The set to compare to.
        wfn2 (Wavefunction) - The set to compare to.

    Kwargs:
        rtol (float) - The relative tolerance parameter (as in numpy.isclose).
        atol (float) - The absolute tolerance parameter (as in numpy.isclose).

    Returns:
        (bool) - if closeness is satisfied
    """
    assert wfn1._symmetry_map == wfn2._symmetry_map
    assert wfn1._conserved == wfn2._conserved
    assert wfn1._conserve_spin == wfn2._conserve_spin
    assert wfn1._conserve_number == wfn2._conserve_number
    assert wfn1._norb == wfn2._norb
    assert wfn1._civec.keys() == wfn2._civec.keys()
    for k in wfn1._civec:
        assert FqeData_isclose(wfn1._civec[k], wfn2._civec[k], **kwargs)
    return True
