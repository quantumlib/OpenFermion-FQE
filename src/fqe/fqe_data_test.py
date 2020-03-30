#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http:gc/www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""Unittesting for the fqe_data module
"""

import sys
import copy

import unittest

from io import StringIO

import numpy

from fqe import fqe_data
from fqe import fci_graph


class FqeDataTest(unittest.TestCase):
    """Unit tests
    """


    def test_fqe_init(self):
        """Check that we initialize the private values
        """
        test = fqe_data.FqeData(2, 4, 10)
        self.assertEqual(test.n_electrons(), 6)
        self.assertEqual(test.nalpha(), 2)
        self.assertEqual(test.nbeta(), 4)
        self.assertEqual(test.lena(), 45)
        self.assertEqual(test.lenb(), 210)
        pre_graph = fci_graph.FciGraph(2, 4, 10)
        test = fqe_data.FqeData(2, 4, 10, fcigraph=pre_graph)
        self.assertIsInstance(test, fqe_data.FqeData)




    def test_fqe_data_scale(self):
        """Scale the entire vector
        """
        test = fqe_data.FqeData(1, 1, 2)
        test.scale(0. + .0j)
        ref = numpy.zeros((2, 2), dtype=numpy.complex128)
        self.assertTrue(numpy.allclose(test.coeff, ref))




    def test_fqe_data_generator(self):
        """Access each element of any given vector
        """
        test = fqe_data.FqeData(1, 1, 2)
        gtest = test.generator()
        testx = list(next(gtest))
        self.assertListEqual([1, 1, .0 + 0.j], testx)
        testx = list(next(gtest))
        self.assertListEqual([1, 2, .0 + .0j], testx)
        testx = list(next(gtest))
        self.assertListEqual([2, 1, .0 + .0j], testx)
        testx = list(next(gtest))
        self.assertListEqual([2, 2, .0 + .0j], testx)




    def test_fqe_data_set_add_element_and_retrieve(self):
        """Set elements and retrieve them one by one
        """
        test = fqe_data.FqeData(1, 1, 2)
        test[(2, 2)] = 3.14 + .00159j
        self.assertEqual(test[(2, 2)], 3.14 + .00159j)
        test[(2, 1)] = 1.61 + .00803j
        self.assertEqual(test[(2, 1)], 1.61 + .00803j)




    def test_fqe_data_init_vec(self):
        """Set vectors in the fqedata set using different strategies
        """
        test = fqe_data.FqeData(1, 1, 2)
        test.set_wfn(strategy='ones')
        ref = numpy.ones((2, 2), dtype=numpy.complex128)
        self.assertTrue(numpy.allclose(test.coeff, ref))
        test.set_wfn(strategy='zero')
        ref = numpy.zeros((2, 2), dtype=numpy.complex128)
        self.assertTrue(numpy.allclose(test.coeff, ref))



    def test_fqe_data_set_wfn_data(self):
        """Set vectors in the fqedata set from a data block
        """
        test = fqe_data.FqeData(1, 1, 2)
        ref = numpy.random.rand(2, 2) + 1.j*numpy.random.rand(2, 2)
        test.set_wfn(strategy='from_data', raw_data=ref)
        self.assertTrue(numpy.allclose(test.coeff, ref))




    def test_fqe_data_manipulation(self):
        """The fqedata can be conjugated in place
        """
        test = fqe_data.FqeData(1, 1, 2)
        ref = numpy.random.rand(2, 2) + 1.j*numpy.random.rand(2, 2)
        test.set_wfn(strategy='from_data', raw_data=ref)
        self.assertTrue(numpy.allclose(test.beta_inversion(),
                                       ref[:, (1, 0)]))
        test.conj()
        self.assertTrue(numpy.allclose(test.coeff, numpy.conj(ref)))




    def test_fqe_data_initialize_errors(self):
        """There are many ways to not initialize a wavefunction
        """
        bad0 = numpy.ones((5, 3), dtype=numpy.complex64)
        bad1 = numpy.ones((4, 6), dtype=numpy.complex64)
        good1 = numpy.random.rand(2, 2) + numpy.random.rand(2, 2)*1.j
        test = fqe_data.FqeData(1, 1, 2)
        self.assertRaises(ValueError, test.set_wfn)
        self.assertRaises(ValueError, test.set_wfn, strategy='from_data')
        self.assertRaises(AttributeError, test.set_wfn, strategy='ones', raw_data=1)
        self.assertRaises(ValueError, test.set_wfn, strategy='onse')
        self.assertRaises(ValueError, test.set_wfn, strategy='ones', raw_data=good1)
        self.assertRaises(ValueError, test.set_wfn, strategy='from_data',
                          raw_data=bad0)
        self.assertRaises(ValueError, test.set_wfn, strategy='from_data',
                          raw_data=bad1)
        self.assertIsNone(test.set_wfn(strategy='from_data', raw_data=good1))
        bad_graph = fci_graph.FciGraph(5, 6, 7)
        self.assertRaises(ValueError,
                          fqe_data.FqeData,
                          1,
                          1,
                          2,
                          fcigraph=bad_graph)




    def test_fqe_data_vacuum(self):
        """Make sure that the vacuum exists
        """
        test = fqe_data.FqeData(0, 0, 2)
        self.assertEqual(test.n_electrons(), 0)
        self.assertEqual(test.nalpha(), 0)
        self.assertEqual(test.nbeta(), 0)
        self.assertEqual(test.lena(), 1)
        self.assertEqual(test.lenb(), 1)




    def test_apply_diagonal_unitary_dim(self):
        """Diagonal evoltion requires only the diagonal elements
        """
        test = fqe_data.FqeData(0, 2, 4)
        h1e = numpy.random.rand(4, 4)
        test.set_wfn(strategy='random')
        self.assertRaises(ValueError, test.evolve_diagonal, h1e)




    def test_apply_diagonal_inplace(self):
        """Check apply_diagonal_inplace for special cases
        """
        test = fqe_data.FqeData(1, 2, 4)
        test.set_wfn(strategy='random')
        bad_h1e = numpy.random.rand(6, 6)
        self.assertRaises(ValueError, test.apply_diagonal_inplace, bad_h1e)
        h1e = numpy.ones(8, dtype=numpy.complex128)
        ref = copy.deepcopy(test)
        print(ref.coeff)
        test.apply_diagonal_inplace(h1e)
        print(test.coeff)
        self.assertTrue(numpy.allclose(ref.coeff * 3.0, test.coeff))




    def test_1_body(self):
        """Check apply for one body terms
        """
        norb = 4
        scale = 4.071607802007311
        h1e_spa = numpy.zeros((norb, norb), dtype=numpy.complex128)
        for i in range(norb):
            for j in range(norb):
                h1e_spa[i, j] += (i+j) * 0.02
            h1e_spa[i, i] += i * 2.0

        h1e_spin = numpy.zeros((2*norb, 2*norb), dtype=numpy.complex128)
        h1e_spin[:norb, :norb] = h1e_spa
        h1e_spin[norb:, norb:] = h1e_spa

        wfn = numpy.asarray([[-0.9986416294264632 + 0.j,
                              0.0284839005060597 + 0.j,
                              0.0189102058837960 + 0.j,
                              -0.0096809878541792 + 0.j,
                              -0.0096884853951631 + 0.j,
                              0.0000930227399218 + 0.j],
                             [0.0284839005060596 + 0.j,
                              -0.0008124361774354 + 0.j,
                              -0.0005393690860379 + 0.j,
                              0.0002761273781438 + 0.j,
                              0.0002763412278424 + 0.j,
                              -0.0000026532545717 + 0.j],
                             [0.0189102058837960 + 0.j,
                              -0.0005393690860379 + 0.j,
                              -0.0003580822950200 + 0.j,
                              0.0001833184879206 + 0.j,
                              0.0001834604608161 + 0.j,
                              -0.0000017614718954 + 0.j],
                             [-0.0096809878541792 + 0.j,
                              0.0002761273781438 + 0.j,
                              0.0001833184879206 + 0.j,
                              -0.0000938490075630 + 0.j,
                              -0.0000939216898957 + 0.j,
                              0.0000009017769626 + 0.j],
                             [-0.0096884853951631 + 0.j,
                              0.0002763412278424 + 0.j,
                              0.0001834604608161 + 0.j,
                              -0.0000939216898957 + 0.j,
                              -0.0000939944285181 + 0.j,
                              0.0000009024753531 + 0.j],
                             [0.0000930227399218 + 0.j,
                              -0.0000026532545717 + 0.j,
                              -0.0000017614718954 + 0.j,
                              0.0000009017769626 + 0.j,
                              0.0000009024753531 + 0.j,
                              -0.0000000086650004 + 0.j]],
                            dtype=numpy.complex128)

        work = fqe_data.FqeData(2, 2, norb)
        work.coeff = numpy.copy(wfn)
        test = work.apply(tuple([h1e_spa]))
        self.assertTrue(numpy.allclose(numpy.multiply(wfn, scale), test.coeff))
        test = work.apply(tuple([h1e_spin]))
        self.assertTrue(numpy.allclose(numpy.multiply(wfn, scale), test.coeff))
        rdm1 = work.rdm1(work)
        energy = numpy.tensordot(h1e_spa, rdm1[0], axes=([0, 1], [0, 1]))
        self.assertAlmostEqual(energy, scale)




    def test_2_body(self):
        """Check apply for two body terms
        """
        norb = 4
        scale = -7.271991091302982
        h1e_spa = numpy.zeros((norb, norb), dtype=numpy.complex128)
        h2e_spa = numpy.zeros((norb, norb, norb, norb), dtype=numpy.complex128)
        for i in range(norb):
            for j in range(norb):
                for k in range(norb):
                    for l in range(norb):
                        h2e_spa[i, j, k, l] += (i+k)*(j+l)*0.02

        h2e_spin = numpy.zeros((2*norb, 2*norb, 2*norb, 2*norb), dtype=numpy.complex128)
        h2e_spin[norb:, norb:, norb:, norb:] = h2e_spa
        h2e_spin[:norb, norb:, :norb, norb:] = h2e_spa
        h2e_spin[norb:, :norb, norb:, :norb] = h2e_spa
        h1e_spin = numpy.zeros((2*norb, 2*norb), dtype=numpy.complex128)

        wfn = numpy.asarray([[-0. + 0.j,
                              -0.0228521148088829 + 0.j,
                              0.0026141627151228 + 0.j,
                              -0.0350670839771777 + 0.j,
                              0.0040114914627326 + 0.j,
                              0.0649058425241095 + 0.j],
                             [0.0926888005534875 + 0.j,
                              -0.0111089171383541 + 0.j,
                              0.0727533722636203 + 0.j,
                              -0.2088241794624313 + 0.j,
                              -0.1296798587246719 + 0.j,
                              0.1794528138441346 + 0.j],
                             [-0.0106031152277513 + 0.j,
                              -0.0163529872130018 + 0.j,
                              -0.0063065366721411 + 0.j,
                              -0.0031557043526629 + 0.j,
                              0.0179284033727408 + 0.j,
                              0.0295275972773592 + 0.j],
                             [0.1422330484480838 + 0.j,
                              0.0302351700257210 + 0.j,
                              0.1062328653022619 + 0.j,
                              -0.2478900249182474 + 0.j,
                              -0.2072966151359754 + 0.j,
                              0.1410812707838873 + 0.j],
                             [-0.0162707187155699 + 0.j,
                              0.0344029957577157 + 0.j,
                              -0.0164836712764185 + 0.j,
                              0.0864570191796392 + 0.j,
                              0.0170673494703135 + 0.j,
                              -0.1236759770908710 + 0.j],
                             [-0.2632598664406648 + 0.j,
                              -0.0049122508165550 + 0.j,
                              -0.2024668199753644 + 0.j,
                              0.5371585425189981 + 0.j,
                              0.3747249320637363 + 0.j,
                              -0.4061235786466031 + 0.j]],
                            dtype=numpy.complex128)
        work = fqe_data.FqeData(2, 2, norb)
        work.coeff = numpy.copy(wfn)
        test = work.apply(tuple([h1e_spa, h2e_spa]))
        self.assertTrue(numpy.allclose(numpy.multiply(wfn, scale), test.coeff))
        test = work.apply(tuple([h1e_spin, h2e_spin]))
        self.assertTrue(numpy.allclose(numpy.multiply(wfn, scale), test.coeff))

        energy = 0.
        rdm2 = work.rdm12(work)
        energy = numpy.tensordot(h2e_spa, rdm2[1], axes=([0, 1, 2, 3], [0, 1, 2, 3]))
        self.assertAlmostEqual(energy, scale)




    def test_3_body(self):
        """Check appply for three body terms
        """
        norb = 4
        scale = -0.3559955456514945
        h1e_spa = numpy.zeros((norb, norb), dtype=numpy.complex128)
        h2e_spa = numpy.zeros((norb, norb, norb, norb), dtype=numpy.complex128)
        h3e_spa = numpy.zeros((norb, norb, norb, norb, norb, norb), dtype=numpy.complex128)
        for i in range(norb):
            for j in range(norb):
                for k in range(norb):
                    for l in range(norb):
                        for m in range(norb):
                            for n in range(norb):
                                h3e_spa[i, j, k, l, m, n] += (i+l)*(j+m)*(k+n)*0.002

        h3e_spin = numpy.zeros((2*norb, 2*norb,
                                2*norb, 2*norb,
                                2*norb, 2*norb), dtype=numpy.complex128)

        h3e_spin[:norb, :norb, :norb, :norb, :norb, :norb] = \
            h3e_spa[:norb, :norb, :norb, :norb, :norb, :norb]
        h3e_spin[norb:, norb:, :norb, norb:, norb:, :norb] = \
            h3e_spa[:norb, :norb, :norb, :norb, :norb, :norb]
        h3e_spin[:norb, norb:, :norb, :norb, norb:, :norb] = \
            h3e_spa[:norb, :norb, :norb, :norb, :norb, :norb]
        h3e_spin[norb:, :norb, :norb, norb:, :norb, :norb] = \
            h3e_spa[:norb, :norb, :norb, :norb, :norb, :norb]
        h3e_spin[:norb, :norb, norb:, :norb, :norb, norb:] = \
            h3e_spa[:norb, :norb, :norb, :norb, :norb, :norb]
        h3e_spin[norb:, norb:, norb:, norb:, norb:, norb:] = \
            h3e_spa[:norb, :norb, :norb, :norb, :norb, :norb]
        h3e_spin[:norb, norb:, norb:, :norb, norb:, norb:] = \
            h3e_spa[:norb, :norb, :norb, :norb, :norb, :norb]
        h3e_spin[norb:, :norb, norb:, norb:, :norb, norb:] = \
            h3e_spa[:norb, :norb, :norb, :norb, :norb, :norb]

        h2e_spin = numpy.zeros((2*norb, 2*norb,
                                2*norb, 2*norb), dtype=numpy.complex128)
        h1e_spin = numpy.zeros((2*norb, 2*norb), dtype=numpy.complex128)

        wfn = numpy.asarray([[-0.0314812075046431 + 0.j,
                              -0.0297693820182802 + 0.j,
                              -0.3098997729788456 + 0.j,
                              -0.0160305969536710 + 0.j,
                              -0.1632524087723557 + 0.j,
                              0.0034291897632257 + 0.j],
                             [0.0164437672481284 + 0.j,
                              0.0992736004782678 + 0.j,
                              -0.3815809991854478 + 0.j,
                              0.0473449883500741 + 0.j,
                              -0.1676924530298831 + 0.j,
                              0.0862645617838693 + 0.j],
                             [0.1945647573956160 + 0.j,
                              0.4887086137642586 + 0.j,
                              -0.0626741792078922 + 0.j,
                              0.2409165890485374 + 0.j,
                              0.0882595335020335 + 0.j,
                              0.2992959491992316 + 0.j],
                             [0.0054805849814896 + 0.j,
                              0.0441542029539851 + 0.j,
                              -0.1990143955204461 + 0.j,
                              0.0209311955324630 + 0.j,
                              -0.0893288238000901 + 0.j,
                              0.0403909822493594 + 0.j],
                             [0.0715644878248155 + 0.j,
                              0.2095150416316441 + 0.j,
                              -0.2162188374553924 + 0.j,
                              0.1024657089267621 + 0.j,
                              -0.0574510118765082 + 0.j,
                              0.1413852823605560 + 0.j],
                             [-0.0035087946895569 + 0.j,
                              0.0261754436118892 + 0.j,
                              -0.2259825345335916 + 0.j,
                              0.0119418158614160 + 0.j,
                              -0.1073075831421863 + 0.j,
                              0.0314016025783124 + 0.j]],
                            dtype=numpy.complex128)
        work = fqe_data.FqeData(2, 2, norb)
        work.coeff = numpy.copy(wfn)
        test = work.apply(tuple([h1e_spa, h2e_spa, h3e_spa]))
        self.assertTrue(numpy.allclose(numpy.multiply(wfn, scale), test.coeff))
        test = work.apply(tuple([h1e_spin, h2e_spin, h3e_spin]))
        self.assertTrue(numpy.allclose(numpy.multiply(wfn, scale), test.coeff))

        energy = 0.
        rdm3 = work.rdm123(work)
        energy = numpy.tensordot(h3e_spa,
                                 rdm3[2],
                                 axes=([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]))
        self.assertAlmostEqual(energy, scale)


    def test_lowfilling_2_body(self):
        """Check low filling 2 body functions
        """
        norb = 8
        scale = -127.62690492408638
        h1e_spa = numpy.zeros((norb, norb), dtype=numpy.complex128)
        h2e_spa = numpy.zeros((norb, norb, norb, norb), dtype=numpy.complex128)
        for i in range(norb):
            for j in range(norb):
                h1e_spa[i, j] += (i+j) * 0.02
                for k in range(norb):
                    for l in range(norb):
                        h2e_spa[i, j, k, l] += (i+k)*(j+l)*0.02
            h1e_spa[i, i] += i * 2.0

        h2e_spin = numpy.zeros((2*norb, 2*norb,
                                2*norb, 2*norb), dtype=numpy.complex128)
        h2e_spin[:norb, :norb, :norb, :norb] = h2e_spa
        h2e_spin[norb:, norb:, norb:, norb:] = h2e_spa
        h2e_spin[:norb, norb:, :norb, norb:] = h2e_spa
        h2e_spin[norb:, :norb, norb:, :norb] = h2e_spa

        h1e_spin = numpy.zeros((2*norb, 2*norb), dtype=numpy.complex128)
        h1e_spin[:norb, :norb] = h1e_spa
        h1e_spin[norb:, norb:] = h1e_spa

        wfn = numpy.asarray([[-0.0932089487476626 + 0.j,
                              -0.0706587098642184 + 0.j,
                              -0.0740438603927790 + 0.j,
                              -0.0805502046061131 + 0.j,
                              -0.0879038682813978 + 0.j,
                              -0.0955090389840755 + 0.j,
                              -0.1031456871031518 + 0.j,
                              0.0478568383864201 + 0.j,
                              0.0633827785124240 + 0.j,
                              0.0745259163770606 + 0.j,
                              0.0840931201605038 + 0.j,
                              0.0928361109948333 + 0.j,
                              0.1010454451455466 + 0.j,
                              0.0100307721513250 + 0.j,
                              0.0151358782333622 + 0.j,
                              0.0186106754313070 + 0.j,
                              0.0213309864526485 + 0.j,
                              0.0236300924351082 + 0.j,
                              0.0044254845218365 + 0.j,
                              0.0070218120829127 + 0.j,
                              0.0087913981858984 + 0.j,
                              0.0101147655457833 + 0.j,
                              0.0023845600555775 + 0.j,
                              0.0038543613277461 + 0.j,
                              0.0048365508368202 + 0.j,
                              0.0013785456136308 + 0.j,
                              0.0022237693216097 + 0.j,
                              0.0007983521363725 + 0.j],
                             [-0.0706587098642178 + 0.j,
                              -0.0535799644805679 + 0.j,
                              -0.0561621201665438 + 0.j,
                              -0.0611127396392170 + 0.j,
                              -0.0667080800401726 + 0.j,
                              -0.0724963145713807 + 0.j,
                              -0.0783104615368883 + 0.j,
                              0.0362946558453897 + 0.j,
                              0.0480801293766977 + 0.j,
                              0.0565452115431265 + 0.j,
                              0.0638177200315103 + 0.j,
                              0.0704673713830835 + 0.j,
                              0.0767142655027953 + 0.j,
                              0.0076098535772196 + 0.j,
                              0.0114848140629482 + 0.j,
                              0.0141238594629412 + 0.j,
                              0.0161911260447960 + 0.j,
                              0.0179393197803398 + 0.j,
                              0.0033581855204487 + 0.j,
                              0.0053290007599329 + 0.j,
                              0.0066727963434466 + 0.j,
                              0.0076782000447207 + 0.j,
                              0.0018096892986237 + 0.j,
                              0.0029253127397735 + 0.j,
                              0.0036709502967018 + 0.j,
                              0.0010461510220687 + 0.j,
                              0.0016874859899826 + 0.j,
                              0.0006056338822255 + 0.j],
                             [-0.0740438603927787 + 0.j,
                              -0.0561621201665439 + 0.j,
                              -0.0588835317647236 + 0.j,
                              -0.0640892811820629 + 0.j,
                              -0.0699730349182325 + 0.j,
                              -0.0760611581510978 + 0.j,
                              -0.0821784870045671 + 0.j,
                              0.0380493572253727 + 0.j,
                              0.0504150338726772 + 0.j,
                              0.0593032803495607 + 0.j,
                              0.0669438933923514 + 0.j,
                              0.0739337523853319 + 0.j,
                              0.0805033453158931 + 0.j,
                              0.0079802568306227 + 0.j,
                              0.0120457790229829 + 0.j,
                              0.0148161421021887 + 0.j,
                              0.0169875044698748 + 0.j,
                              0.0188247358774655 + 0.j,
                              0.0035224345072349 + 0.j,
                              0.0055902910608630 + 0.j,
                              0.0070007963509055 + 0.j,
                              0.0080565705944120 + 0.j,
                              0.0018984262136024 + 0.j,
                              0.0030689226269545 + 0.j,
                              0.0038513663784351 + 0.j,
                              0.0010974040292900 + 0.j,
                              0.0017700790623805 + 0.j,
                              0.0006350955899765 + 0.j],
                             [-0.0805502046061130 + 0.j,
                              -0.0611127396392171 + 0.j,
                              -0.0640892811820629 + 0.j,
                              -0.0697709929145195 + 0.j,
                              -0.0761928007143594 + 0.j,
                              -0.0828392816369428 + 0.j,
                              -0.0895196906394810 + 0.j,
                              0.0414093333097644 + 0.j,
                              0.0548777958277241 + 0.j,
                              0.0645653775224049 + 0.j,
                              0.0728978585871946 + 0.j,
                              0.0805244351502411 + 0.j,
                              0.0876956887910217 + 0.j,
                              0.0086875935025343 + 0.j,
                              0.0131155090310228 + 0.j,
                              0.0161344201159119 + 0.j,
                              0.0185018723715049 + 0.j,
                              0.0205060708439888 + 0.j,
                              0.0038354916415492 + 0.j,
                              0.0060878144653612 + 0.j,
                              0.0076247166205728 + 0.j,
                              0.0087755824097035 + 0.j,
                              0.0020673955074163 + 0.j,
                              0.0033422558667758 + 0.j,
                              0.0041946064275996 + 0.j,
                              0.0011950411403562 + 0.j,
                              0.0019274878665384 + 0.j,
                              0.0006913896613698 + 0.j],
                             [-0.0879038682813978 + 0.j,
                              -0.0667080800401727 + 0.j,
                              -0.0699730349182325 + 0.j,
                              -0.0761928007143594 + 0.j,
                              -0.0832228884203967 + 0.j,
                              -0.0905006437743802 + 0.j,
                              -0.0978177104156161 + 0.j,
                              0.0452070446002034 + 0.j,
                              0.0599220813094387 + 0.j,
                              0.0705132985316171 + 0.j,
                              0.0796279867576509 + 0.j,
                              0.0879744736965231 + 0.j,
                              0.0958260519616467 + 0.j,
                              0.0094871413422558 + 0.j,
                              0.0143247308172882 + 0.j,
                              0.0176246438642299 + 0.j,
                              0.0202138107719987 + 0.j,
                              0.0224068123423515 + 0.j,
                              0.0041893882517588 + 0.j,
                              0.0066502602384193 + 0.j,
                              0.0083300744834015 + 0.j,
                              0.0095884672142172 + 0.j,
                              0.0022584221967322 + 0.j,
                              0.0036512802189782 + 0.j,
                              0.0045826772251442 + 0.j,
                              0.0013054316974975 + 0.j,
                              0.0021054626468748 + 0.j,
                              0.0007550406190823 + 0.j],
                             [-0.0955090389840754 + 0.j,
                              -0.0724963145713808 + 0.j,
                              -0.0760611581510978 + 0.j,
                              -0.0828392816369428 + 0.j,
                              -0.0905006437743802 + 0.j,
                              -0.0984337272436734 + 0.j,
                              -0.1064119434705389 + 0.j,
                              0.0491363849304351 + 0.j,
                              0.0651423541994335 + 0.j,
                              0.0766700738224950 + 0.j,
                              0.0865959280828922 + 0.j,
                              0.0956893872104286 + 0.j,
                              0.1042472167970307 + 0.j,
                              0.0103147081264282 + 0.j,
                              0.0155765596317315 + 0.j,
                              0.0191676597499300 + 0.j,
                              0.0219867194465590 + 0.j,
                              0.0243756043574970 + 0.j,
                              0.0045557939936183 + 0.j,
                              0.0072326704541114 + 0.j,
                              0.0090605735850667 + 0.j,
                              0.0104304446906017 + 0.j,
                              0.0024562390786220 + 0.j,
                              0.0039713166539076 + 0.j,
                              0.0049846092976104 + 0.j,
                              0.0014197505092668 + 0.j,
                              0.0022897685595554 + 0.j,
                              0.0008209402995558 + 0.j],
                             [-0.1031456871031516 + 0.j,
                              -0.0783104615368884 + 0.j,
                              -0.0821784870045671 + 0.j,
                              -0.0895196906394810 + 0.j,
                              -0.0978177104156160 + 0.j,
                              -0.1064119434705388 + 0.j,
                              -0.1150574703780492 + 0.j,
                              0.0530841730676872 + 0.j,
                              0.0703885797701098 + 0.j,
                              0.0828591379542015 + 0.j,
                              0.0936022826400370 + 0.j,
                              0.1034488646606583 + 0.j,
                              0.1127192007021945 + 0.j,
                              0.0111465358327448 + 0.j,
                              0.0168351205472870 + 0.j,
                              0.0207193263523689 + 0.j,
                              0.0237699704752715 + 0.j,
                              0.0263563247899062 + 0.j,
                              0.0049242160312684 + 0.j,
                              0.0078183881518368 + 0.j,
                              0.0097953491006253 + 0.j,
                              0.0112774977763678 + 0.j,
                              0.0026551893586058 + 0.j,
                              0.0042932196362405 + 0.j,
                              0.0053889243371543 + 0.j,
                              0.0015347279404732 + 0.j,
                              0.0024751324196980 + 0.j,
                              0.0008871979051059 + 0.j],
                             [0.0478568383864203 + 0.j,
                              0.0362946558453899 + 0.j,
                              0.0380493572253729 + 0.j,
                              0.0414093333097646 + 0.j,
                              0.0452070446002036 + 0.j,
                              0.0491363849304353 + 0.j,
                              0.0530841730676874 + 0.j,
                              -0.0245896077262418 + 0.j,
                              -0.0325787119037790 + 0.j,
                              -0.0383196917055469 + 0.j,
                              -0.0432537738587132 + 0.j,
                              -0.0477667877832336 + 0.j,
                              -0.0520077495879191 + 0.j,
                              -0.0051569376472646 + 0.j,
                              -0.0077837684268431 + 0.j,
                              -0.0095734521543978 + 0.j,
                              -0.0109759092324096 + 0.j,
                              -0.0121623297719945 + 0.j,
                              -0.0022761672502031 + 0.j,
                              -0.0036123061115051 + 0.j,
                              -0.0045236052861667 + 0.j,
                              -0.0052056302394887 + 0.j,
                              -0.0012267594356805 + 0.j,
                              -0.0019831309524021 + 0.j,
                              -0.0024887367357783 + 0.j,
                              -0.0007091921185375 + 0.j,
                              -0.0011439505861497 + 0.j,
                              -0.0004105047100228 + 0.j],
                             [0.0633827785124241 + 0.j,
                              0.0480801293766978 + 0.j,
                              0.0504150338726773 + 0.j,
                              0.0548777958277243 + 0.j,
                              0.0599220813094388 + 0.j,
                              0.0651423541994336 + 0.j,
                              0.0703885797701099 + 0.j,
                              -0.0325787119037789 + 0.j,
                              -0.0431710947588731 + 0.j,
                              -0.0507874726011044 + 0.j,
                              -0.0573366752009018 + 0.j,
                              -0.0633296145334488 + 0.j,
                              -0.0689635416033258 + 0.j,
                              -0.0068343851445359 + 0.j,
                              -0.0103171588116347 + 0.j,
                              -0.0126911520866933 + 0.j,
                              -0.0145524023549991 + 0.j,
                              -0.0161276826541549 + 0.j,
                              -0.0030172211717766 + 0.j,
                              -0.0047888872655546 + 0.j,
                              -0.0059976549435675 + 0.j,
                              -0.0069026585616367 + 0.j,
                              -0.0016263750218300 + 0.j,
                              -0.0026292915846609 + 0.j,
                              -0.0032998224462503 + 0.j,
                              -0.0009402171689085 + 0.j,
                              -0.0015165713341375 + 0.j,
                              -0.0005441079703957 + 0.j],
                             [0.0745259163770607 + 0.j,
                              0.0565452115431266 + 0.j,
                              0.0593032803495608 + 0.j,
                              0.0645653775224050 + 0.j,
                              0.0705132985316173 + 0.j,
                              0.0766700738224951 + 0.j,
                              0.0828591379542016 + 0.j,
                              -0.0383196917055468 + 0.j,
                              -0.0507874726011044 + 0.j,
                              -0.0597577654738798 + 0.j,
                              -0.0674750224174231 + 0.j,
                              -0.0745398719073566 + 0.j,
                              -0.0811841157083844 + 0.j,
                              -0.0080409915614317 + 0.j,
                              -0.0121403686992950 + 0.j,
                              -0.0149359962345797 + 0.j,
                              -0.0171288718393260 + 0.j,
                              -0.0189856868739057 + 0.j,
                              -0.0035506750471442 + 0.j,
                              -0.0056361815471105 + 0.j,
                              -0.0070595682080437 + 0.j,
                              -0.0081256669495500 + 0.j,
                              -0.0019141787176441 + 0.j,
                              -0.0030947583955419 + 0.j,
                              -0.0038842133504801 + 0.j,
                              -0.0011066095688561 + 0.j,
                              -0.0017849313027864 + 0.j,
                              -0.0006402627649506 + 0.j],
                             [0.0840931201605039 + 0.j,
                              0.0638177200315105 + 0.j,
                              0.0669438933923515 + 0.j,
                              0.0728978585871947 + 0.j,
                              0.0796279867576511 + 0.j,
                              0.0865959280828924 + 0.j,
                              0.0936022826400372 + 0.j,
                              -0.0432537738587131 + 0.j,
                              -0.0573366752009018 + 0.j,
                              -0.0674750224174230 + 0.j,
                              -0.0762014388911379 + 0.j,
                              -0.0841935419819257 + 0.j,
                              -0.0917127501271965 + 0.j,
                              -0.0090788459751881 + 0.j,
                              -0.0137092367988992 + 0.j,
                              -0.0168684763149219 + 0.j,
                              -0.0193477423708684 + 0.j,
                              -0.0214480172066346 + 0.j,
                              -0.0040098065252804 + 0.j,
                              -0.0063656567859586 + 0.j,
                              -0.0079741053378899 + 0.j,
                              -0.0091792705012122 + 0.j,
                              -0.0021619812807855 + 0.j,
                              -0.0034956033622817 + 0.j,
                              -0.0043875577124121 + 0.j,
                              -0.0012498813526957 + 0.j,
                              -0.0020159924385349 + 0.j,
                              -0.0007230068845453 + 0.j],
                             [0.0928361109948334 + 0.j,
                              0.0704673713830836 + 0.j,
                              0.0739337523853320 + 0.j,
                              0.0805244351502413 + 0.j,
                              0.0879744736965232 + 0.j,
                              0.0956893872104287 + 0.j,
                              0.1034488646606584 + 0.j,
                              -0.0477667877832334 + 0.j,
                              -0.0633296145334488 + 0.j,
                              -0.0745398719073566 + 0.j,
                              -0.0841935419819257 + 0.j,
                              -0.0930385680749462 + 0.j,
                              -0.1013633964849744 + 0.j,
                              -0.0100287970688324 + 0.j,
                              -0.0151457408869239 + 0.j,
                              -0.0186385502523309 + 0.j,
                              -0.0213808655430506 + 0.j,
                              -0.0237050202449841 + 0.j,
                              -0.0044302766233828 + 0.j,
                              -0.0070338881689814 + 0.j,
                              -0.0088120902029262 + 0.j,
                              -0.0101449441689803 + 0.j,
                              -0.0023889937934257 + 0.j,
                              -0.0038628759058110 + 0.j,
                              -0.0048488135739727 + 0.j,
                              -0.0013811372618551 + 0.j,
                              -0.0022276672932052 + 0.j,
                              -0.0007987714877135 + 0.j],
                             [0.1010454451455467 + 0.j,
                              0.0767142655027955 + 0.j,
                              0.0805033453158932 + 0.j,
                              0.0876956887910219 + 0.j,
                              0.0958260519616469 + 0.j,
                              0.1042472167970309 + 0.j,
                              0.1127192007021947 + 0.j,
                              -0.0520077495879190 + 0.j,
                              -0.0689635416033258 + 0.j,
                              -0.0811841157083844 + 0.j,
                              -0.0917127501271965 + 0.j,
                              -0.1013633964849744 + 0.j,
                              -0.1104498479460201 + 0.j,
                              -0.0109220524146663 + 0.j,
                              -0.0164969501381274 + 0.j,
                              -0.0203040666204849 + 0.j,
                              -0.0232945155349613 + 0.j,
                              -0.0258300805618901 + 0.j,
                              -0.0048258445032997 + 0.j,
                              -0.0076626991670747 + 0.j,
                              -0.0096008361985846 + 0.j,
                              -0.0110541015974222 + 0.j,
                              -0.0026026264096256 + 0.j,
                              -0.0042085508090221 + 0.j,
                              -0.0052830037727099 + 0.j,
                              -0.0015046604086598 + 0.j,
                              -0.0024268648080439 + 0.j,
                              -0.0008700379823503 + 0.j],
                             [0.0100307721513248 + 0.j,
                              0.0076098535772195 + 0.j,
                              0.0079802568306225 + 0.j,
                              0.0086875935025341 + 0.j,
                              0.0094871413422557 + 0.j,
                              0.0103147081264281 + 0.j,
                              0.0111465358327447 + 0.j,
                              -0.0051569376472645 + 0.j,
                              -0.0068343851445359 + 0.j,
                              -0.0080409915614316 + 0.j,
                              -0.0090788459751880 + 0.j,
                              -0.0100287970688323 + 0.j,
                              -0.0109220524146661 + 0.j,
                              -0.0010820888500130 + 0.j,
                              -0.0016336916745034 + 0.j,
                              -0.0020098092566090 + 0.j,
                              -0.0023047841926255 + 0.j,
                              -0.0025545105496307 + 0.j,
                              -0.0004778214850290 + 0.j,
                              -0.0007584626578703 + 0.j,
                              -0.0009499904411404 + 0.j,
                              -0.0010934276452102 + 0.j,
                              -0.0002576070346975 + 0.j,
                              -0.0004164935290353 + 0.j,
                              -0.0005227440490688 + 0.j,
                              -0.0001489428167668 + 0.j,
                              -0.0002402549461684 + 0.j,
                              -0.0000861964148710 + 0.j],
                             [0.0151358782333620 + 0.j,
                              0.0114848140629481 + 0.j,
                              0.0120457790229828 + 0.j,
                              0.0131155090310226 + 0.j,
                              0.0143247308172880 + 0.j,
                              0.0155765596317313 + 0.j,
                              0.0168351205472868 + 0.j,
                              -0.0077837684268430 + 0.j,
                              -0.0103171588116346 + 0.j,
                              -0.0121403686992949 + 0.j,
                              -0.0137092367988990 + 0.j,
                              -0.0151457408869238 + 0.j,
                              -0.0164969501381272 + 0.j,
                              -0.0016336916745034 + 0.j,
                              -0.0024667869775821 + 0.j,
                              -0.0030350785201450 + 0.j,
                              -0.0034809513036669 + 0.j,
                              -0.0038585754441044 + 0.j,
                              -0.0007215479252973 + 0.j,
                              -0.0011454555012199 + 0.j,
                              -0.0014348503746547 + 0.j,
                              -0.0016516564616475 + 0.j,
                              -0.0003890681283363 + 0.j,
                              -0.0006290811229198 + 0.j,
                              -0.0007896155046868 + 0.j,
                              -0.0002249669274892 + 0.j,
                              -0.0003628927530546 + 0.j,
                              -0.0001301821479848 + 0.j],
                             [0.0186106754313068 + 0.j,
                              0.0141238594629411 + 0.j,
                              0.0148161421021886 + 0.j,
                              0.0161344201159117 + 0.j,
                              0.0176246438642297 + 0.j,
                              0.0191676597499298 + 0.j,
                              0.0207193263523687 + 0.j,
                              -0.0095734521543977 + 0.j,
                              -0.0126911520866931 + 0.j,
                              -0.0149359962345795 + 0.j,
                              -0.0168684763149217 + 0.j,
                              -0.0186385502523307 + 0.j,
                              -0.0203040666204846 + 0.j,
                              -0.0020098092566090 + 0.j,
                              -0.0030350785201450 + 0.j,
                              -0.0037347476660277 + 0.j,
                              -0.0042839226801363 + 0.j,
                              -0.0047492190140155 + 0.j,
                              -0.0008878484665224 + 0.j,
                              -0.0014095989495987 + 0.j,
                              -0.0017659030573584 + 0.j,
                              -0.0020329282320441 + 0.j,
                              -0.0004788118523197 + 0.j,
                              -0.0007742401712282 + 0.j,
                              -0.0009718798519265 + 0.j,
                              -0.0002768770599085 + 0.j,
                              -0.0004466352998364 + 0.j,
                              -0.0001602070064119 + 0.j],
                             [0.0213309864526483 + 0.j,
                              0.0161911260447959 + 0.j,
                              0.0169875044698746 + 0.j,
                              0.0185018723715048 + 0.j,
                              0.0202138107719985 + 0.j,
                              0.0219867194465588 + 0.j,
                              0.0237699704752713 + 0.j,
                              -0.0109759092324094 + 0.j,
                              -0.0145524023549990 + 0.j,
                              -0.0171288718393259 + 0.j,
                              -0.0193477423708682 + 0.j,
                              -0.0213808655430504 + 0.j,
                              -0.0232945155349611 + 0.j,
                              -0.0023047841926256 + 0.j,
                              -0.0034809513036669 + 0.j,
                              -0.0042839226801363 + 0.j,
                              -0.0049144373497770 + 0.j,
                              -0.0054488583542370 + 0.j,
                              -0.0010183568456194 + 0.j,
                              -0.0016169602405916 + 0.j,
                              -0.0020258756040775 + 0.j,
                              -0.0023324347971913 + 0.j,
                              -0.0005492732437389 + 0.j,
                              -0.0008882353764349 + 0.j,
                              -0.0011150442437882 + 0.j,
                              -0.0003176411092681 + 0.j,
                              -0.0005123990519779 + 0.j,
                              -0.0001837766721926 + 0.j],
                             [0.0236300924351080 + 0.j,
                              0.0179393197803396 + 0.j,
                              0.0188247358774653 + 0.j,
                              0.0205060708439886 + 0.j,
                              0.0224068123423513 + 0.j,
                              0.0243756043574968 + 0.j,
                              0.0263563247899060 + 0.j,
                              -0.0121623297719944 + 0.j,
                              -0.0161276826541547 + 0.j,
                              -0.0189856868739055 + 0.j,
                              -0.0214480172066344 + 0.j,
                              -0.0237050202449838 + 0.j,
                              -0.0258300805618898 + 0.j,
                              -0.0025545105496307 + 0.j,
                              -0.0038585754441044 + 0.j,
                              -0.0047492190140155 + 0.j,
                              -0.0054488583542370 + 0.j,
                              -0.0060420985947139 + 0.j,
                              -0.0011289124125787 + 0.j,
                              -0.0017926732858756 + 0.j,
                              -0.0022462383823203 + 0.j,
                              -0.0025863865172608 + 0.j,
                              -0.0006089870988719 + 0.j,
                              -0.0009848620958035 + 0.j,
                              -0.0012364191938801 + 0.j,
                              -0.0003521920455659 + 0.j,
                              -0.0005681403170259 + 0.j,
                              -0.0002037461412343 + 0.j],
                             [0.0044254845218365 + 0.j,
                              0.0033581855204487 + 0.j,
                              0.0035224345072349 + 0.j,
                              0.0038354916415492 + 0.j,
                              0.0041893882517588 + 0.j,
                              0.0045557939936183 + 0.j,
                              0.0049242160312684 + 0.j,
                              -0.0022761672502030 + 0.j,
                              -0.0030172211717765 + 0.j,
                              -0.0035506750471441 + 0.j,
                              -0.0040098065252804 + 0.j,
                              -0.0044302766233828 + 0.j,
                              -0.0048258445032996 + 0.j,
                              -0.0004778214850290 + 0.j,
                              -0.0007215479252973 + 0.j,
                              -0.0008878484665224 + 0.j,
                              -0.0010183568456194 + 0.j,
                              -0.0011289124125787 + 0.j,
                              -0.0002110827207494 + 0.j,
                              -0.0003351244017925 + 0.j,
                              -0.0004198279099549 + 0.j,
                              -0.0004833016536958 + 0.j,
                              -0.0001138436123904 + 0.j,
                              -0.0001840902953333 + 0.j,
                              -0.0002310879292246 + 0.j,
                              -0.0000658413060907 + 0.j,
                              -0.0001062184195155 + 0.j,
                              -0.0000381092073714 + 0.j],
                             [0.0070218120829127 + 0.j,
                              0.0053290007599329 + 0.j,
                              0.0055902910608629 + 0.j,
                              0.0060878144653612 + 0.j,
                              0.0066502602384193 + 0.j,
                              0.0072326704541114 + 0.j,
                              0.0078183881518367 + 0.j,
                              -0.0036123061115051 + 0.j,
                              -0.0047888872655546 + 0.j,
                              -0.0056361815471104 + 0.j,
                              -0.0063656567859585 + 0.j,
                              -0.0070338881689813 + 0.j,
                              -0.0076626991670746 + 0.j,
                              -0.0007584626578703 + 0.j,
                              -0.0011454555012199 + 0.j,
                              -0.0014095989495987 + 0.j,
                              -0.0016169602405916 + 0.j,
                              -0.0017926732858757 + 0.j,
                              -0.0003351244017925 + 0.j,
                              -0.0005321088384111 + 0.j,
                              -0.0006666614741756 + 0.j,
                              -0.0007675214276221 + 0.j,
                              -0.0001807752507452 + 0.j,
                              -0.0002923455706983 + 0.j,
                              -0.0003670083285081 + 0.j,
                              -0.0001045660424699 + 0.j,
                              -0.0001687007953383 + 0.j,
                              -0.0000605278120603 + 0.j],
                             [0.0087913981858983 + 0.j,
                              0.0066727963434466 + 0.j,
                              0.0070007963509054 + 0.j,
                              0.0076247166205728 + 0.j,
                              0.0083300744834015 + 0.j,
                              0.0090605735850667 + 0.j,
                              0.0097953491006252 + 0.j,
                              -0.0045236052861667 + 0.j,
                              -0.0059976549435675 + 0.j,
                              -0.0070595682080436 + 0.j,
                              -0.0079741053378898 + 0.j,
                              -0.0088120902029261 + 0.j,
                              -0.0096008361985845 + 0.j,
                              -0.0009499904411404 + 0.j,
                              -0.0014348503746548 + 0.j,
                              -0.0017659030573584 + 0.j,
                              -0.0020258756040775 + 0.j,
                              -0.0022462383823203 + 0.j,
                              -0.0004198279099549 + 0.j,
                              -0.0006666614741756 + 0.j,
                              -0.0008353123926024 + 0.j,
                              -0.0009617709694017 + 0.j,
                              -0.0002265042646076 + 0.j,
                              -0.0003663260682315 + 0.j,
                              -0.0004599168552371 + 0.j,
                              -0.0001310344644699 + 0.j,
                              -0.0002114149700167 + 0.j,
                              -0.0000758540138051 + 0.j],
                             [0.0101147655457833 + 0.j,
                              0.0076782000447207 + 0.j,
                              0.0080565705944120 + 0.j,
                              0.0087755824097035 + 0.j,
                              0.0095884672142172 + 0.j,
                              0.0104304446906017 + 0.j,
                              0.0112774977763678 + 0.j,
                              -0.0052056302394886 + 0.j,
                              -0.0069026585616367 + 0.j,
                              -0.0081256669495499 + 0.j,
                              -0.0091792705012121 + 0.j,
                              -0.0101449441689802 + 0.j,
                              -0.0110541015974222 + 0.j,
                              -0.0010934276452103 + 0.j,
                              -0.0016516564616476 + 0.j,
                              -0.0020329282320441 + 0.j,
                              -0.0023324347971913 + 0.j,
                              -0.0025863865172609 + 0.j,
                              -0.0004833016536958 + 0.j,
                              -0.0007675214276221 + 0.j,
                              -0.0009617709694017 + 0.j,
                              -0.0011074680397378 + 0.j,
                              -0.0002607899089926 + 0.j,
                              -0.0004218076478657 + 0.j,
                              -0.0005296107343134 + 0.j,
                              -0.0001508871579564 + 0.j,
                              -0.0002434581563674 + 0.j,
                              -0.0000873510874605 + 0.j],
                             [0.0023845600555774 + 0.j,
                              0.0018096892986237 + 0.j,
                              0.0018984262136024 + 0.j,
                              0.0020673955074162 + 0.j,
                              0.0022584221967322 + 0.j,
                              0.0024562390786220 + 0.j,
                              0.0026551893586058 + 0.j,
                              -0.0012267594356805 + 0.j,
                              -0.0016263750218300 + 0.j,
                              -0.0019141787176440 + 0.j,
                              -0.0021619812807854 + 0.j,
                              -0.0023889937934256 + 0.j,
                              -0.0026026264096256 + 0.j,
                              -0.0002576070346975 + 0.j,
                              -0.0003890681283363 + 0.j,
                              -0.0004788118523197 + 0.j,
                              -0.0005492732437389 + 0.j,
                              -0.0006089870988719 + 0.j,
                              -0.0001138436123904 + 0.j,
                              -0.0001807752507451 + 0.j,
                              -0.0002265042646076 + 0.j,
                              -0.0002607899089926 + 0.j,
                              -0.0000614256118145 + 0.j,
                              -0.0000993474215613 + 0.j,
                              -0.0001247332113847 + 0.j,
                              -0.0000355422683851 + 0.j,
                              -0.0000573511578580 + 0.j,
                              -0.0000205833183553 + 0.j],
                             [0.0038543613277461 + 0.j,
                              0.0029253127397735 + 0.j,
                              0.0030689226269545 + 0.j,
                              0.0033422558667758 + 0.j,
                              0.0036512802189782 + 0.j,
                              0.0039713166539076 + 0.j,
                              0.0042932196362405 + 0.j,
                              -0.0019831309524021 + 0.j,
                              -0.0026292915846609 + 0.j,
                              -0.0030947583955418 + 0.j,
                              -0.0034956033622817 + 0.j,
                              -0.0038628759058110 + 0.j,
                              -0.0042085508090221 + 0.j,
                              -0.0004164935290354 + 0.j,
                              -0.0006290811229199 + 0.j,
                              -0.0007742401712282 + 0.j,
                              -0.0008882353764349 + 0.j,
                              -0.0009848620958035 + 0.j,
                              -0.0001840902953333 + 0.j,
                              -0.0002923455706983 + 0.j,
                              -0.0003663260682315 + 0.j,
                              -0.0004218076478657 + 0.j,
                              -0.0000993474215613 + 0.j,
                              -0.0001606958398292 + 0.j,
                              -0.0002017758795160 + 0.j,
                              -0.0000574980065047 + 0.j,
                              -0.0000927895458989 + 0.j,
                              -0.0000333080137258 + 0.j],
                             [0.0048365508368202 + 0.j,
                              0.0036709502967018 + 0.j,
                              0.0038513663784351 + 0.j,
                              0.0041946064275996 + 0.j,
                              0.0045826772251442 + 0.j,
                              0.0049846092976103 + 0.j,
                              0.0053889243371543 + 0.j,
                              -0.0024887367357783 + 0.j,
                              -0.0032998224462503 + 0.j,
                              -0.0038842133504801 + 0.j,
                              -0.0043875577124120 + 0.j,
                              -0.0048488135739727 + 0.j,
                              -0.0052830037727099 + 0.j,
                              -0.0005227440490688 + 0.j,
                              -0.0007896155046868 + 0.j,
                              -0.0009718798519265 + 0.j,
                              -0.0011150442437882 + 0.j,
                              -0.0012364191938802 + 0.j,
                              -0.0002310879292246 + 0.j,
                              -0.0003670083285080 + 0.j,
                              -0.0004599168552371 + 0.j,
                              -0.0005296107343134 + 0.j,
                              -0.0001247332113847 + 0.j,
                              -0.0002017758795160 + 0.j,
                              -0.0002533796646063 + 0.j,
                              -0.0000722062275436 + 0.j,
                              -0.0001165384801596 + 0.j,
                              -0.0000418403418188 + 0.j],
                             [0.0013785456136309 + 0.j,
                              0.0010461510220687 + 0.j,
                              0.0010974040292901 + 0.j,
                              0.0011950411403563 + 0.j,
                              0.0013054316974975 + 0.j,
                              0.0014197505092668 + 0.j,
                              0.0015347279404732 + 0.j,
                              -0.0007091921185375 + 0.j,
                              -0.0009402171689085 + 0.j,
                              -0.0011066095688561 + 0.j,
                              -0.0012498813526957 + 0.j,
                              -0.0013811372618550 + 0.j,
                              -0.0015046604086598 + 0.j,
                              -0.0001489428167668 + 0.j,
                              -0.0002249669274892 + 0.j,
                              -0.0002768770599085 + 0.j,
                              -0.0003176411092681 + 0.j,
                              -0.0003521920455659 + 0.j,
                              -0.0000658413060907 + 0.j,
                              -0.0001045660424699 + 0.j,
                              -0.0001310344644699 + 0.j,
                              -0.0001508871579564 + 0.j,
                              -0.0000355422683851 + 0.j,
                              -0.0000574980065047 + 0.j,
                              -0.0000722062275436 + 0.j,
                              -0.0000205802660191 + 0.j,
                              -0.0000332207592566 + 0.j,
                              -0.0000119319467562 + 0.j],
                             [0.0022237693216097 + 0.j,
                              0.0016874859899826 + 0.j,
                              0.0017700790623805 + 0.j,
                              0.0019274878665384 + 0.j,
                              0.0021054626468749 + 0.j,
                              0.0022897685595555 + 0.j,
                              0.0024751324196980 + 0.j,
                              -0.0011439505861497 + 0.j,
                              -0.0015165713341375 + 0.j,
                              -0.0017849313027864 + 0.j,
                              -0.0020159924385349 + 0.j,
                              -0.0022276672932052 + 0.j,
                              -0.0024268648080439 + 0.j,
                              -0.0002402549461685 + 0.j,
                              -0.0003628927530546 + 0.j,
                              -0.0004466352998365 + 0.j,
                              -0.0005123990519779 + 0.j,
                              -0.0005681403170259 + 0.j,
                              -0.0001062184195155 + 0.j,
                              -0.0001687007953383 + 0.j,
                              -0.0002114149700167 + 0.j,
                              -0.0002434581563674 + 0.j,
                              -0.0000573511578580 + 0.j,
                              -0.0000927895458989 + 0.j,
                              -0.0001165384801596 + 0.j,
                              -0.0000332207592566 + 0.j,
                              -0.0000536359914926 + 0.j,
                              -0.0000192728640005 + 0.j],
                             [0.0007983521363725 + 0.j,
                              0.0006056338822255 + 0.j,
                              0.0006350955899765 + 0.j,
                              0.0006913896613698 + 0.j,
                              0.0007550406190823 + 0.j,
                              0.0008209402995558 + 0.j,
                              0.0008871979051059 + 0.j,
                              -0.0004105047100228 + 0.j,
                              -0.0005441079703957 + 0.j,
                              -0.0006402627649506 + 0.j,
                              -0.0007230068845453 + 0.j,
                              -0.0007987714877135 + 0.j,
                              -0.0008700379823503 + 0.j,
                              -0.0000861964148710 + 0.j,
                              -0.0001301821479848 + 0.j,
                              -0.0001602070064120 + 0.j,
                              -0.0001837766721926 + 0.j,
                              -0.0002037461412343 + 0.j,
                              -0.0000381092073714 + 0.j,
                              -0.0000605278120603 + 0.j,
                              -0.0000758540138051 + 0.j,
                              -0.0000873510874605 + 0.j,
                              -0.0000205833183553 + 0.j,
                              -0.0000333080137258 + 0.j,
                              -0.0000418403418188 + 0.j,
                              -0.0000119319467562 + 0.j,
                              -0.0000192728640005 + 0.j,
                              -0.0000069327159862 + 0.j]],
                            dtype=numpy.complex128)
        work = fqe_data.FqeData(2, 2, norb)
        work.coeff = numpy.copy(wfn)
        test = work.apply(tuple([h1e_spa, h2e_spa]))
        self.assertTrue(numpy.allclose(numpy.multiply(wfn, scale), test.coeff))
        test = work.apply(tuple([h1e_spin, h2e_spin]))
        self.assertTrue(numpy.allclose(numpy.multiply(wfn, scale), test.coeff))
        low_fill = work.rdm12(work)
        half_fill = work._rdm12_halffilling(work)
        self.assertTrue(numpy.allclose(low_fill[0], half_fill[0]))
        self.assertTrue(numpy.allclose(low_fill[1], half_fill[1]))


    def test_s2_inplace(self):
        """Check application of S^2 operator
        """
        work = fqe_data.FqeData(2, 1, 3)
        work.set_wfn(strategy='ones')
        work.apply_inplace_s2()
        self.assertTrue(numpy.allclose(work.coeff,
                                       numpy.asarray([[0.75+0.j, 0.75+0.j, 1.75+0.j],
                                                      [0.75+0.j, -0.25+0.j, 0.75+0.j],
                                                      [1.75+0.j, 0.75+0.j, 0.75+0.j]],
                                                     dtype=numpy.complex128)))


    def test_data_print(self):
        """Check that data is printed correctly
        """
        numpy.random.seed(seed=409)
        work = fqe_data.FqeData(2, 1, 3)
        work.set_wfn(strategy='random')
        work.print_sector()
        ref_string = 'Sector N = 3 : S_z = 1\n' + \
        '11:1 (0.10674272124334523+0.12411053427712923j)\n' + \
        '11:10 (-0.07909947552516094-0.1711854580294638j)\n' + \
        '11:100 (-0.2277638468354617+0.3098449753503317j)\n' + \
        '101:1 (-0.2929155787304851+0.1907728639293026j)\n' + \
        '101:10 (-0.04075713791110513-0.10838093068708356j)\n' + \
        '101:100 (-0.4879989468488005+0.1699026592505128j)\n' + \
        '110:1 (-0.40736015315949137+0.12671170294093023j)\n' + \
        '110:10 (0.437996918272229+0.010753267360406955j)\n' + \
        '110:100 (0.11458915833820198-0.008003969193832962j)\n'
        save_stdout = sys.stdout
        sys.stdout = chkprint = StringIO()
        work.print_sector()
        sys.stdout = save_stdout
        outstring = chkprint.getvalue()
        self.assertEqual(outstring, ref_string)