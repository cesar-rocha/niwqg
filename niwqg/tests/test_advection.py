import numpy as np
from numpy.random import randn
import unittest
from niwqg import CoupledModel
from niwqg import QGModel

class QGNIWTester(unittest.TestCase):
    """ A class for testing the QGNIW kernel (real and complex 2d ffts)
        Note: 1d plane wave pass test with machine precision
              2d (slanted) plane wave has an error O(10^{-13})
                                                                        """
    def setUp(self):
        self.m =  CoupledModel.Model(use_filter=False)
        k, l  = 2*np.pi*5/self.m.L, 2*np.pi*9/self.m.L
        self.m.set_q(np.sin(k*self.m.x + l*self.m.y))
        self.m.set_phi(np.sin(k*self.m.x + l*self.m.y))

    def test_jacobian(self, rtol=1e-12):
        """ Jacobian must be zero for plane waves """

        jacq = self.m.jacobian_psi_q()
        errorq = jacq.std()
        self.assertTrue(errorq<rtol, 'Implementation of advection J(\psi,q) is broken')

        jacphi2 = self.m.jacobian_phic_phi()
        errorphi2 = jacphi2.std()
        self.assertTrue(errorphi2<rtol, 'Implementation of advection J(\phic,\phi)\
                                        is broken')

        jacphi = self.m.jacobian_psi_phi()
        errorphi = jacphi.std()
        self.assertTrue(errorphi<rtol, 'Implementation of advection J(\psi,\phi) is broken')

class QGTester(unittest.TestCase):
    """ A class for testing the QG model (rffts)
            Note: 1d plane wave pass test with machine precision
                  2d (slanted) plane wave has an error O(10^{-13})
                                                                        """
    def setUp(self):
        self.m =  QGModel.Model(use_filter=False)
        k, l  = 2*np.pi*5/self.m.L, 2*np.pi*9/self.m.L
        self.m.set_q(np.sin(k*self.m.x + l*self.m.x))

    def test_jacobian(self, rtol=1e-12):
        """ Jacobian must be zero for plane waves """

        jac = self.m.jacobian_psi_q()
        error = jac.std()

        self.assertTrue(error<rtol, 'Implementation of advection J(\psi,q) is broken')

if __name__ == "__main__":
    unittest.main()
