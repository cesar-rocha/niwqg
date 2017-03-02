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
        self.m =  CoupledModel.Model(use_filter=False,nu4=1e14,nu4w=0.)
        self.m.tmax = 10*self.m.dt
        k, l  = 2*np.pi*5/self.m.L, 2*np.pi*9/self.m.L
        self.qi = np.sin(k*self.m.x + l*self.m.y)
        self.m.set_q(self.qi)
        self.m.set_phi(self.qi*0)

    def test_hyperviscosity(self, rtol=1e-15):
        """ Test if the hyperviscosity implementation simply damps
                the Fourier coefficiants individualy. """

        self.m.run()
        qfh = self.m.fft(self.qi)*np.exp(-self.m.nu4*self.m.wv4*self.m.tmax)
        self.assertTrue(np.allclose(qfh,self.m.qh,rtol=rtol), 'Implementation of\
                             hypterviscosity is broken in CoupledModel')

class QGTester(unittest.TestCase):
    """ A class for testing the QG model (rffts)
            Note: 1d plane wave pass test with machine precision
                  2d (slanted) plane wave has an error O(10^{-13})
                                                                        """
    def setUp(self):
        self.m =  QGModel.Model(use_filter=False, nu4 = 1e14)
        self.m.tmax = 100*self.m.dt
        k, l  = 2*np.pi*5/self.m.L, 2*np.pi*9/self.m.L
        self.qi = np.sin(k*self.m.x + l*self.m.x)
        self.m.set_q(self.qi)

    def test_hyperviscosity(self, rtol=1e-15):
        """ Test if the hyperviscosity implementation simply damps
                the Fourier coefficiants individualy. """

        self.m.run()
        qfh = self.m.fft(self.qi)*np.exp(-self.m.nu4*self.m.wv4*self.m.tmax)
        self.assertTrue(np.allclose(qfh,self.m.qh,rtol=rtol), 'Implementation of\
                             hypterviscosity is broken QGModel')

if __name__ == "__main__":
    unittest.main()
