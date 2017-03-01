import numpy as np
from numpy.random import randn
import unittest
from niwqg import CoupledModel
from niwqg import QGModel

def relative_error(var1,var2):
    diffvar = np.abs(var1-var2)
    return max(diffvar/var1,diffvar/var2).real

class QGNIWTester(unittest.TestCase):
    """ A class for testing the QGNIW kernel (real and complex 2d ffts) """
    def setUp(self):
        self.m =  CoupledModel.Model(use_filter=False)
        self.qi = randn(self.m.ny, self.m.nx)
        self.phii = randn(self.m.ny,self.m.nx)+ 1j*randn(self.m.ny,self.m.nx)

    def test_forward_backward(self, rtol=1e-15):
        """ Compares variable with its ifft(fft)"""

        qn = self.m.ifft(self.m.fft(self.qi)).real
        phin = self.m.ifft(self.m.fft(self.phii))
        self.assertTrue(np.allclose(qn,self.qi,rtol=rtol), "FFT is broken")
        self.assertTrue(np.allclose(phin,self.phii,rtol=rtol), "FFT is broken")

    def test_parseval(self, rtol=1.e-15):
        """ Compares variance calculated in physical and spectral space """

        # real field
        self.m.set_q(self.qi)
        var_q_spec, var_q_phys = self.m.spec_var(self.m.qh), self.qi.var()
        error_var_q = relative_error(var_q_phys,var_q_spec)
        print("relative error, var q = %5.16f" %error_var_q)
        self.assertTrue(error_var_q<rtol,"QGNIW Kernel fft does not satisfy Parseval's relation")

        # complex field
        self.m.set_phi(self.phii)
        var_phi_spec, var_phi_phys = self.m.spec_var(self.m.phih), self.phii.var()
        error_var_phi = relative_error(var_phi_phys,var_phi_spec)
        print("relative error, var phi = %5.16f" %error_var_phi)
        self.assertTrue(error_var_phi<rtol,"QGNIW Kernel fft does not satisfy Parseval's relation")

class QGTester(unittest.TestCase):
    """ A class for testing the QG model (rffts) """
    def setUp(self):
        self.m =  QGModel.Model(use_filter=False)
        self.qi = randn(self.m.ny, self.m.nx)

    def test_forward_backward(self, rtol=1e-15):
        """ Compares variable with its ifft(fft)"""

        qn = self.m.ifft(self.m.fft(self.qi))
        self.assertTrue(np.allclose(qn,self.qi,rtol=rtol), "RFFT is broken")

    def test_parseval(self, rtol=1.e-15):
        """ Compares variance calculated in physical and spectral space """

        self.m.set_q(self.qi)
        var_q_spec, var_q_phys = self.m.spec_var(self.m.qh), self.qi.var()
        error_var_q = relative_error(var_q_phys,var_q_spec)
        print("relative error, var q = %5.16f" %error_var_q)
        self.assertTrue(error_var_q<rtol,"QG Model fft does not satisfy Parseval's relation")

if __name__ == "__main__":
    unittest.main()
