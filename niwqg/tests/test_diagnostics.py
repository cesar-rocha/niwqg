import numpy as np
from numpy.random import randn
import unittest
from niwqg import CoupledModel
from niwqg import QGModel
from niwqg import InitialConditions as ic

class QGNIWTester(unittest.TestCase):
    """ A class for diagnostics of the QGNIW model """
    def setUp(self):
        U0 = 0.05
        self.m =  CoupledModel.Model(use_filter=False, U=-U0, tdiags=1)

        k0 = 10*(2*np.pi/self.m.L)
        q = ic.LambDipole(self.m, U=U0,R = 2*np.pi/k0)
        phi = (np.ones_like(q) + 1j)*5*U0/np.sqrt(2)

        self.m.set_q(q)
        self.m.set_phi(phi)

        self.m.run()

    def test_energy(self, rtol=1e-15):
        """ Diagnosed QG kientic energy must be the same as Ke from energy equation """

        KE_qg = self.m.diagnostics['ke_qg']['value']
        Ke = self.m.diagnostics['Ke']['value']
        self.assertTrue(np.allclose(KE_qg,Ke,rtol=rtol), "KE QG diagnostic is incorrect")

        KE_niw = self.m.diagnostics['ke_niw']['value']
        Kw = self.m.diagnostics['Kw']['value']
        self.assertTrue(np.allclose(KE_niw,Kw,rtol=rtol), "KE NIW diagnostic is incorrect")

        PE_niw = self.m.diagnostics['pe_niw']['value']
        Pw = self.m.diagnostics['Pw']['value']
        self.assertTrue(np.allclose(PE_niw,Pw,rtol=rtol), "PE NIW diagnostic is incorrect")

class QGTester(unittest.TestCase):
    """ A class for diagnostics of the QG model """
    def setUp(self):
        U0 = 0.05
        self.m =  QGModel.Model(use_filter=False, U=-U0, tdiags=1, passive_scalar=True)

        k0 = 10*(2*np.pi/self.m.L)
        q = ic.LambDipole(self.m, U=U0,R = 2*np.pi/k0)
        c = ic.PlaneWave(self.m,k=k0,l=k0)*q.mean()

        self.m.set_q(q)
        self.m.set_c(c)

        self.m.run()

    def test_energy(self, rtol=1e-14):
        """ Diagnosed QG kientic energy must be the same as Ke from energy equation """

        KE_qg = self.m.diagnostics['ke_qg']['value']
        Ke = self.m.diagnostics['Ke']['value']

        self.assertTrue(np.allclose(KE_qg,Ke,rtol=rtol), "KE QG diagnostic is incorrect")

    def test_tracer_variance(self, rtol=1e-14):
        """ Diagnosed passive-scalar variance must be the same cvar from variance equation """
        C2 = self.m.diagnostics['C2']['value']
        cvar = self.m.diagnostics['cvar']['value']
        self.assertTrue(np.allclose(C2,cvar,rtol=rtol), "Passive-scalar variance diagnostic is incorrect")

if __name__ == "__main__":
    unittest.main()
