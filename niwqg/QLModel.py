import numpy as np
from . import Kernel

class Model(Kernel.Kernel):
    """ A subclass that represents the YBJ-QG coupled model """

    def __init__(
        self,
        **kwargs
        ):

        super(Model, self).__init__(**kwargs)

    def _allocate_variables(self):
        """ Allocate variables in memory """

        self.dtype_real = np.dtype('float64')
        self.dtype_cplx = np.dtype('complex128')
        self.shape_real = (self.ny, self.nx)
        self.shape_cplx = (self.ny, self.nx)

        # vorticity
        self.q  = np.zeros(self.shape_real,  self.dtype_real)
        self.qh = np.zeros(self.shape_cplx,  self.dtype_cplx)
        self.qh0 = np.zeros(self.shape_cplx, self.dtype_cplx)
        self.qh1 = np.zeros(self.shape_cplx, self.dtype_cplx)

        # stream function
        self.p  = np.zeros(self.shape_real,  self.dtype_real)
        self.ph = np.zeros(self.shape_cplx,  self.dtype_cplx)

        # wave amplitude
        self.phi = np.zeros(self.shape_real,  self.dtype_cplx)
        self.phih = np.zeros(self.shape_cplx,  self.dtype_cplx)

    def jacobian_psi_phi(self):
        """ Compute the Jacobian phix and phiy. """
        self.ph_q = -self.wv2i*self.qh
        self.uq, self.vq = self.ifft(-self.il*self.ph_q).real, self.ifft(self.ik*self.ph_q).real
        return self.fft( (self.uq*self.phix + self.vq*self.phiy) )

    def _invert(self):
        """ From qh compute ph and compute velocity. """

        # the wavy PV
        self.phich = self.fft(np.conj(self.phi))
        self.phi2 = np.abs(self.phi)**2
        self.jacph = self.jacobian_phic_phi()
        self.gphi2h = -self.wv2*self.fft(self.phi2)
        self.qwh = (0.5*self.gphi2h  + 1j*self.jacph)/self.f/2.
        self.qwh *= self.filtr
        # invert for psi
        self.ph = -self.wv2i*(self.qh-self.qwh)
        # calculate in physical space
        self.p = self.ifft(self.ph).real


    def _initialize_class_diagnostics(self):
        pass

    def _calc_class_derived_fields(self):
        pass
