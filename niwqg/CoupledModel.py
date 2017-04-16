import numpy as np
from . import Kernel
from .Diagnostics import *

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



    def jacobian_phic_phi(self):
        """ Compute the Jacobian phix and phiy. """
        self.phix, self.phiy = self.ifft(self.ik*self.phih), self.ifft(self.il*self.phih)
        jach = self.fft((1j*(np.conj(self.phix)*self.phiy - np.conj(self.phiy)*self.phix)).real)
        jach[0,0] = 0
        return jach
        #phic = np.conj(self.phi)
        #return self.ik*self.fft(phic*self.phiy) - self.il*self.fft(phic*self.phix)

    def jacobian_psi_phi(self):
        """ Compute the Jacobian phix and phiy. """
        jach = self.fft( (self.u*self.phix + self.v*self.phiy) )
        jach[0,0] = 0
        return jach

    def jacobian_psi_q(self):
        """ Compute the Jacobian between psi and q. Return in Fourier space. """
        self.u, self.v = self.ifft(-self.il*self.ph).real, self.ifft(self.ik*self.ph).real
        q = self.ifft(self.qh).real
        jach = self.ik*self.fft(self.u*q) + self.il*self.fft(self.v*q)
        jach[0,0] = 0
        #jach[0],jach[:,0] = 0, 0
        return jach

    def _invert(self):
        """ From qh compute ph and compute velocity. """

        # the wavy PV
        self.phi2 = np.abs(self.phi)**2
        self.gphi2h = -self.wv2*self.fft(self.phi2)
        self.qwh = 0.5*(0.5*self.gphi2h  + self.jacobian_phic_phi())/self.f
        self.qwh *= self.filtr

        # invert for psi
        self.pw = self.ifft((self.wv2i*self.qwh)).real
        self.pv = self.ifft(-(self.wv2i*self.qh)).real
        self.p = self.pv+self.pw
        self.ph = self.fft(self.p)

        # calcuate q
        self.q = self.ifft(self.qh).real

    def _calc_ke_qg_decomp(self):
        self.phq = -self.wv2i*self.qh
        self.ke_qg_q = 0.5*self.spec_var(self.wv*self.phq)

        self.phw = self.wv2i*self.qwh
        self.ke_qg_w = 0.5*self.spec_var(self.wv*self.phw)

        self.uq, self.vq = self.ifft(-self.il*self.phq).real, self.ifft(self.ik*self.phq).real
        self.uw, self.vw = self.ifft(-self.il*self.phw).real, self.ifft(self.ik*self.phw).real
        self.ke_qg_qw = (self.uq*self.uw).mean() + (self.vq*self.vw).mean()

    def _initialize_class_diagnostics(self):

        add_diagnostic(self, 'ke_qg_q',
                description='Quasigeostrophic Kinetic Energy, q-flow',
                units=r'm^2 s^{-2}',
                types = 'scalar',
                function = (lambda self: self.ke_qg_q)
        )

        add_diagnostic(self, 'ke_qg_w',
                description='Quasigeostrophic Kinetic Energy, w-flow',
                units=r'm^2 s^{-2}',
                types = 'scalar',
                function = (lambda self: self.ke_qg_w)
        )

        add_diagnostic(self, 'ke_qg_qw',
                description='Quasigeostrophic Kinetic Energy, cross-term q-w',
                units=r'm^2 s^{-2}',
                types = 'scalar',
                function = (lambda self: self.ke_qg_qw)
        )

    def _calc_class_derived_fields(self):
        self._calc_ke_qg_decomp()
        #self._invert()
        #self._calc_rel_vorticity()

    def _calc_rel_vorticity(self):
        """ from psi compute relative vorticity """
        #self._invert()
        #self.qh_psi = -self.wv2*self.ph
        self.qw = self.ifft(self.qwh).real
        self.q_psi = (self.q-self.qw)

        #self.qh_psi = self.qh-self.qwh
        #self.q_psi = self.ifft(self.qh_psi).real

