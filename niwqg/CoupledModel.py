import numpy as np
from . import Kernel
from .Diagnostics import *

class Model(Kernel.Kernel):

    """ A subclass that represents the Xie & Vanneste coupled model
        of single-vertical wavenumber near-inertial waves and barotropic
        quasigeostrophic flow.

        It defines the quasigeostrophic inversion relation–––with wave effects
        –––and the diagnostics specific to this subclass.

        Reference
        ----------

        "A generalised-Lagrangian-mean model of the
            interactions between near-inertial waves
            and mean flow," Journal of Fluid Mechanics,
            (2015), vol. 774, pp. 143–169. doi:10.1017/jfm.2015.251

    """

    def __init__(
        self,
        **kwargs
        ):

        self.model = " Coupled Model"

        super(Model, self).__init__(**kwargs)

    def _allocate_variables(self):

        """ Allocate variables so that variable addresses are close in memory.
        """

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

        """ Compute the Jacobian between conj(phi) and phi–––a wave potential
            vorticity term.

        Returns
        -------
        complex array of floats
            The Fourier transform of Jacobian(conj(phi),phi)
        """

        self.phix, self.phiy = self.ifft(self.ik*self.phih), self.ifft(self.il*self.phih)
        jach = self.fft((1j*(np.conj(self.phix)*self.phiy - np.conj(self.phiy)*self.phix)).real)
        jach[0,0] = 0
        return jach

    def _invert(self):

        """ Calculate the streamfunction given the potential vorticity.
            The algorithm is:
                1) Calculate wave potential vorticity
                2) Invert for wave, pw, and vortex stremfunctions, pv.
                3) Calculate geostrophic stremfunction, p = pv+pw.
        """

        # the wave PV
        self.phi2 = np.abs(self.phi)**2
        self.gphi2h = -self.wv2*self.fft(self.phi2)
        self.qwh = 0.5*(0.5*self.gphi2h  + self.jacobian_phic_phi())/self.f
        self.qwh *= self.filtr

        # invert for psi
        self.pwh = self.wv2i*self.qwh
        self.pvh = -self.wv2i*self.qh
        self.ph = self.pvh+self.pwh
        self.p =  self.ifft(self.ph).real
        self.pv = self.ifft(self.pvh).real
        self.pw = self.ifft(self.pwh).real

        # calcuate q
        self.q = self.ifft(self.qh).real

    def _calc_ke_qg_decomp(self):

        """ Compute vortex, wave, and cross terms of the geostrophic kinetic
            energy.
        """

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

        """ Compute subclass-specific derived fields.
        """

        self._calc_ke_qg_decomp()

    def _calc_rel_vorticity(self):

        """  Compute the geostrophic relative vorticity–––the Laplacian of the
                streamfuctions.
        """

        self.qw = self.ifft(self.qwh).real
        self.q_psi = (self.q-self.qw)
