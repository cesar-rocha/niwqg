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


    def _step_etdrk4(self):
        """ march the system forward using a ETDRK4 scheme """

        # q-equation
        self.qh0 = self.qh.copy()
        Fn0 = -self.jacobian_psi_q()
        self.qh = (self.expch_h*self.qh0 + Fn0*self.Qh)*self.filtr
        self.qh1 = self.qh.copy()

        # phi-equation
        self.phih0 = self.phih.copy()
        #self._calc_rel_vorticity()
        self.q = self.ifft(self.qh).real
        Fn0w = -self.jacobian_psi_phi() - 0.5j*self.fft(self.phi*self.q)
        self.phih = (self.expch_hw*self.phih0 + Fn0w*self.Qhw)*self.filtr
        self.phih1 = self.phih.copy()

        # q-equation
        self.phi = self.ifft(self.phih)
        self._invert()
        Fna = -self.jacobian_psi_q()
        self.qh = (self.expch_h*self.qh0 + Fna*self.Qh)*self.filtr

        # phi-equation
        #self._calc_rel_vorticity()
        self.q = self.ifft(self.qh).real
        Fnaw = -self.jacobian_psi_phi() - 0.5j*self.fft(self.phi*self.q)
        self.phih = (self.expch_hw*self.phih0 + Fnaw*self.Qhw)*self.filtr

        # q-equation
        self.phi = self.ifft(self.phih)
        self._invert()
        Fnb = -self.jacobian_psi_q()
        self.qh = (self.expch_h*self.qh1 + ( 2.*Fnb - Fn0 )*self.Qh)*self.filtr

        # phi-equation
        #self._calc_rel_vorticity()
        self.q = self.ifft(self.qh).real
        Fnbw = -self.jacobian_psi_phi() - 0.5j*self.fft(self.phi*self.q)
        self.phih = (self.expch_hw*self.phih1 + ( 2.*Fnbw - Fn0w )*self.Qhw)*self.filtr

        # q-equation
        self.phi = self.ifft(self.phih)
        self._invert()
        Fnc = -self.jacobian_psi_q()
        self.qh = (self.expch*self.qh0 + Fn0*self.f0 +  2.*(Fna+Fnb)*self.fab\
                  + Fnc*self.fc)*self.filtr

        # phi-equation
        #self._calc_rel_vorticity()
        self.q = self.ifft(self.qh).real
        Fncw = -self.jacobian_psi_phi() - 0.5j*self.fft(self.phi*self.q)
        self.phih = (self.expchw*self.phih0 + Fn0w*self.f0w +  2.*(Fnaw+Fnbw)*self.fabw\
                  + Fncw*self.fcw)*self.filtr

        # invert
        self.phi = self.ifft(self.phih)
        self._invert()

        # calcuate q
        self.q = self.ifft(self.qh).real

    def _init_etdrk4(self):

        """ This performs pre-computations for the Expotential Time Differencing
            Fourth Order Runge Kutta time stepper. The linear part is calculated
            exactly.
            See Cox and Matthews, J. Comp. Physics., 176(2):430-455, 2002.
                Kassam and Trefethen, IAM J. Sci. Comput., 26(4):1214-233, 2005. """


        #
        # coefficients for q-equation
        #

        # the exponent for the linear part
        self.c = np.zeros((self.nl,self.nk),self.dtype_cplx)
        self.c += -self.nu4*self.wv4 - 1j*self.k*self.U
        ch = self.c*self.dt
        self.expch = np.exp(ch)
        self.expch_h = np.exp(ch/2.)
        self.expch2 = np.exp(2.*ch)

        M = 32.  # number of points for line integral in the complex plane
        rho = 1.  # radius for complex integration
        r = rho*np.exp(2j*np.pi*((np.arange(1.,M+1))/M)) # roots for integral
        LR = ch[...,np.newaxis] + r[np.newaxis,np.newaxis,...]
        LR2 = LR*LR
        LR3 = LR2*LR
        self.Qh   =  self.dt*(((np.exp(LR/2.)-1.)/LR).mean(axis=-1))
        self.f0  =  self.dt*( ( ( -4. - LR + ( np.exp(LR)*( 4. - 3.*LR + LR2 ) ) )/ LR3 ).mean(axis=-1) )
        self.fab =  self.dt*( ( ( 2. + LR + np.exp(LR)*( -2. + LR ) )/ LR3 ).mean(axis=-1) )
        self.fc  =  self.dt*( ( ( -4. -3.*LR - LR2 + np.exp(LR)*(4.-LR) )/ LR3 ).mean(axis=-1) )

        #
        # coefficients for phi-equation
        #

        # the exponent for the linear part
        self.c = np.zeros((self.nl,self.nk),self.dtype_cplx)  -1j*self.k*self.U
        self.c += -self.nu4w*self.wv4 - 0.5j*self.f*(self.wv2/self.kappa2)
        ch = self.c*self.dt
        self.expchw = np.exp(ch)
        self.expch_hw = np.exp(ch/2.)
        self.expch2w = np.exp(2.*ch)

        LR = ch[...,np.newaxis] + r[np.newaxis,np.newaxis,...]
        LR2 = LR*LR
        LR3 = LR2*LR
        self.Qhw   =  self.dt*(((np.exp(LR/2.)-1.)/LR).mean(axis=-1))
        self.f0w  =  self.dt*( ( ( -4. - LR + ( np.exp(LR)*( 4. - 3.*LR + LR2 ) ) )/ LR3 ).mean(axis=-1) )
        self.fabw =  self.dt*( ( ( 2. + LR + np.exp(LR)*( -2. + LR ) )/ LR3 ).mean(axis=-1) )
        self.fcw  =  self.dt*( ( ( -4. -3.*LR - LR2 + np.exp(LR)*(4.-LR) )/ LR3 ).mean(axis=-1) )

    def jacobian_phic_phi(self):
        """ Compute the Jacobian phix and phiy. """
        self.phix, self.phiy = self.ifft(self.ik*self.phih), self.ifft(self.il*self.phih)
        return self.fft(np.conj(self.phix)*self.phiy - np.conj(self.phiy)*self.phix)

    def jacobian_psi_phi(self):
        """ Compute the Jacobian phix and phiy. """
        self.ph_q = -self.wv2i*self.qh
        self.uq, self.vq = self.ifft(-self.il*self.ph_q).real, self.ifft(self.ik*self.ph_q).real
        return self.fft( (self.uq*self.phix + self.vq*self.phiy) )


    def jacobian_psi_q(self):
        """ Compute the Jacobian between psi and q. Return in Fourier space. """
        self.u, self.v = self.ifft(-self.il*self.ph).real, self.ifft(self.ik*self.ph).real
        q = self.ifft(self.qh).real
        return self.ik*self.fft(self.u*q) + self.il*self.fft(self.v*q)

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
