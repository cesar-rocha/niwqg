import numpy as np
from . import Kernel

class Model(Kernel.Kernel):

    """ A subclass that represents the Young & Ben Jelloul uncoupled model
        of single-vertical wavenumber near-inertial waves and STEADY
        barotropic quasigeostrophic flow.

        It defines the quasigeostrophic inversion relation and the diagnostics
        specific to this subclass.

        Reference
        ----------
        Young, W. R. & Ben Jelloul, M. 1997 "Propagation of near-inertial
        oscillations through a geostrophic flow." J. Mar. Res. 55 (4), 735–766.

    """

    def __init__(
        self,
        **kwargs
        ):

        self.model = " YBJ Model (Steady QG flow)"

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

        # stream function
        self.p  = np.zeros(self.shape_real,  self.dtype_real)
        self.ph = np.zeros(self.shape_cplx,  self.dtype_cplx)

        # wave amplitude
        self.phi = np.zeros(self.shape_real,  self.dtype_cplx)
        self.phih = np.zeros(self.shape_cplx,  self.dtype_cplx)


    def _step_etdrk4(self):
        """ march the system forward using a ETDRK4 scheme """

        # phi-equation
        self.phih0 = self.phih.copy()
        self._calc_grad_phi()
        Fn0w = -self.jacobian_psi_phi() - 0.5j*self.fft(self.phi*self.q_psi)
        self.phih = (self.expch_hw*self.phih0 + Fn0w*self.Qhw)*self.filtr
        self.phih1 = self.phih.copy()

        # phi-equation
        self._calc_grad_phi()
        Fnaw = -self.jacobian_psi_phi() - 0.5j*self.fft(self.phi*self.q_psi)
        self.phih = (self.expch_hw*self.phih0 + Fnaw*self.Qhw)*self.filtr

        # phi-equation
        self._calc_grad_phi()
        Fnbw = -self.jacobian_psi_phi() - 0.5j*self.fft(self.phi*self.q_psi)
        self.phih = (self.expch_hw*self.phih1 + ( 2.*Fnbw - Fn0w )*self.Qhw)*self.filtr

        # phi-equation
        self._calc_rel_vorticity()
        self._calc_grad_phi()
        Fncw = -self.jacobian_psi_phi() - 0.5j*self.fft(self.phi*self.q_psi)
        self.phih = (self.expchw*self.phih0 + Fn0w*self.f0w +  2.*(Fnaw+Fnbw)*self.fabw\
                  + Fncw*self.fcw)*self.filtr

        # physical space
        self.phi = self.ifft(self.phih)

    def _initialize_etdrk4(self):

        """ This performs pre-computations for the Expotential Time Differencing
            Fourth Order Runge Kutta time stepper. The linear part is calculated
            exactly.
            See Cox and Matthews, J. Comp. Physics., 176(2):430-455, 2002.
                Kassam and Trefethen, IAM J. Sci. Comput., 26(4):1214-233, 2005. """


        M = 32.  # number of points for line integral in the complex plane
        rho = 1.  # radius for complex integration
        r = rho*np.exp(2j*np.pi*((np.arange(1.,M+1))/M)) # roots for integral

        # the exponent for the linear part
        self.c = np.zeros((self.nl,self.nk),self.dtype_cplx)  -1j*self.k*self.U
        self.c += -self.nu4w*self.wv4 - 0.5j*self.f*(self.wv2/self.kappa2)\
                        - self.nuw*self.wv2 - self.muw
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

    def jacobian_psi_phi(self):
        """ Compute the Jacobian phix and phiy. """
        return self.fft( (self.u*self.phix + self.v*self.phiy) )

    def _calc_grad_phi(self):
        """ Calculates grad phi """
        self.phix, self.phiy = self.ifft(self.ik*self.phih), self.ifft(self.il*self.phih)

    def _invert(self):
        """ From qh compute ph and compute velocity. """

        self.ph = -self.wv2i*self.qh


    def _initialize_class_diagnostics(self):
        pass

    def _calc_class_derived_fields(self):
        pass
