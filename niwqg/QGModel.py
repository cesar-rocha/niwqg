import numpy as np
from numpy import pi
import logging
import h5py
from numpy import pi
import logging, os
from Diagnostics import *
from Saving import *

class QGModel(object):

    """" Barotropic QG model """

    def __init__(
        self,
        # grid size parameters
        nx=64,                     # grid resolution
        ny=None,
        L=1e6,                     # domain size is L [m]
        # timestepping parameters
        dt=7200.,                   # numerical timestep
        twrite=1000.,               # interval for cfl and ke writeout (in timesteps)
        tswrite=2,
        tmax=1576800000.,           # total time of integration
        tavestart=315360000.,       # start time for averaging
        taveint=86400.,             # time interval used for summation in longterm average in seconds
        use_filter = True,
        # constants
        U = .0,                     # uniform zonal flow
        nu4=5.e9,                   # hyperviscosity
        beta = 0,                   # beta
        dealias = False,
        save_snapshots=True,
        overwrite=True,
        tsave_snapshots=10,  # in snapshots
        tdiags = 10,  # diagnostics
        path = 'output/'):

        # put all the parameters into the object
        # grid
        self.nx = nx
        self.ny = nx
        self.L = L
        self.W = L

        # timestepping
        self.dt = dt
        self.twrite = twrite
        self.tswrite = tswrite
        self.tmax = tmax
        self.tavestart = tavestart
        self.taveint = taveint
        self.tdiags = tdiags
        # fft
        self.dealias = dealias

        # constants
        self.U = U
        self.beta = beta
        self.nu4 = nu4

        # saving stuff
        self.save_snapshots = save_snapshots
        self.overwrite = overwrite
        self.tsnaps = tsave_snapshots
        self.path = path

        # flags
        self.use_filter = use_filter

        self._initialize_logger()
        self._initialize_grid()
        self._allocate_variables()
        self._initialize_filter()
        self._init_etdrk4()
        self._initialize_time()

        # initialize path to save
        init_save_snapshots(self, self.path)
        save_setup(self, )

        self.cflmax = .5

        # fft
        self._init_fft()
        self._initialize_diagnostics()

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

    def run_with_snapshots(self, tsnapstart=0., tsnapint=432000.):
        """Run the model forward, yielding to user code at specified intervals.

        Parameters
        ----------

        tsnapstart : int
            The timestep at which to begin yielding.
        tstapint : int
            The interval at which to yield.
        """

        tsnapints = np.ceil(tsnapint/self.dt)

        while(self.t < self.tmax):
            self._step_forward()
            if self.t>=tsnapstart and (self.tc%tsnapints)==0:
                yield self.t
        return

    def run(self):
        """Run the model forward without stopping until the end."""
        while(self.t < self.tmax):
            self._step_forward()

    def _step_forward(self):

        self._step_etdrk4()
        increment_diagnostics(self,)
        self._print_status()
        save_snapshots(self,fields=['t','q','p'])

    def _initialize_time(self):
        """Set up timestep stuff"""
        self.t=0        # actual time
        self.tc=0       # timestep number
        self.taveints = np.ceil(self.taveint/self.dt)

    ### initialization routines, only called once at the beginning ###
    def _initialize_grid(self):
        """Set up spatial and spectral grids and related constants"""
        self.x,self.y = np.meshgrid(
            np.arange(0.5,self.nx,1.)/self.nx*self.L,
            np.arange(0.5,self.ny,1.)/self.ny*self.W )

        self.dk = 2.*pi/self.L
        self.dl = 2.*pi/self.L

        # wavenumber grids
        self.nl = self.ny
        self.nk = self.nx//2+1
        self.ll = self.dl*np.append( np.arange(0.,self.nx/2),
            np.arange(-self.nx/2,0.) )
        self.kk = self.dk*np.arange(0.,self.nk)

        self.k, self.l = np.meshgrid(self.kk, self.ll)
        self.ik = 1j*self.k
        self.il = 1j*self.l
        # physical grid spacing
        self.dx = self.L / self.nx
        self.dy = self.W / self.ny

        # constant for spectral normalizations
        self.M = self.nx*self.ny

        # isotropic wavenumber^2 grid
        # the inversion is not defined at kappa = 0
        self.wv2 = self.k**2 + self.l**2
        self.wv = np.sqrt( self.wv2 )
        self.wv4 = self.wv2**2

        iwv2 = self.wv2 != 0.
        self.wv2i = np.zeros_like(self.wv2)
        self.wv2i[iwv2] = self.wv2[iwv2]**-1

    def _initialize_background(self):
        raise NotImplementedError(
            'needs to be implemented by Model subclass')

    def _initialize_inversion_matrix(self):
        raise NotImplementedError(
            'needs to be implemented by Model subclass')

    def _initialize_forcing(self):
        raise NotImplementedError(
            'needs to be implemented by Model subclass')

    def _initialize_filter(self):
        """Set up frictional filter."""
        # this defines the spectral filter (following Arbic and Flierl, 2003)

        if self.use_filter:
            cphi=0.65*pi
            wvx=np.sqrt((self.k*self.dx)**2.+(self.l*self.dy)**2.)
            self.filtr = np.exp(-23.6*(wvx-cphi)**4.)
            self.filtr[wvx<=cphi] = 1.
            self.logger.info(' Using filter')
        elif self.dealias:
            self.filtr = np.ones_like(self.wv2)
            self.filtr[self.nx/3:2*self.nx/3,:] = 0.
            self.filtr[:,self.ny/3:2*self.ny/3] = 0.
            self.logger.info(' Dealiasing with 2/3 rule')
        else:
            self.filtr = np.ones_like(self.wv2)
            self.logger.info(' No dealiasing; no filter')


    def _do_external_forcing(self):
        pass

    # logger
    def _initialize_logger(self):

        self.logger = logging.getLogger(__name__)

        fhandler = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s: %(message)s')

        fhandler.setFormatter(formatter)

        if not self.logger.handlers:
            self.logger.addHandler(fhandler)

        self.log_level = 1
        self.logger.setLevel(self.log_level*10)

        # this prevents the logger from propagating into the ipython notebook log
        self.logger.propagate = False
        self.logger.info(' Logger initialized')


    def _step_etdrk4(self):
        """ march the system forward using a ETDRK4 scheme """


        # q-equation
        self.qh0 = self.qh.copy()
        Fn0 = -self.jacobian_psi_q()
        self.qh = (self.expch_h*self.qh0 + Fn0*self.Qh)*self.filtr
        self.qh1 = self.qh.copy()

        self._invert()
        Fna = -self.jacobian_psi_q()
        self.qh = (self.expch_h*self.qh0 + Fna*self.Qh)*self.filtr

        self._invert()
        Fnb = -self.jacobian_psi_q()
        self.qh = (self.expch_h*self.qh1 + ( 2.*Fnb - Fn0 )*self.Qh)*self.filtr

        self._invert()
        Fnc = -self.jacobian_psi_q()
        self.qh = (self.expch*self.qh0 + Fn0*self.f0 +  2.*(Fna+Fnb)*self.fab\
                  + Fnc*self.fc)*self.filtr

        # invert
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
        self.c += self.beta*self.ik*self.wv2i
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


    def jacobian_psi_q(self):
        """ Compute the Jacobian between psi and q. Return in Fourier space. """
        self.u, self.v = self.ifft(-self.il*self.ph).real, self.ifft(self.ik*self.ph).real
        q = self.ifft(self.qh).real
        return self.ik*self.fft(self.u*q) + self.il*self.fft(self.v*q)

    def _invert(self):
        """ From qh compute ph and compute velocity. """

        # invert for psi
        self.ph = -self.wv2i*(self.qh)

        # physical space
        self.p = self.ifft(self.ph)

    def set_q(self,q):
        """ Initialize pv """
        self.q = q
        self.qh = self.fft(self.q)

    def _init_fft(self):
        self.fft =  (lambda x : np.fft.rfft2(x))
        self.ifft = (lambda x : np.fft.irfft2(x))

    def _print_status(self):
        """Output some basic stats."""
        self.tc += 1
        self.t += self.dt

        if (self.log_level) and ((self.tc % self.twrite)==0):
            self.ke = self._calc_ke_qg()
            self.cfl = self._calc_cfl()
            self.logger.info('Step: %i, Time: %4.3e, P: %4.3e , KE QG: %4.3e, CFL: %4.3f'
                    , self.tc,self.t, self.t/self.tmax, self.ke, self.cfl )

            assert self.cfl<self.cflmax, self.logger.error('CFL condition violated')

    def _calc_ke_qg(self):
        return 0.5*self.spec_var(self.wv*self.ph)

    def _calc_ens(self):
        return 0.5*self.spec_var(self.qh)

    def spec_var(self, ph):
        """ compute variance of p from Fourier coefficients ph """
        var_dens = 2. * np.abs(ph)**2 / self.M**2
        # only half of coefs [0] and [nx/2+1] due to symmetry in real fft2
        var_dens[...,0] = var_dens[...,0]/2.
        var_dens[...,-1] = var_dens[...,-1]/2.
        return var_dens.sum()

    ### All the diagnostic stuff follows. ###
    def _calc_cfl(self):

        # avoid destruction by fftw
        self.u = self.ifft(-self.il*self.ph)
        self.v = self.ifft(self.ik*self.ph)

        return np.abs(np.hstack([self.u, self.v])).max()*self.dt/self.dx

    # saving stuff

    def _initialize_diagnostics(self):

        self.diagnostics = dict()

        add_diagnostic(self,'time',
                description='Time',
                units='seconds',
                types = 'scalar',
                function = (lambda self: self.t)
        )

        add_diagnostic(self, 'ke_qg',
                description='Quasigeostrophic Kinetic Energy',
                units=r'm^2 s^{-2}',
                types = 'scalar',
                function = (lambda self: self._calc_ke_qg())
        )


        add_diagnostic(self,'ens',
                description='Quasigeostrophic Potential Enstrophy',
                units=r's^{-2}',
                types = 'scalar',
                function = (lambda self: 0.5*(self.q**2).mean())
        )


    def _calc_derived_fields(self):
        """Should be implemented by subclass."""
        pass
