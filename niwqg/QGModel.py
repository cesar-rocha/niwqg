import numpy as np
from numpy import pi
import logging
import h5py
from numpy import pi
import logging, os
from .Diagnostics import *
from .Saving import *

class Model(object):

    """ Python class that represents the barotropic quasigeostrophic
        pseudospectral model in a doubly periodic domain. Physical parameters
        observe SI units.

    Parameters
    -----------
    nx: integer (optional)
            Number of grid points in the x-direction.
            The number of modes is nx/2+1.
    ny: integer (optional)
            Number of grid points in the y-direction.
            If None, then ny=nx.
    L:  float (optional)
            Domain size.
    dt: float (optional)
            Time step for time integration.
    twrite: integer (optional)
            Print model status to screen every twrite time steps.
    tmax: float (optional)
            Total time of simulation.
    U: float (optional)
            Uniform zonal flow
    use_filter: bool (optional)
            If True, then uses exponential spectral filter.
    nu4: float (optional)
            Fouth-order hyperdiffusivity of potential vorticity.
    nu: float (optional)
            Diffusivity of potential vorticity.
    mu: float (optional)
            Linear drag of potential vorticity.
    passive_scalar: bool (optional)
            If True, then calculates passive scalar solution.
    nu4c: float (optional)
            Fouth-order hyperdiffusivity of passive scalar.
    nuc: float (optional)
            Diffusivity of passive scalar.
    muc: float (optional)
            Linear drag of passive scalar.
    dealias: bool (optional)
            If True, then dealias solution using 2/3 rule.
    save_to_disk: bool (optional)
            If True, then save parameters and snapshots to disk.
    overwrite: bool (optional)
            If True, then overwrite extant files.
    tsave_snapshots: integer (optional)
            Save snapshots every tsave_snapshots time steps.
    tdiags: integer (optional)
            Calculate diagnostics every tdiags time steps.
    path: string (optional)
            Location for saving output files.

    """

    def __init__(
        self,
        nx=128,
        ny=None,
        L=5e5,
        dt=10000.,
        twrite=1000,
        tswrite=10,
        tmax=250000.,
        use_filter = True,
        U = .0,
        nu4=5.e9,
        nu = 0,
        mu = 0,
        beta = 0,
        passive_scalar = False,
        nu4c = 5.e9,
        nuc = 0,
        muc = 0,
        dealias = False,
        save_to_disk=False,
        overwrite=True,
        tsave_snapshots=10,
        tdiags = 10,
        path = 'output/'):

        self.nx = nx
        self.ny = nx
        self.L = L
        self.W = L

        self.dt = dt
        self.twrite = twrite
        self.tswrite = tswrite
        self.tmax = tmax
        self.tdiags = tdiags
        self.passive_scalar = passive_scalar
        self.dealias = dealias

        self.U = U
        self.beta = beta
        self.nu4 = nu4
        self.nu = nu
        self.mu = mu
        self.nu4c = nu4c
        self.nuc = nuc
        self.muc = muc

        self.save_to_disk = save_to_disk
        self.overwrite = overwrite
        self.tsnaps = tsave_snapshots
        self.path = path

        self.use_filter = use_filter

        self._initialize_logger()
        self._initialize_grid()
        self._allocate_variables()
        self._initialize_filter()
        self._initialize_etdrk4()
        self._initialize_time()

        initialize_save_snapshots(self, self.path)
        save_setup(self, )

        self.cflmax = .5

        self._initialize_fft()

        self._initialize_diagnostics()


    def _allocate_variables(self):

        """ Allocate variables so that variable addresses are close in memory.
        """

        self.dtype_real = np.dtype('float64')
        self.dtype_cplx = np.dtype('complex128')
        self.shape_real = (self.ny, self.nx)
        self.shape_cplx = (self.ny, self.nx//2+1)

        # vorticity
        self.q  = np.zeros(self.shape_real,  self.dtype_real)
        self.qh = np.zeros(self.shape_cplx,  self.dtype_cplx)
        self.qh0 = np.zeros(self.shape_cplx, self.dtype_cplx)
        self.qh1 = np.zeros(self.shape_cplx, self.dtype_cplx)

        # stream function
        self.p  = np.zeros(self.shape_real,  self.dtype_real)
        self.ph = np.zeros(self.shape_cplx,  self.dtype_cplx)

    def run_with_snapshots(self, tsnapstart=0., tsnapint=432000.):

        """ Run the model for prescribed time and yields to user code.

            Parameters
            ----------

            tsnapstart : float
                            The timestep at which to begin yielding.
            tstapint : int (number of time steps)
                            The interval at which to yield.

        """

        tsnapints = np.ceil(tsnapint/self.dt)

        while(self.t < self.tmax):
            self._step_forward()
            if self.t>=tsnapstart and (self.tc%tsnapints)==0:
                yield self.t
        return

    def run(self):

        """ Run the model until the end (`tmax`).

            The algorithm is:
                1) Save snapshots (i.e., save the initial condition).
                2) Take a tmax/dt steps forward.
                3) Save diagnostics.
        """

        # save initial conditions
        if self.save_to_disk:
            save_snapshots(self,fields=['t','q','p'])

        # run the model
        while(self.t < self.tmax):
            self._step_forward()

        # save diagnostics
        if self.save_to_disk:
            save_diagnostics(self)

    def _step_forward(self):

        """ Step solutions forwards. The algorithm is:
                1) Take one time step with ETDRK4 scheme.
                2) Incremente diagnostics.
                3) Print status.
                4) Save snapshots.
        """

        self._step_etdrk4()
        increment_diagnostics(self,)
        self._print_status()
        save_snapshots(self,fields=['t','q','p'])

    def _initialize_time(self):

        """ Initialize model clock and other time variables.
        """

        self.t=0        # time
        self.tc=0       # time-step number

    ### initialization routines, only called once at the beginning ###
    def _initialize_grid(self):

        """ Create spatial and spectral grids and normalization constants.
        """

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

        """Set up spectral filter or dealiasing."""

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

    def _initialize_logger(self):

        """ Initialize logger.
        """

        self.logger = logging.getLogger(__name__)

        fhandler = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s: %(message)s')

        fhandler.setFormatter(formatter)

        if not self.logger.handlers:
            self.logger.addHandler(fhandler)

        self.logger.setLevel(10)

        # this prevents the logger from propagating into the ipython notebook log
        self.logger.propagate = False
        self.logger.info(' Logger initialized')


    def _step_etdrk4(self):

        """ Take one step forward using an exponential time-dfferencing method
            with a Runge-Kutta 4 scheme.

            Rereferences
            ------------
            See Cox and Matthews, J. Comp. Physics., 176(2):430-455, 2002.
            Kassam and Trefethen, IAM J. Sci. Comput., 26(4):1214-233, 2005.

        """

        self.qh0 = self.qh.copy()
        Fn0 = -self.jacobian_psi_q()
        self.qh = (self.expch_h*self.qh0 + Fn0*self.Qh)*self.filtr
        self.qh1 = self.qh.copy()

        if self.passive_scalar:
            self.ch0 = self.ch.copy()
            Fn0c = -self.jacobian_psi_c()
            self.ch = (self.expch_hc*self.ch0 + Fn0c*self.Qhc)*self.filtr
            self.ch1 = self.ch.copy()

            self._calc_derived_fields()
            c1 = self._calc_ep_c()

        self._invert()
        k1 = self._calc_ep_psi()

        Fna = -self.jacobian_psi_q()
        self.qh = (self.expch_h*self.qh0 + Fna*self.Qh)*self.filtr

        if self.passive_scalar:
            Fnac = -self.jacobian_psi_c()
            self.ch = (self.expch_hc*self.ch0 + Fnac*self.Qhc)*self.filtr

            self._calc_derived_fields()
            c2 = self._calc_ep_c()

        self._invert()
        k2 = self._calc_ep_psi()

        Fnb = -self.jacobian_psi_q()
        self.qh = (self.expch_h*self.qh1 + ( 2.*Fnb - Fn0 )*self.Qh)*self.filtr

        if self.passive_scalar:
            Fnbc = -self.jacobian_psi_c()
            self.ch = (self.expch_hc*self.ch1 + ( 2.*Fnbc - Fn0c )*self.Qhc)*self.filtr

            self._calc_derived_fields()
            c3 = self._calc_ep_c()

        self._invert()
        k3 = self._calc_ep_psi()

        Fnc = -self.jacobian_psi_q()
        self.qh = (self.expch*self.qh0 + Fn0*self.f0 +  2.*(Fna+Fnb)*self.fab\
                  + Fnc*self.fc)*self.filtr

        if self.passive_scalar:
            Fncc = -self.jacobian_psi_c()
            self.ch = (self.expchc*self.ch0 + Fn0c*self.f0c+  2.*(Fnac+Fnbc)*self.fabc\
                  + Fncc*self.fcc)*self.filtr

            self._calc_derived_fields()
            c4 = self._calc_ep_c()
            self.cvar += self.dt*(c1 + 2*(c2+c3) + c4)/6.


        # invert
        self._invert()

        # calcuate q
        self.q = self.ifft(self.qh).real

        if self.passive_scalar:
            self.c = self.ifft(self.ch).real

        k4 = self._calc_ep_psi()
        self.Ke += self.dt*(k1 + 2*(k2+k3) + k4)/6.


    def _initialize_etdrk4(self):

        """ Compute coefficients of the exponential time-dfferencing method
            with a Runge-Kutta 4 scheme.

            Rereferences
            ------------
            See Cox and Matthews, J. Comp. Physics., 176(2):430-455, 2002.
            Kassam and Trefethen, IAM J. Sci. Comput., 26(4):1214-233, 2005.

        """
        #
        # coefficients for q-equation
        #

        # the exponent for the linear part
        c = np.zeros((self.nl,self.nk),self.dtype_cplx)
        c += -self.nu4*self.wv4 - self.nu*self.wv2 - self.mu - 1j*self.k*self.U
        c += self.beta*self.ik*self.wv2i
        ch = c*self.dt
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


        if self.passive_scalar:
            #
            # coefficients for c-equation
            #

            # the exponent for the linear part
            c = np.zeros((self.nl,self.nk),self.dtype_cplx)
            c += -self.nu4c*self.wv4 - self.nuc*self.wv2 - self.muc
            ch = c*self.dt
            self.expchc = np.exp(ch)
            self.expch_hc = np.exp(ch/2.)
            self.expch2c = np.exp(2.*ch)

            r = rho*np.exp(2j*np.pi*((np.arange(1.,M+1))/M)) # roots for integral
            LR = ch[...,np.newaxis] + r[np.newaxis,np.newaxis,...]
            LR2 = LR*LR
            LR3 = LR2*LR
            self.Qhc   =  self.dt*(((np.exp(LR/2.)-1.)/LR).mean(axis=-1))
            self.f0c  =  self.dt*( ( ( -4. - LR + ( np.exp(LR)*( 4. - 3.*LR + LR2 ) ) )/ LR3 ).mean(axis=-1) )
            self.fabc =  self.dt*( ( ( 2. + LR + np.exp(LR)*( -2. + LR ) )/ LR3 ).mean(axis=-1) )
            self.fcc  =  self.dt*( ( ( -4. -3.*LR - LR2 + np.exp(LR)*(4.-LR) )/ LR3 ).mean(axis=-1) )


    def jacobian_psi_q(self):

        """ Compute the advective term–––the Jacobian between psi and q.

            Returns
            -------
            complex array of floats
                The Fourier transform of Jacobian(psi,q)
        """

        self.u, self.v = self.ifft(-self.il*self.ph).real, self.ifft(self.ik*self.ph).real
        q = self.ifft(self.qh).real
        return self.ik*self.fft(self.u*q) + self.il*self.fft(self.v*q)

    def jacobian_psi_c(self):

        """ Compute the advective term of the passive scalar equation–––the
            Jacobian between psi and c.

        Returns
        -------
        complex array of floats
            The Fourier transform of Jacobian(psi,c)
        """

        self.c = self.ifft(self.ch).real
        return self.ik*self.fft(self.u*self.c) + self.il*self.fft(self.v*self.c)

    def _invert(self):

        """ Calculate the streamfunction given the potential vorticity.
        """
        # invert for psi
        self.ph = -self.wv2i*(self.qh)

        # physical space
        self.p = self.ifft(self.ph)

    def set_q(self,q):

        """ Initialize the potential vorticity.

        Parameters
        ----------
        q: an array of floats of dimension (nx,ny):
                The potential vorticity in physical space.
        """

        self.q = q
        self.qh = self.fft(self.q)
        self._invert()
        self.Ke = self._calc_ke_qg()

    def set_c(self,c):

        """ Initialize the potential vorticity.

        Parameters
        ----------
        c: an array of floats of dimension (nx,ny):
                The passive scalar in physical space.
        """

        self.c = c
        self.ch = self.fft(self.c)
        self.cvar = self.spec_var(self.ch)

    def _initialize_fft(self):

        """ Define the two-dimensional FFT methods.
        """

        self.fft =  (lambda x : np.fft.rfft2(x))
        self.ifft = (lambda x : np.fft.irfft2(x))

    def _print_status(self):

        """ Print out the the model status.
                Step: integer
                        Number of time steps completed
                Time: float
                        The elapsed time.
                P: float
                        The percentage of simulation completed.
                Ke: float
                        The geostrophic kinetic energy.
                CFL: float
                        The CFL number.
        """

        self.tc += 1
        self.t += self.dt

        if (self.tc % self.twrite)==0:
            self.ke = self._calc_ke_qg()
            self.cfl = self._calc_cfl()
            self.logger.info('Step: %i, Time: %4.3e, P: %4.3e , Ke: %4.3e, CFL: %4.3f'
                    , self.tc,self.t, self.t/self.tmax, self.ke, self.cfl )

            assert self.cfl<self.cflmax, self.logger.error('CFL condition violated')

    def _calc_ke_qg(self):
        """ Compute geostrophic kinetic energy, Ke. """
        return 0.5*self.spec_var(self.wv*self.ph)

    def _calc_ens(self):
        """ Compute geostrophic potential enstrophy. """
        return 0.5*self.spec_var(self.qh)

    def _calc_ep_psi(self):
        """ Compute dissipation of Ke """
        lap2psi = self.ifft(self.wv4*self.ph)
        lapq = self.ifft(-self.wv2*self.qh)
        return self.nu4*(self.q*lap2psi).mean() - self.nu*(self.p*lapq).mean()\
                + self.mu*(self.p*self.q).mean()

    def _calc_ep_c(self):
        """ Compute dissipation of C2 """
        return -2*self.nu4c*(self.lapc**2).mean() - 2*self.nu*self.gradC2\
                - 2*self.muc*self.C2

    def _calc_chi_c(self):
        """ Compute dissipation of gradC2 """
        lap2c = self.ifft(self.wv4*self.ch)
        return 2*self.nu4c*(lap2c*self.lapc).mean() - 2*self.nu*(self.lapc**2).mean()\
                - 2*self.muc*self.gradC2

    def _calc_chi_q(self):
        """"  Calculates dissipation of geostrophic potential
              enstrophy, S. """
        return -self.nu4*self.spec_var(self.wv2*self.qh)

    def spec_var(self, ph):
        """ Compute variance of a variable `p` from its Fourier transform `ph` """
        var_dens = 2. * np.abs(ph)**2 / self.M**2
        # only half of coefs [0] and [nx/2+1] due to symmetry in real fft2
        var_dens[:,0] *= 0.5
        var_dens[:,-1] *= 0.5
        # remove mean
        var_dens[0,0] = 0
        return var_dens.sum()

    def _calc_cfl(self):

        """ Compute the CFL number. """

        # avoid destruction by fftw
        self.u = self.ifft(-self.il*self.ph)
        self.v = self.ifft(self.ik*self.ph)

        return np.abs(np.hstack([self.u, self.v])).max()*self.dt/self.dx


    def _initialize_diagnostics(self):

        """ Initialize the diagnostics dictionary with each diganostic and an
            entry.
        """

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

        add_diagnostic(self, 'Ke',
                description='Quasigeostrophic Kinetic Energy, from energy equation',
                units=r'm^2 s^{-2}',
                types = 'scalar',
                function = (lambda self: self.Ke)
        )

        add_diagnostic(self,'ens',
                description='Quasigeostrophic Potential Enstrophy',
                units=r's^{-2}',
                types = 'scalar',
                function = (lambda self: 0.5*(self.q**2).mean())
        )

        add_diagnostic(self, 'ep_psi',
                description='The hyperviscous dissipation of QG kinetic energy',
                units=r'$m^2 s^{-3}$',
                types = 'scalar',
                function = (lambda self: self._calc_ep_psi())
        )

        add_diagnostic(self, 'chi_q',
                description='The hyperviscous dissipation of QG kinetic energy',
                units=r'$s^{-3}$',
                types = 'scalar',
                function = (lambda self: self._calc_chi_q())
        )

        add_diagnostic(self, 'C2',
                description='Passive tracer variance',
                units=r'[scalar]^2',
                types = 'scalar',
                function = (lambda self: self.C2)
        )

        add_diagnostic(self, 'cvar',
                description='Passive tracer variance, from variance equation',
                units=r'[scalar]^2',
                types = 'scalar',
                function = (lambda self: self.cvar)
        )

        add_diagnostic(self, 'gradC2',
                description='Gradient of Passive tracer variance',
                units=r'[scalar]^2 / m^2',
                types = 'scalar',
                function = (lambda self: self.gradC2)
        )

        add_diagnostic(self, 'Gamma_c',
                description='Rate of generation of passive tracer gradient variance',
                units=r'[scalar]^2 / (m^2 s)',
                types = 'scalar',
                function = (lambda self: self.Gamma_c)
        )

        add_diagnostic(self, 'ep_c',
                description='The dissipation of tracer variance',
                units=r'$s^{-3}$',
                types = 'scalar',
                function = (lambda self: self._calc_ep_c())
        )

        add_diagnostic(self, 'chi_c',
                description='The dissipation of tracer gradient variance',
                units=r'$s^{-3}$',
                types = 'scalar',
                function = (lambda self: self._calc_chi_c())
        )

    def _calc_derived_fields(self):
        """ Compute derived fields necessary for model diagnostics. """

        if self.passive_scalar:
            self.C2 =  self.spec_var(self.ch)
            self.gradC2 =  self.spec_var(self.wv*self.ch)

            self.lapc = self.ifft(-self.wv2*self.ch)
            self.Gamma_c = 2*(self.lapc*self.ifft(self.jacobian_psi_c())).mean()

        else:
            self.C2, self.gradC2, self.cvar = 0., 0., 0.
            self.c, self.ch = 0., 0.
            self.lapc, self.Gamma_c = np.array([0.]), 0.
