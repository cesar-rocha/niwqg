import numpy as np
from numpy import pi
import logging, os
import h5py

from .Diagnostics import *
from .Saving import *

class Kernel(object):

    """" QG-NIW model """

    def __init__(
        self,
        # grid size parameters
        nx=64,                     # grid resolution
        ny=None,
        L=1e6,                     # domain size is L [m]
        # timestepping parameters
        dt=7200.,                   # numerical timestep
        twrite=1000.,               # interval for cfl and ke writeout (in timesteps)
        tmax=1576800000.,           # total time of integration
        use_filter = True,
        cflmax = 0.8,               # largest CFL allowed
        # constants
        U = .0,                     # uniform zonal flow
        f = 1.,                     # coriolis parameter (not necessary for two-layer model
        N = 1.,                     # buoyancy frequency
        m = 1.,                     # vertical wavenumber
        g= 9.81,                    # acceleration due to gravity
        nu4=5.e9,                   # hyperviscosity
        nu4w=5.e5,                  # hyperviscosity waves
        dealias = False,
        save_to_disk=True,
        overwrite=True,
        tsave_snapshots=10,         # interval fro saving snapshots (in timesteps)
        tdiags=10,                  # interval for diagnostics (in timesteps)
        path = 'output/'):

        # put all the parameters into the object
        # grid
        self.nx = nx
        self.ny = nx
        self.L = L
        self.W = L

        # time
        self.dt = dt
        self.twrite = twrite
        self.tmax = tmax
        # fft
        self.dealias = dealias

        # constants
        self.U = U
        self.g = g
        self.nu4 = nu4
        self.nu4w = nu4w
        self.f = f
        self.N = N
        self.m = m
        self.kappa = self.m*self.f/self.N
        self.kappa2 = self.kappa**2
        self.cflmax = cflmax

        # nondimensional parameters
        self.hslash = self.f/self.kappa2

        # saving stuff
        self.save_to_disk = save_to_disk
        self.overwrite = overwrite
        self.tsnaps = tsave_snapshots

        self.tdiags = tdiags
        self.path = path

        # flags
        self.use_filter = use_filter

        # initializations
        self._initialize_logger()
        self._initialize_grid()
        self._allocate_variables()
        self._initialize_filter()
        self._init_etdrk4()
        self._initialize_time()

        init_save_snapshots(self,self.path)
        save_setup(self,)

        # fft
        self._init_fft()

        # diagnostics
        self._initialize_diagnostics()

    def _allocate_variables(self):
        """ Allocate variables in memory """

        raise NotImplementedError(
            'needs to be implemented by Model subclass')


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
        save_snapshots(self,fields=['t','q','phi'])

    def _initialize_time(self):
        """Set up timestep stuff"""
        self.t=0        # actual time
        self.tc=0       # timestep number

    def _initialize_grid(self):
        """Set up spatial and spectral grids and related constants"""
        self.x,self.y = np.meshgrid(
            np.arange(0.5,self.nx,1.)/self.nx*self.L,
            np.arange(0.5,self.ny,1.)/self.ny*self.W )

        self.dk = 2.*pi/self.L
        self.dl = 2.*pi/self.L

        # wavenumber grids
        self.nl = self.ny
        self.nk = self.nl
        self.ll = self.dl*np.append( np.arange(0.,self.nx/2),
            np.arange(-self.nx/2,0.) )
        self.kk = self.ll.copy()

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

    def _initialize_filter(self):
        """Set up frictional filter or dealiasing."""
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

    def _initialize_logger(self):

        self.logger = logging.getLogger(__name__)

        fhandler = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s: %(message)s')

        fhandler.setFormatter(formatter)

        if not self.logger.handlers:
            self.logger.addHandler(fhandler)

        self.log_level = 1
        self.logger.setLevel(self.log_level*10)

        self.logger.propagate = False
        self.logger.info(' Logger initialized')


    def _step_etdrk4(self):
        """ march the system forward using a ETDRK4 scheme """

        raise NotImplementedError(
            'needs to be implemented by Model subclass')

    def _init_etdrk4(self):

        """ This performs pre-computations for the Expotential Time Differencing
            Fourth Order Runge Kutta time stepper. The linear part is calculated
            exactly.
            See Cox and Matthews, J. Comp. Physics., 176(2):430-455, 2002.
                Kassam and Trefethen, IAM J. Sci. Comput., 26(4):1214-233, 2005. """

        raise NotImplementedError(
            'needs to be implemented by Model subclass')

    def jacobian_psi_phi(self):
        """ Compute the Jacobian phix and phiy. """
        raise NotImplementedError(
            'needs to be implemented by Model subclass')

    def jacobian_psi_q(self):
        """ Compute the Jacobian between psi and q. Return in Fourier space. """

        raise NotImplementedError(
            'needs to be implemented by Model subclass')

    def _invert(self):
        """ From qh compute ph and compute velocity. """

        raise NotImplementedError(
            'needs to be implemented by Model subclass')

    def _calc_rel_vorticity(self):
        """ from psi compute relative vorticity """
        self.qh_psi = -self.wv2*self.ph
        self.q_psi = self.ifft(self.qh_psi).real

    def _calc_strain(self):
        """ from psi compute geostrophic rate of strain """
        pxx,pyy = self.ifft(-self.k*self.k*self.ph).real, self.ifft(-self.l*self.l*self.ph).real
        pxy = self.ifft(-self.k*self.l*self.ph).real
        self.qg_strain =  4*(pxy**2)+(pxx-pyy)**2

    def _calc_OW(self):
        """ calculate the Okubo-Weiss parameter """
        self._calc_rel_vorticity()
        self._calc_strain()
        return self.qg_strain**2 - self.q_psi**2

    def set_q(self,q):
        """ Initialize pv """
        self.q = q
        self.qh = self.fft(self.q)
        self._invert()
        self._calc_rel_vorticity()
        self.u, self.v = self.ifft(-self.il*self.ph).real, self.ifft(self.ik*self.ph).real

    def set_phi(self,phi):
        """ Initialize pv """
        self.phi = phi
        self.phih = self.fft(self.phi)

    def _init_fft(self):
        self.fft =  (lambda x : np.fft.fft2(x))
        self.ifft = (lambda x : np.fft.ifft2(x))

    def _print_status(self):
        """Output some basic stats."""
        self.tc += 1
        self.t += self.dt

        if (self.log_level) and ((self.tc % self.twrite)==0):
            self.ke = self._calc_ke_qg()
            self.kew = self._calc_ke_niw()
            self.pew = self._calc_pe_niw()
            self.cfl = self._calc_cfl()
            self.logger.info('Step: %i, Time: %4.3e, P: %4.3e , KE QG: %4.3e, KE NIW: %4.3e, PE NIW: %4.3e,CFL: %4.3f'
                    , self.tc,self.t, self.t/self.tmax,self.ke,self.kew,self.pew,self.cfl )

            assert self.cfl<self.cflmax, self.logger.error('CFL condition violated')

    def _calc_ke_qg(self):
        return 0.5*self.spec_var(self.wv*self.ph)

    def _calc_ke_niw(self):
        return 0.5*(np.abs(self.phi)**2).mean()

    def _calc_pe_niw(self):
        self.phix, self.phiy = self.ifft(self.ik*self.phih),self.ifft(self.il*self.phih)
        return 0.25*( np.abs(self.phix)**2 +  np.abs(self.phiy)**2 ).mean()/self.kappa2

    def _calc_conc(self):
        """ Calculates the concentratin parameter of NIWs """
        self.upsilon = np.abs(self.phi)**2 -  (np.abs(self.phi)**2).mean()
        return (self.upsilon*self.q_psi).mean()/self.upsilon.std()/self.q_psi.std()

    def _calc_skewness(self):
        """ calculates skewness in relative vorticity """
        self.qpsi = self.ifft(-self.wv2*self.ph).real
        return ( (self.qpsi**3).mean() / (((self.qpsi**2).mean())**1.5) )

    def _calc_ens(self):
        """ calculates potential enstrophy """
        return 0.5*(self.q**2).mean()

    def _calc_ep_phi(self):
        """ calculates dissipation of NIW KE due to hyperviscosity """
        return self.nu4*(np.abs(self.lapphi)**2).mean()

    def spec_var(self, ph):
        """ compute variance of p from Fourier coefficients ph """
        var_dens = np.abs(ph)**2 / self.M**2
        var_dens[0,0] = 0.
        return var_dens.sum()

    def _calc_cfl(self):
        return np.abs(np.hstack([self.u, self.v,np.abs(self.phi)])).max()*self.dt/self.dx

    def _calc_energy_conversion(self):

        self.u, self.v = self.ifft(-self.il*self.ph).real, self.ifft(self.ik*self.ph).real
        self._calc_rel_vorticity()

        J_psi_phi = self.u*self.phix+self.v*self.phiy
        self.lapphi = np.fft.ifft2(-self.wv2*self.phih)

        # div fluxes
        divFw = 0.5*self.hslash*(np.conj(self.phi)*self.lapphi).imag

        # correlations
        self.gamma1 = -(0.5*self.q_psi*divFw).mean()/self.f
        self.gamma2 = -0.5*self.hslash*((np.conj(self.lapphi)*J_psi_phi).real).mean()/self.f
        self.pi = (0.5*self.phi.mean()*(self.q_psi*np.conj(self.phi)).mean()).imag

    def _calc_icke_niw(self):
        self.ke_niw = self._calc_ke_niw()
        self.cke_niw = 0.5*(np.abs(self.phi.mean())**2)
        self.ike_niw = self.ke_niw-self.cke_niw

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


        add_diagnostic(self, 'ens',
                description='Quasigeostrophic Potential Enstrophy',
                units=r's^{-2}',
                types = 'scalar',
                function = (lambda self: 0.5*(self.q**2).mean())
        )


        add_diagnostic(self, 'ke_niw',
                description='Near-inertial Kinetic Energy',
                units=r'm^2 s^{-2}',
                types = 'scalar',
                function = (lambda self: self.ke_niw)
        )

        add_diagnostic(self, 'cke_niw',
                description='Kinetic Energy of Laterally Coherent Near-Inertial Waves',
                units=r'm^2 s^{-2}',
                types = 'scalar',
                function = (lambda self: self.cke_niw)
        )

        add_diagnostic(self, 'ike_niw',
                description='Kinetic Energy of Laterally Incoherent Near-Inertial Waves',
                units=r'm^2 s^{-2}',
                types = 'scalar',
                function = (lambda self: self.ike_niw)
        )

        add_diagnostic(self, 'pe_niw',
                description='Near-inertial Potential Energy',
                units=r'm^2 s^{-2}',
                types = 'scalar',
                function = (lambda self: self._calc_pe_niw())
        )

        add_diagnostic(self, 'conc_niw',
                description='Correlation between relative vorticity and near-inertial KE',
                units=r'unitless',
                types = 'scalar',
                function = (lambda self: self._calc_conc())
        )

        add_diagnostic(self, 'skew',
                description='Skewness',
                units=r'unitless',
                types = 'scalar',
                function = (lambda self: self._calc_skewness())
        )

        add_diagnostic(self, 'gamma_r',
                description='The energy conversion due to refraction',
                units=r'$m^2 s^{-3}$',
                types = 'scalar',
                function = (lambda self: self.gamma1)
        )

        add_diagnostic(self, 'gamma_a',
                description='The energy conversion due to advection',
                units=r'$m^2 s^{-3}$',
                types = 'scalar',
                function = (lambda self: self.gamma2)
        )

        add_diagnostic(self, 'pi',
                description='The NIW kinetic energy conversion from coherent to incoherent',
                units=r'$m^2 s^{-3}$',
                types = 'scalar',
                function = (lambda self: self.pi)
        )

        add_diagnostic(self, 'ep_phi',
                description='The NIW kinetic energy dissipation due to hyper-viscosity',
                units=r'$m^2 s^{-3}$',
                types = 'scalar',
                function = (lambda self: self._calc_ep_phi())
        )

    def _calc_derived_fields(self):
        """Should be implemented by subclass."""
        self._calc_energy_conversion()
        self._calc_icke_niw()
