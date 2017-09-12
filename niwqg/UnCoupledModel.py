import numpy as np
from . import Kernel
from .Diagnostics import *

class Model(Kernel.Kernel):

    """ A subclass that represents the Young & Ben Jelloul uncoupled model
        of single-vertical wavenumber near-inertial waves and barotropic
        quasigeostrophic.

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

        self.model = " Uncoupled Model"

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


    def _invert(self):

        """ Calculate the streamfunction given the potential vorticity.
        """

        # invert for psi
        self.ph = -(self.wv2i*self.qh)
        self.p = self.ifft(self.ph).real

        # calcuate q
        self.q = self.ifft(self.qh).real

    def _initialize_class_diagnostics(self):

        """ Compute subclass-specific derived fields.
        """
        pass

    def _calc_class_derived_fields(self):
        """  Compute the geostrophic relative vorticity–––the Laplacian of the
                streamfuctions.
        """
        pass
