import numpy as np
import scipy.special as special

def McWilliams1984(model,k0=6,E=0.5):

    """ Generate random vorticity field with red spectrum given in
        Mcwilliams, J.C. (1984): 'The emergence of isolated coherent vortices in
                                 turbulent flow,' Journal of Fluid Mechanics, 146,
                                 pp. 21–43. doi: 10.1017/S0022112084001750.

       Parameters
       -----------
        model: python class
                The model class.
        k0: float
                The centroid of the energy spectrum.
        E:  float
                The energy (the strengh of the random field).

       Return
       ------
        q: array of floats
                Potential vorticity (physical space).

    """

    ckappa = np.zeros_like(model.wv2)
    nhx,nhy = model.wv2.shape
    kc2 = k0**2

    fk = model.wv != 0
    ckappa[fk] = np.sqrt( model.wv2[fk]*(1. + (model.wv2[fk]/kc2)**2) )**-1

    phase = np.random.rand(nhx,nhy)*2*np.pi
    ph = ckappa*np.cos(phase) + 1j*ckappa*np.sin(phase)
    ph = model.fft(model.ifft(ph).real)
    Eaux = 0.5*model.spec_var( model.wv*ph )
    pih = np.sqrt(E/Eaux)*ph
    qih = -model.wv2*pih

    return model.ifft(qih).real

def Danioux2015(model,k0=6,E=0.5):
    """ Generates single wavenumber random vorticity field

       Parameters
       -----------
        model: python class
                The model class.
        k0: float
                The single wavenumber of the vorticity field.
        E:  float
                The energy (the strengh of the random field).

       Return
       -----------
       q: float
            Vorticity (physical space).

                                                                             """
    ckappa = np.zeros_like(model.wv2)
    nhx,nhy = model.wv2.shape
    kc2 = k0**2

    fk = model.wv != 0
    ckappa[fk] = np.sqrt( model.wv[fk]*np.exp(-(model.wv2[fk]/kc2 )))

    phase = np.random.rand(nhx,nhy)*2*np.pi
    ph = ckappa*np.cos(phase) + 1j*ckappa*np.sin(phase)
    ph = model.fft(model.ifft(ph).real)
    Eaux = 0.5*model.spec_var( model.wv*ph )
    pih = np.sqrt(E/Eaux)*ph
    qih = -model.wv2*pih

    return model.ifft(qih).real

def LambDipole(model, U=.01,R = 1.):

    """ Generate Lamb's dipole vorticity field.

       Parameters
       -----------
        U: float
                Translation speed (dipole's strength).
        R: float
                Diple's radius.

       Return
       -------
        q: array of floats
              Vorticity (physical space).

    """

    N = model.nx
    x, y = model.x, model.y
    x0,y0 = x[N//2,N//2],y[N//2,N//2]

    r = np.sqrt( (x-x0)**2 + (y-y0)**2 )
    s = np.zeros_like(r)

    for i in range(N):
        for j in range(N):
            if r[i,j] == 0.:
                s[i,j] = 0.
            else:
                s[i,j] = (y[i,j]-y0)/r[i,j]

    lam = (3.8317)/R
    C = -(2.*U*lam)/(special.j0(lam*R))
    q = np.zeros_like(r)
    q[r<=R] = C*special.j1(lam*r[r<=R])*s[r<=R]

    return q


def WavePacket(model, k=10, l=0, R = 1,x0=0.,y0=0.):

    """ Generates Gaussian wave-packet initial condition.

       Parameters
       -----------
       model: python class
                    The model class
       k,l : int
                    Wavenumber vector
       R: float
                    Decay scale of the Gaussian packet.

       Return
       -------
       phi: array of complex floats
                    near-inertial velocity (physical space)

    """

    N = model.nx
    x, y = model.x, model.y

    r = np.sqrt( (x-x0)**2 + (y-y0)**2 )

    phi = np.exp(1j*(k*(x-x0)+l*(y-y0)))
    phi[r>R] = 0.+0j

    return phi

def PlaneWave(model, k=10,l=0,phase=0.):

    """ Generate plane-wave initial condition.

       Parameters
       -----------
       model: python class
                    The model class
        k,l : float
                    Wavenumber vector
        phase: float
                    Phase [0, 2pi)

       Return
       -------
        phi: array of complex floats
                    near-inertial velocity (physical space)

    """

    phi = np.exp(1j*(k*x+l*y)+phase)

    return phi
