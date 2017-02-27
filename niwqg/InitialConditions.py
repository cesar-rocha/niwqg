import numpy as np
import scipy.special as special

def McWilliams1984(model,k0=6,E=0.5):
    """ Generates random vorticity field with red spectrum given in
        Mcwilliams, J.C. (1984): 'The emergence of isolated coherent vortices in
                                 turbulent flow,' Journal of Fluid Mechanics, 146,
                                 pp. 21–43. doi: 10.1017/S0022112084001750.

       parameters
       -----------
        - model: model class
        - k0: centroid of the energy spectrum
        - E:  energy level

       return
       -----------
        - q: vorticity (physical space)

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

       parameters
       -----------
        - model: model class
        - k0: wavenumber
        - E:  energy level

       return
       -----------
        - q: vorticity (physical space)

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
    """ Generates Lamb's dipole vorticity field.

       parameters
       -----------
        - U: translation speed (dipole's strength)
        - R: radius

       return
       -----------
        - q: vorticity (physical space)


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
    """ Generates wave packet.

       parameters
       -----------
        - k,l : wavenumber vector
        - R: radius

       return
       -----------
        - phi: complexified NIWs velocity (physical space)


    """

    N = model.nx
    x, y = model.x, model.y

    r = np.sqrt( (x-x0)**2 + (y-y0)**2 )

    phi = np.exp(1j*(k*(x-x0)+l*(y-y0)))
    phi[r>R] = 0.+0j

    return phi

def PlaneWave(model, k=10,l=0,phase=0.):
    """ Generates plane wave.

       parameters
       -----------
        - k,l : wavenumber vector
        - phase: phase

       return
       -----------
        - phi: complexified NIWs velocity (physical space)


    """

    phi = np.exp(1j*(k*x+l*y)+phase)

    return phi

