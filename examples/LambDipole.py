"""
 Initially laterally coherent near-inertial oscillation
    coupled with Lamb dipole.

 This example runs in about 20 secs on a MacPro 2.2 GHz Intel Core i7,
    16 GB 1600 MHz DDR3.
"""
import timeit
start = timeit.default_timer()

import matplotlib.pyplot as plt
plt.rcParams['contour.negative_linestyle'] = 'dashed'
import numpy as np

from niwqg import CoupledModel as Model
from niwqg import InitialConditions as ic

plt.close('all')

# parameters
nx = 128
f0 = 1.e-4
N = 0.01
L = 2*np.pi*200e3
λz = 280
m = 2*np.pi/λz

# eddy parameters
k0 = 10*(2*np.pi/L)
L0 = 2*np.pi/k0

# initial conditions
U = 1.e-1
phi0 = 2*U
U0 = U
u0 = phi0

# simulation parameters
Te = (U0*k0)**-1 # eddy turn-over time scale
Tf = 2*np.pi/f0

dt = .025*Te
tmax = 10*Te

m = Model.Model(L=L,nx=nx, tmax = tmax,dt = dt,
                m=m,N=N,f=f0, twrite=int(0.25*Tf/dt),
                nu4=10.e8,nu4w=10.e8,use_filter=False,
                U =-U, tdiags=2,)
#nu4=7.5e8,nu4w=7.5e8,use_filter=False,

# initial conditions
q = ic.LambDipole(m, U=U,R = 2*np.pi/k0)
phi = (np.ones_like(q) + 1j)*u0/np.sqrt(2)

m.set_q(q)
m.set_phi(phi)

# run the model
m.run()

# get diagnostics
time = m.diagnostics['time']['value']
KE_qg = m.diagnostics['ke_qg']['value']
PE_niw = m.diagnostics['pe_niw']['value']
KE_niw = m.diagnostics['ke_niw']['value']
ENS_qg = m.diagnostics['ens']['value']

g1 = m.diagnostics['gamma_r']['value']
g2 = m.diagnostics['gamma_a']['value']
pi = m.diagnostics['pi']['value']
cKE_niw = m.diagnostics['cke_niw']['value']
iKE_niw = m.diagnostics['ike_niw']['value']

ep_phi = m.diagnostics['ep_phi']['value']
ep_psi = m.diagnostics['ep_psi']['value']
chi_q =  m.diagnostics['chi_q']['value']
chi_phi =  m.diagnostics['chi_phi']['value']

dt = time[1]-time[0]
dPE = np.gradient(PE_niw,dt)
dKE = np.gradient(KE_qg,dt)
diKE_niw = np.gradient(iKE_niw,dt)

res_ke = dKE-(-g1-g2+ep_psi)
res_pe = dPE-g1-g2-chi_phi

fig = plt.figure(figsize=(16,9))
lw, alp = 3.,.5
KE0 = KE_qg[0]

ax = fig.add_subplot(221)
plt.plot(time/Te,KE_qg/KE0,label='KE QG',linewidth=lw,alpha=alp)
plt.plot(time/Te,KE_niw/KE_niw[0],label='KE NIW',linewidth=lw,alpha=alp)
plt.plot(time/Te,ENS_qg/ENS_qg[0],label='ENS QG',linewidth=lw,alpha=alp)
plt.xticks([])
plt.ylabel(r'Energy/Enstrophy $[E/E_0, Z/Z_0]$')
plt.legend(loc=3)

ax = fig.add_subplot(222)
plt.plot(time/Te,(KE_qg-KE_qg[0])/KE0,label='KE QG',linewidth=lw,alpha=alp)
plt.plot(time/Te,(PE_niw-PE_niw[0])/KE0,label='PE NIW',linewidth=lw,alpha=alp)
plt.plot(time/Te,(KE_niw-KE_niw[0])/KE0,label='KE NIW',linewidth=lw,alpha=alp)
plt.xticks([])
plt.ylabel(r'Energy  change $[(E-E_0) \times {2}/{U_0^2} ]$')
plt.legend(loc=3)

ax = fig.add_subplot(223)
plt.plot(time/Te,Te*g1/KE0,label=r'Refrac. conversion $\Gamma_r$',linewidth=lw,alpha=alp)
plt.plot(time/Te,Te*g2/KE0,label=r'Adv. conversion $\Gamma_a$',linewidth=lw,alpha=alp)
plt.plot(time/Te,Te*chi_phi/KE0,label=r'PE NIW diss. $\chi_\phi$',linewidth=lw,alpha=alp)
plt.plot(time/Te,Te*(g1+g2+chi_phi)/KE0,label=r'$(\Gamma_r+\Gamma_a+\chi_\phi)$',linewidth=lw,alpha=alp)
plt.plot(time/Te,Te*dPE/KE0,'k--',label=r'PE NIW tendency $\dot K_e$',linewidth=lw,alpha=alp)
plt.legend(loc=3,ncol=2)
plt.xlabel(r"Time [$t \times U_0 k_0$]")
plt.ylabel(r'Power $[\dot E \times {2 k_0}/{U_0} ]$')

ax = fig.add_subplot(224)
plt.plot(time/Te,Te*pi/KE0,label=r'Inc. KE NIW conversion $\Pi$',linewidth=lw,alpha=alp)
plt.plot(time/Te,Te*ep_psi/KE0,label=r'KE NIW disspation $\epsilon_\phi$',linewidth=lw,alpha=alp)
plt.plot(time/Te,Te*(pi+ep_phi)/KE0,label=r'$\pi+\epsilon_\phi$',linewidth=lw,alpha=alp)
plt.plot(time/Te,Te*diKE_niw/KE0,'k--',label=r'Inc. NIW KE tendency',linewidth=lw,alpha=alp)
plt.xlabel(r"Time [$t \times U_0 k_0$]")
plt.ylabel(r'Power $[\dot E \times {2 k_0}/{U_0} ]$')
plt.legend(loc=1)


stop = timeit.default_timer()
print("Time elapsed: %3.2f seconds" %(stop - start))
