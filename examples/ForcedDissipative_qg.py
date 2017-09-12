"""
    Forced-disspative QG: still testing.
"""
import timeit
start = timeit.default_timer()

import matplotlib.pyplot as plt
plt.rcParams['contour.negative_linestyle'] = 'dashed'
import numpy as np

from niwqg import QGModel
from niwqg import InitialConditions as ic
import cmocean

from pyspec import spectrum

plt.close('all')

# parameters
nx = 128
L = 2*np.pi

# eddy parameters
k0 = 15*(2*np.pi/L)
L0 = 2*np.pi/k0


# energy input
epsilon = .00001

# dissipation
mu = 0.005
Tmu = 1./mu
dt = .1
tmax = 15*Tmu

kf = 7
dkf = 1

path = "128/FD_diss"
#path = "512/lamb/large_amp"
m = QGModel.Model(L=L,nx=nx, tmax = tmax,dt = dt, twrite=int(20*dt),
                    nu4=0e9,nu4c=0.e9, nuc = 0,mu=mu, use_filter=True,save_to_disk=True,
                    tsave_snapshots=25,path=path,
                    U = 0., tdiags=1, beta = 0e-10,passive_scalar=False,
                    forcing=True,wavenumber_forcing=kf,width_forcing=dkf,epsilon_q=epsilon )

m.set_q(np.zeros([m.nx]*2))
m._invert()

# run the model
m.run()


# diagnostics
time = m.diagnostics['time']['value']
KE_qg = m.diagnostics['ke_qg']['value']
ENS_qg = m.diagnostics['ens']['value']
ep_psi = m.diagnostics['ep_psi']['value']
chi_q =  m.diagnostics['chi_q']['value']

energy_input = m.diagnostics['energy_input']['value']

dt = time[1]-time[0]
dKE = np.gradient(KE_qg,dt)

plt.figure(figsize=(12,6))
lw, alp = 3.,.5
plt.plot(time*mu,np.ones(time.size)*epsilon/mu/2,'r--',linewidth=1.25)
plt.plot(time*mu,epsilon/(2*mu)*(1-np.exp(-2*mu*time)),'k--',linewidth=1.25)
plt.plot(time*mu,KE_qg,
            linewidth=lw,alpha=alp)
#plt.plot(time/Te,Te*np.ones(time.size)*epsilon/KE_qg[0],'r--')
plt.xlabel(r"Time [$t \times \mu$]")
plt.ylabel(r'Energy [$K$]')

plt.figure(figsize=(12,6))
plt.plot(time*mu,epsilon*np.ones_like(time),'r--',linewidth=1.25,label=r'$\epsilon$')
plt.plot(time*mu,2*mu*KE_qg,label=r'$2\mu K$')
plt.plot(time*mu,energy_input,label=r'$-\langle \psi$ force$\rangle$')
plt.plot(time*mu,dKE,label=r'$\dot K$')

plt.legend()
plt.xlabel(r"Time [$t \times \mu$]")
plt.ylabel(r'Power')


plt.figure()
plt.plot(time*mu,energy_input,label=r'$-\langle \psi$ force$\rangle$')
plt.plot(time*mu,epsilon*np.ones_like(time),'r--',linewidth=1.25,label=r'$\epsilon$')
plt.plot(time*mu,2*mu*KE_qg,label=r'$2\mu K$')


# calculate spectrum
E = 0.5 * np.abs(m.wv*m.ph)**2
ki, Er = spectrum.calc_ispec(m.kk, m.ll, E, ndim=2)


# plt.figure(figsize=(12,6))
# plt.plot(time/Te,Te*ep_psi/KE_qg[0], label=r'Dissipation $-\epsilon_\psi$',
#             linewidth=lw,alpha=alp)
# plt.plot(time/Te,Te*energy_input/KE_qg[0], label=r'Forcing $W$',
#                         linewidth=lw,alpha=alp)
# plt.plot(time/Te,Te*(energy_input+ep_psi)/KE_qg[0], label=r'Forcing+Dissipation',
#                         linewidth=lw,alpha=alp)
# plt.plot(time/Te,Te*dKE/KE_qg[0],'k--',label=r'KE tendency $\dot K_e$',
#             linewidth=lw,alpha=alp)
# #plt.plot(time/Te,Te*np.ones(time.size)*epsilon/KE_qg[0],'r--')
# plt.xlabel(r"Time [$t \times U_0 k_0$]")
# plt.ylabel(r'Power $[\dot E \times {2 k_0}/{U_0} ]$')
# plt.legend(loc=4)
#
# plt.figure(figsize=(12,6))
# plt.plot(time/Te,Te*ep_c/C2[0], label=r'KE dissipation $-\epsilon_\psi$',
#             linewidth=lw,alpha=alp)
# plt.plot(time/Te,Te*dC2/C2[0],'k--',label=r'KE tendency $\dot K_e$',
#             linewidth=lw,alpha=alp)
# plt.xlabel(r"Time [$t \times U_0 k_0$]")
# plt.ylabel(r'Power $[\dot E \times {2 k_0}/{U_0} ]$')
# plt.legend(loc=4)

stop = timeit.default_timer()
print("Time elapsed: %3.2f seconds" %(stop - start))
