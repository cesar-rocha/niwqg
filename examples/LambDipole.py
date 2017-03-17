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
from scipy import integrate

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
                m=m,N=N,f=f0, twrite=int(1*Tf/dt),
                nu4=5e11,nu4w=0e10, nu=20,nuw=50e0, mu=0.e-7,muw=0e-7,use_filter=False,
                U =-U, tdiags=1,save_to_disk=False, dealias=False,)

# initial conditions
q = ic.LambDipole(m, U=U,R = 2*np.pi/k0)
phi = (np.ones_like(q) + 1j)*u0/np.sqrt(2)
#phi = ic.WavePacket(m, k=k0*2, l=0, R=2*np.pi/k0)
m.set_q(q)
m.set_phi(phi)

# run the model
m.run()

# get diagnostics
time = m.diagnostics['time']['value']
KE_qg = m.diagnostics['ke_qg']['value']
Ke = m.diagnostics['Ke']['value']
Pw = m.diagnostics['Pw']['value']
Kw = m.diagnostics['Kw']['value']
PE_niw = m.diagnostics['pe_niw']['value']
KE_niw = m.diagnostics['ke_niw']['value']
ENS_qg = m.diagnostics['ens']['value']

g1 = m.diagnostics['gamma_r']['value']
g2 = m.diagnostics['gamma_a']['value']
x1 = m.diagnostics['xi_r']['value']
x2 = m.diagnostics['xi_a']['value']

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

res_ke = dKE-(-g1-g2+x1+x2+ep_psi)
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
plt.plot(time/Te,Te*dPE/KE0,'k--',label=r'PE NIW tendency $\dot P_w$',linewidth=lw,alpha=alp)
plt.legend(loc=1,ncol=2)
plt.xlabel(r"Time [$t \times U_0 k_0$]")
plt.ylabel(r'Power $[\dot E \times {2 k_0}/{U_0} ]$')

ax = fig.add_subplot(224)
plt.plot(time/Te,-Te*g1/KE0,label=r'Refrac. conversion $\Gamma_r$',linewidth=lw,alpha=alp)
plt.plot(time/Te,-Te*g2/KE0,label=r'Adv. conversion $\Gamma_a$',linewidth=lw,alpha=alp)
plt.plot(time/Te,Te*ep_psi/KE0,label=r'KE QG diss. $\epsilon_\psi$',linewidth=lw,alpha=alp)
plt.plot(time/Te,Te*x1/KE0,label=r'KE QG diss. $\Xi_r$',linewidth=lw,alpha=alp)
plt.plot(time/Te,Te*x2/KE0,label=r'KE QG diss. $\Xi_a$',linewidth=lw,alpha=alp)
plt.plot(time/Te,Te*(-g1-g2+x1+x2+ep_psi)/KE0,label=r'$(-\Gamma_r-\Gamma_a+\Xi_r+\Xi_a+\epsilon_\psi)$',linewidth=lw,alpha=alp)
plt.plot(time/Te,Te*dKE/KE0,'k--',label=r'KE NIW tendency $\dot K_e$',linewidth=lw,alpha=alp)
plt.xlabel(r"Time [$t \times U_0 k_0$]")
plt.ylabel(r'Power $[\dot E \times {2 k_0}/{U_0} ]$')
plt.legend(loc=4)

plt.savefig("energy_budget_filter_diss.png")

## calculate relative contribution
i = g1.size

KE, PE = KE_qg[i-1]-KE_qg[0], PE_niw[i-1]-PE_niw[0]

G1, G2 = integrate.simps(y=g1[:i],x=time[:i]),  integrate.simps(y=g2[:i],x=time[:i])
X1 = -integrate.simps(y=x1[:i],x=time[:i])
X2 = -integrate.simps(y=x2[:i],x=time[:i])
G1_Pw, G2_Pw = G1/PE, G2/PE
G1_Ke, G2_Ke, X1_Ke, X2_Ke = G1/KE, G2/KE, X1/KE, X2/KE
G_Ke = G1_Ke+G2_Ke
CHI_Pw = integrate.simps(y=chi_phi[:i],x=time[:i])/PE
EP_Ke = -integrate.simps(y=ep_psi[:i],x=time[:i])/KE

RES_PE = 1-(G1_Pw+G2_Pw+CHI_Pw)
RES_KE = 1+(G1_Ke+G2_Ke+X1_Ke+X2_Ke+EP_Ke)

res = (dKE+dPE+ep_psi+chi_phi)
RES =  integrate.simps(y=res,x=time)/KE

stop = timeit.default_timer()
print("Time elapsed: %3.2f seconds" %(stop - start))

from pyspec import spectrum

dx = L/nx
specq = spectrum.TWODimensional_spec(m.q,d1=dx,d2=dx)
specqpsi = spectrum.TWODimensional_spec(m.q_psi,d1=dx,d2=dx)

lapq = m.ifft(-m.wv2*m.qh).real
specep = spectrum.TWODimensional_spec(m.nu*m.p*lapq,d1=dx,d2=dx)

lap2phi =  m.ifft(-m.wv4*m.phih)
lapphi =  m.ifft(-m.wv2*m.phih)
specchi = spectrum.TWODimensional_spec(m.nuw*np.abs(lapphi)**2,d1=dx,d2=dx)


fig = plt.figure(figsize=(10,7.5))

imax = -1
imax = specq.ki.size*2//3

ax = fig.add_subplot(221)
ax.loglog(specq.ki[:imax]/k0,specq.ispec[:imax])
ax.set_title(r"Potential Enstrophy, $q^2$")

ax2 = fig.add_subplot(222)
ax2.loglog(specqpsi.ki[:imax]/k0,specqpsi.ispec[:imax])
ax2.set_title(r"Enstrophy, $(\nabla^2 \psi)^2$")

ax3 = fig.add_subplot(223)
ax3.loglog(specep.ki[:imax]/k0,specep.ispec[:imax])
ax3.set_title(r"q-diss. $\nu_e (\nabla^2 q) (\nabla^2 \psi) $")

ax4 = fig.add_subplot(224)
ax4.loglog(specchi.ki[:imax]/k0,specchi.ispec[:imax])
ax4.set_title(r"phi-diss. $\nu_w |\nabla(\nabla^2\phi)|^2$")

# p, pw, and ppsi
#p = m.ifft(m.ph).real
#pw = m.ifft(-m.phw).real
# p, pw, pv = m.p, m.pw, m.pv
# ppsi = m.ifft(-m.wv2i*m.qh_psi).real
# qh_psi = m.q-m.qwh
# ppsi = m.ifft(-m.wv2i*qh_psi).real
#
# pv_2 = p-pw
# pw_2 = p - pv_2
# pw_3 = m.ifft(-m.wv2i*m.qwh).real


#ppsi = p - pw
