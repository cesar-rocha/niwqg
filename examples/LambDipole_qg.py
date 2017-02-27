"""
    Barotropic, f-plane solution with initial condition
        given by the Lamb dipole.

 This example runs in about 3 sec on a MacPro 2.2 GHz Intel Core i7,
    16 GB 1600 MHz DDR3.
"""
import timeit
start = timeit.default_timer()

import matplotlib.pyplot as plt
plt.rcParams['contour.negative_linestyle'] = 'dashed'
import numpy as np

#from CoupledModel import Model
from niwqg import QGModel as Model
from niwqg import InitialConditions as ic

plt.close('all')

# parameters
nx = 128
L = 2*np.pi*200e3

# eddy parameters
k0 = 10*(2*np.pi/L)
L0 = 2*np.pi/k0

# initial conditions
U = 1.e-1

# simulation parameters
Te = (U*k0)**-1

dt = .05*Te
tmax = 10*Te

path = "128/lamb/moderate_filter"
#path = "512/lamb/large_amp"
m = Model.QGModel(L=L,nx=nx, tmax = tmax,dt = dt, twrite=int(0.1*Te/dt),
                    nu4=7.5e8, use_filter=False,save_to_disk=False,
                    tsave_snapshots=5,path=path,
                    U =-U, tdiags=1, beta = 0.)

#q = McWilliams1984(m,k0=k0,E=U0**2/2)
q = ic.LambDipole(m, U=U,R = 2*np.pi/k0)

m.set_q(q)

m._invert()

# plot initial condition
xqmin, yqmin = [],[]
xqmax, yqmax = [],[]
R = 2*np.pi/k0
r = k0*L/2/np.pi

# run the model
m.run()

time = m.diagnostics['time']['value']
KE_qg = m.diagnostics['ke_qg']['value']
ENS_qg = m.diagnostics['ens']['value']
ep_psi = m.diagnostics['ep_psi']['value']
chi_q =  m.diagnostics['chi_q']['value']
dt = time[1]-time[0]
dKE = np.gradient(KE_qg,dt)

plt.figure(figsize=(12,6))
lw, alp = 3.,.5
plt.plot(time/Te,Te*ep_psi/KE_qg[0], label=r'KE dissipation $-\epsilon_\psi$',
            linewidth=lw,alpha=alp)
plt.plot(time/Te,Te*dKE/KE_qg[0],'k--',label=r'KE tendency $\dot K_e$',
            linewidth=lw,alpha=alp)
plt.xlabel(r"Time [$t \times U_0 k_0$]")
plt.ylabel(r'Power $[\dot E \times {2 k_0}/{U_0} ]$')
plt.legend(loc=4)

stop = timeit.default_timer()
print("Time elapsed: %3.2f seconds" %(stop - start))
