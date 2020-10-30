from boutdata import collect
import numpy as np

def loading_data(path):

    """
    collecting the data from the BOUT++ simulations 
    normalizing to SI units (not all)
    input:
    path: to BOUT++ dmp files

    return:
    n, Pe,n0,T0,Lx,Lz,B0,phi,dt, t_array
    """
    e = 1.602176634e-19    

    n0 = collect("Nnorm", path=path, info=False)
    T0 = collect("Tnorm", path=path, info=False)
    Cs0 = collect('Cs0',path=path, info=False)
    wci = collect("Omega_ci", path=path, info=False)
    rhos = collect('rho_s0', path=path, info=False)

    n = collect("Ne", path=path, info=False) * n0 
    Pe = collect("Pe", path=path, info=False) * n0 * T0 * e
    Pi = collect("Pi", path=path, info=False) * n0 * T0 * e
    # Rzrad = collect("Rzrad",path=path, info=False)
    
    phi = collect("phi", path=path, info=False) 
    psi =collect("psi", path=path, info=False)
    psi_zero=collect("psi_zero",path=path,info=False)
    
    t_array = collect("t_array", path=path, info=False)/wci
    dt = (t_array[1] - t_array[0]) 

   # R0 = collect('R0', path=path, info=False) * rhos
    B0 = collect('Bnorm', path=path, info=False)
    dx = collect('dx', path=path, info=False) * rhos * rhos
    dx=dx[0, 0]
    dz = collect('dz', path=path, info=False)
    dy = collect('dy', path=path, info=False) 
    dy =dy[0,0]
   
    external_field=collect('external_field',path=path,info=False)    
    beta_e= collect('beta_e',path=path,info=False)
    Vi=collect('Vi',path=path,info=False) * Cs0 
    NVi=collect('NVi',path=path,info=False) * n0 * Cs0
    Jpar=collect('Jpar',path=path,info=False) * n0 * Cs0
    Vort=collect('Vort',path=path,info=False) * wci
    VePsi=collect('VePsi',path=path,info=False)  / Cs0 / e *T0   


    np.savez(path+'data.npz', n=n, Pe=Pe,Pi=Pi,n0=n0,T0=T0,B0=B0,phi=phi,dt=dt, t_array=t_array,psi=psi,psi_zero=psi_zero,external_field=external_field,beta_e=beta_e,Vi=Vi,NVi=NVi,Jpar=Jpar,dx=dx,dz=dz,dy=dy,Vort=Vort,VePsi=VePsi)

    return n, Pe,Pi,n0,T0,B0,phi,dt, t_array,psi,psi_zero,external_field,beta_e,Vi,NVi,Jpar,dx,dz,dy,Vort,VePsi

def load_npz(path):
    f = np.load(path+"/data.npz")
    n=f['n']
    Pe=f['Pe']
    Pi=f['Pi']
    n0=f['n0']
    T0=f['T0']
    B0=f['B0']
    phi=f['phi']
    dt=f['dt']  
    t_array=f['t_array']
    #Rzrad=f['Rzrad']
    psi=f['psi']
    psi_zero=f['psi_zero']
    external_field=f['external_field']
    beta_e=f['beta_e']
    Vi=f['Vi']
    Jpar=f['Jpar']
    dx=f['dx']
    dz=f['dz']
    Vort=f['Vort']
    VePsi=f['VePsi']


    return n, Pe,Pi,n0,T0,B0,phi,dt, t_array,psi,psi_zero,external_field,beta_e,Vi,Jpar,dx,dz,Vort,VePsi



def TotalEnergy(path,phi, n0, Vort, VePsi, NVi, Vi, Jpar,Pe,Pi,dx,dy,dz):



    m_i = 1.672621898e-27


    E_dichte= -m_i* (phi + Pi/n0 )* Vort + m_i * Vi * NVi +3/2 * (Pi + Pe) + Jpar * VePsi
    
    E_dichte2= -m_i* (phi + Pi/n0 )* Vort + m_i * Vi * NVi +3/2 * (Pi + Pe)

    
    E1=np.trapz(np.trapz(np.trapz(E_dichte,dx=dz),dx=dy),dx=dx)

    E2=np.trapz(np.trapz(np.trapz(E_dichte2,dx=dz),dx=dy),dx=dx)


    plt.rc('font', family='Serif')
    plt.figure(figsize=(8,4.5))
    plt.plot( E1,'v', label=r'Energy')

    plt.grid(alpha=0.5)
    plt.xlabel(r't ', fontsize=18)
    plt.ylabel(r'Energy', fontsize=18)
    plt.tick_params('both', labelsize=14)
    plt.tight_layout()
    plt.legend(fancybox=True, loc='upper left', framealpha=0, fontsize=12)
    plt.savefig(path+'/E1.png', dpi=300)
    plt.show()

    plt.rc('font', family='Serif')
    plt.figure(figsize=(8,4.5))
    plt.plot( E2,'v', label=r'Energy')

    plt.grid(alpha=0.5)
    plt.xlabel(r't ', fontsize=18)
    plt.ylabel(r'Energy', fontsize=18)
    plt.tick_params('both', labelsize=14)
    plt.tight_layout()
    plt.legend(fancybox=True, loc='upper left', framealpha=0, fontsize=12)
    plt.savefig(path+'/E2.png', dpi=300)
    plt.show()

    return E1,E2


def TotalEnergy2(path,T0,B0,phi, Pi,Pe,mi,n0,dz, dx,beta_e,psi,Vi,jpar,n):

    qe = 1.602176634e-19
    m_i = 1.672621898e-27
    m_e= 9.10938356e-31

    
    
    Nabla_xPhi=np.gradient(phi,dx,axis=1)

    Nabla_zPhi=np.gradient(phi,dz,axis=3)

    NablaP_Phi=Nabla_xPhi+Nabla_zPhi


    Nabla_xPi=np.gradient(Pi,dx,axis=1)

    Nabla_zPi=np.gradient(Pi,dz,axis=3)

    NablaP_Pi=Nabla_xPi+Nabla_zPi
   

    Nabla_xpsi=np.gradient(psi,dx,axis=1)

    Nabla_zpsi=np.gradient(psi,dz,axis=3)

    NablaP_psi=Nabla_xpsi+Nabla_zpsi

    E_dichte = m_i * n0 / 2 * (np.abs(NablaP_Phi/B0 + NablaP_Pi / (qe*n0*B0)))**2 + 0.5 * m_i * n * Vi**2 + 3/2*(Pe+Pi) + 1/4 * beta_e * (np.abs(NablaP_psi))**2 + 0.5*m_e/(m_i*n)*jpar**2

    E=np.trapz(np.trapz(np.trapz(E_dichte,dx=dz),dx=dy),dx=dx)

    plt.rc('font', family='Serif')
    plt.figure(figsize=(8,4.5))
    plt.plot( E,'v', label=r'Energy')

    plt.grid(alpha=0.5)
    plt.xlabel(r't ', fontsize=18)
    plt.ylabel(r'Energy', fontsize=18)
    plt.tick_params('both', labelsize=14)
    plt.tight_layout()
    plt.legend(fancybox=True, loc='upper left', framealpha=0, fontsize=12)
    plt.savefig(path+'/E3.png', dpi=300)
    plt.show()

    return E
