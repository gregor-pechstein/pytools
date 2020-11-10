from boutdata import collect
import numpy as np
import matplotlib.pyplot as plt

def loading_data(path,version):

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

    
    phi = collect("phi", path=path, info=False) * T0 

    
    t_array = collect("t_array", path=path, info=False)/wci
    dt = (t_array[1] - t_array[0]) 

   # R0 = collect('R0', path=path, info=False) * rhos
    B0 = collect('Bnorm', path=path, info=False)
    dx = collect('dx', path=path, info=False) * rhos * rhos
    dx=dx[0, 0]
    dz = collect('dz', path=path, info=False)
    dy = collect('dy', path=path, info=False) 
    dy =dy[0,0]
   
    
    beta_e= collect('beta_e',path=path,info=False)
  
    Jpar=collect('Jpar',path=path,info=False) * n0 * Cs0
    Vort=collect('Vort',path=path,info=False) * wci
 
   
    if (version==2):
        Pi = collect("Pi", path=path, info=False) * n0 * T0 * e
        # Rzrad = collect("Rzrad",path=path, info=False)
        psi =collect("psi", path=path, info=False)
        psi_zero=collect("psi_zero",path=path,info=False)
        external_field=collect('external_field',path=path,info=False)
        Vi=collect('Vi',path=path,info=False) * Cs0 
        NVi=collect('NVi',path=path,info=False) * n0 * Cs0
        VePsi=collect('VePsi',path=path,info=False)  / Cs0 / e *T0  

        np.savez(path+'data.npz', n=n, Pe=Pe,Pi=Pi,n0=n0,T0=T0,B0=B0,phi=phi,dt=dt, t_array=t_array,psi=psi,psi_zero=psi_zero,external_field=external_field,beta_e=beta_e,Vi=Vi,NVi=NVi,Jpar=Jpar,dx=dx,dz=dz,dy=dy,Vort=Vort,VePsi=VePsi)

        return n, Pe,Pi,n0,T0,B0,phi,dt, t_array,psi,psi_zero,external_field,beta_e,Vi,NVi,Jpar,dx,dz,dy,Vort,VePsi    
    else:
        np.savez(path+'data.npz', n=n, Pe=Pe,n0=n0,T0=T0,B0=B0,phi=phi,dt=dt, t_array=t_array,beta_e=beta_e,Jpar=Jpar,dx=dx,dz=dz,dy=dy,Vort=Vort)

        return n, Pe,n0,T0,B0,phi,dt, t_array,beta_e,Jpar,dx,dz,dy,Vort


def load_npz(path,version):
    f = np.load(path+"/data.npz")
    n=f['n']
    Pe=f['Pe']
    n0=f['n0']
    T0=f['T0']
    B0=f['B0']
    phi=f['phi']
    dt=f['dt']  
    t_array=f['t_array']
    beta_e=f['beta_e']

    Jpar=f['Jpar']
    dx=f['dx']
    dz=f['dz']
    dy=f['dy']
    Vort=f['Vort']


    if (version==2):
        Pi=f['Pi']
        #Rzrad=f['Rzrad']
        psi=f['psi']
        psi_zero=f['psi_zero']
        external_field=f['external_field']
        Vi=f['Vi']
        NVi=f['NVi']
        VePsi=f['VePsi']
        return n, Pe,Pi,n0,T0,B0,phi,dt, t_array,psi,psi_zero,external_field,beta_e,Vi,Jpar,dx,dz,Vort,VePsi

    else:
        return n, Pe,n0,T0,B0,phi,dt, t_array,beta_e,Jpar,dx,dz,dy,Vort




def TotalEnergy1(path,T0,B0,phi, Pe,dz, dx,beta_e,jpar,n):

    qe = 1.602176634e-19
    m_i = 1.672621898e-27
    m_e= 9.10938356e-31



    Grad_perpPhi=np.stack(np.gradient(phi,dx,dz,axis=[1,3]))

    E_perpKin =0.5 * n *m_i /B0**2* np.sum((Grad_perpPhi)**2,axis=0)


    E_Term=3/2*Pe


    E_dichte=E_perpKin+E_Term
    E=np.trapz(np.trapz(E_dichte,dx=dz),dx=dx,axis=1)


    ETerm_f=3/2*np.abs(np.fft.fft(Pe,axis=-1))
    EperpKin_f=np.fft.fft(E_perpKin,axis=-1)

    E_fftsum= EperpKin_f+ETerm_f
    E_fft=np.trapz(E_fftsum,dx=dx, axis=1)
    

    plt.rc('font', family='Serif')
    plt.figure(figsize=(8,4.5))
    plt.plot( E, label=r'Energy')

    plt.grid(alpha=0.5)
    plt.xlabel(r't ', fontsize=18)
    plt.ylabel(r'Energy', fontsize=18)
    plt.tick_params('both', labelsize=14)
    plt.tight_layout()
    plt.legend(fancybox=True, loc='upper left', framealpha=0, fontsize=12)
    plt.savefig(path+'/E.png', dpi=300)
    plt.show()

    np.savez(path+'Energy.npz', E=E)

    return E, E_dichte,E_Term,E_perpKin



def TotalEnergy(path,T0,B0,phi, Pi,Pe,n0,dz, dx,beta_e,psi,Vi,jpar,n):

    qe = 1.602176634e-19
    m_i = 1.672621898e-27
    m_e= 9.10938356e-31

        
    Grad_perpPhi=np.stack(np.gradient(phi,dx,dz,axis=[1,3]))
    Grad_perpPi=np.stack(np.gradient(Pi,dx,dz,axis=[1,3]))
    Grad_perpPsi=np.stack(np.gradient(psi,dx,dz,axis=[1,3]))

    E_perpKin = m_i * n0 *0.5 * np.sum((Grad_perpPhi/B0 + Grad_perpPi/(qe*n0*B0))**2,axis=0)

    E_parIon=0.5 * m_i * n * Vi**2
    E_Term=3/2*(Pe+Pi)
    E_Field= 1/4 * beta_e * np.sum(Grad_perpPsi**2,axis=0) + 0.5*m_e/(m_i*n)*jpar**2

    E_dichte=E_perpKin+E_parIon+E_Term+E_Field
    E=np.trapz(np.trapz(np.trapz(E_dichte,dx=dz),dx=dy),dx=dx)


    plt.rc('font', family='Serif')
    plt.figure(figsize=(8,4.5))
    plt.plot( E, label=r'Energy')

    plt.grid(alpha=0.5)
    plt.xlabel(r't ', fontsize=18)
    plt.ylabel(r'Energy', fontsize=18)
    plt.tick_params('both', labelsize=14)
    plt.tight_layout()
    plt.legend(fancybox=True, loc='upper left', framealpha=0, fontsize=12)
    plt.savefig(path+'/E.png', dpi=300)
    plt.show()

    np.savez(path+'Energy.npz', E=E)

    return E, E_dichte,E_Term,E_Field,E_parIon,E_perpKin



def DifusionCoeficent(path,phi,n,B0,dx, dy, dz,Dim2):

    if (Dim2==True):
        dn_dr=np.zeros(n.shape)
        phi_dz=np.zeros(n.shape)
        for t in np.arange(0,n.shape[0]): 
             dn_dr[t,...] = np.abs( np.gradient(n[t,...],dx,axis=0))
             phi_dz[t,...] =np.abs( np.gradient(phi[t,...],dz,axis=2))

        dn_dr_mean=np.mean(dn_dr[50:,...],axis=0)
        V_ExB_n_mean= np.mean(phi_dz[50:,...]*n[50:,...]/B0,axis=0)

    #Grad_Phi=np.stack( np.gradient(phi,dx,dz,axis=[1,3]))
    #d_n=np.stack(np.gradient(n,dx,dz,axis=[1,3]))  
      #d_n=np.gradient(np.mean(n[50:,:,:,:],axis=0),dx,axis=1))
      #Grad_Phi= np.gradient(phi,dz,axis=3)
    else:    
        Grad_Phi=np.stack( np.gradient(phi,dx,dy,dz,axis=[1,2,3]))
        d_n=np.stack(np.gradient(n,dx,dy,dz,axis=[1,2,3]))

    V_ExB=Grad_Phi/B0
    # absV_ExB=np.sqrt(V_ExB[0,:,:,:,:]**2+V_ExB[1,:,:,:,:]**2)

    # absd_n=np.sqrt(d_n[0,:,:,:,:]**2+d_n[1,:,:,:,:]**2)


    mean_Vn=np.mean(np.multiply(V_ExB[50:,:,:,:],n[50:,:,:,:]),axis=0)
    mean_dn=np.mean(d_n[50:,:,:,:],axis=0)
    D=np.divide(mean_Vn,mean_dn)
   # D= np.mean((phi_dz[50:,...]*n[50:,...])/(B0*dn_dr[50:,...]),axis=0) 
   # D=np.mean(np.multiply(V_ExB,n),axis=1)/np.mean(d_n,axis=1)

    plt.rc('font', family='Serif')
    plt.contourf(D[:,0,:].T);
    plt.colorbar();
    plt.xlabel(r'z ', fontsize=18)
    plt.ylabel(r'x', fontsize=18)
    plt.tick_params('both', labelsize=14)
    plt.tight_layout()
    plt.savefig(path+'/Difusion.png', dpi=300)
    plt.show()

    np.savez(path+'Difusion.npz', D=D)
    
    return D




def Fourier_spectrum(E_Term,E_Field,E_parIon,E_perpKin,Dim2,dx):

     if (Dim2==True):
         ETerm_f=np.fft.fft(E_Term,axis=-1)
         EperpKin_f=np.fft.fft(E_perpKin,axis=-1)

         E_fftsum= EperpKin_f+ETerm_f
     else:
         ETerm_f=np.fft.fft2(E_Term,axes=(-2,-1))
         EField_f=np.fft.fft2(E_Field,axes=(-2,-1))
         EparIon_f=np.fft.fft2(E_parIon,axes=(-2,-1))
         EperpKin_f=np.fft.fft2(E_perpKin,axes=(-2,-1))
    
         E_fftsum= EperpKin_f+EparIon_f+ETerm_f+EField_f
   
     E_fft=np.trapz(E_fftsum,dx=dx, axis=1)

     return E_fft




def TotalEnergy_alternativ(path,phi, n0, Vort, VePsi, NVi, Vi, Jpar,Pe,Pi,dx,dy,dz):



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
