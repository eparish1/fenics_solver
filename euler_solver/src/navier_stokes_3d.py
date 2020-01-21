from euler_3d import  *
import numpy as np
class navierStokesEqns3D:
  eulerEq = eulerEqns3D()
  evalF = eulerEq.evalF
  evalF_Strong = eulerEq.evalF_Strong
  applyJ = eulerEq.applyJ
  applyJT = eulerEq.applyJT
  nvars = 5

  mu = 1./100.*0.2
  def containerMatVec(self,M,Q):
    ''' 
    computes the matrix vector product MQ
    '''
    N,K = np.shape(M)
    print(N,K)
    MQ = [ 0 ]*K
    for i in range(0,N):
      for k in range(0,K):
        if(M[i][k] != None):
          MQ[i] = MQ[i] + M[i][k]*Q[k]
    return MQ

  def evalGs(self,Q):
    gamma = 1.4
    Pr = 0.72
    ## initialize containers
    GC = [[None for i in range(3)] for j in range(3)] 
 
    G11 = [[0. for i in range(self.nvars)] for j in range(self.nvars)] 
    G21 = [[0. for i in range(self.nvars)] for j in range(self.nvars)] 
    G31 = [[0. for i in range(self.nvars)] for j in range(self.nvars)] 
    G12 = [[0. for i in range(self.nvars)] for j in range(self.nvars)] 
    G22 = [[0. for i in range(self.nvars)] for j in range(self.nvars)] 
    G32 = [[0. for i in range(self.nvars)] for j in range(self.nvars)] 
    G13 = [[0. for i in range(self.nvars)] for j in range(self.nvars)] 
    G23 = [[0. for i in range(self.nvars)] for j in range(self.nvars)] 
    G33 = [[0. for i in range(self.nvars)] for j in range(self.nvars)] 

    v1 = Q[1]/Q[0]
    v2 = Q[2]/Q[0]
    v3 = Q[3]/Q[0]
    E = Q[4]/Q[0]
    vsqr = v1**2 + v2**2 + v3**2
   
    mu_by_rho = self.mu/Q[0]
    G11[1][0] = -4./3.*v1*mu_by_rho
    G11[1][1] = 4./3.*mu_by_rho
    G11[2][0] = -v2*mu_by_rho
    G11[2][2] = mu_by_rho
    G11[3][0] = -v3*mu_by_rho
    G11[3][3] = mu_by_rho
    G11[4][0] = -(4./3.*v1**2 + v2**2 + v3**2 + gamma/Pr*(E - vsqr) )*mu_by_rho
    G11[4][1] = (4./3. - gamma/Pr)*v1*mu_by_rho
    G11[4][2] = (1. - gamma/Pr)*v2*mu_by_rho
    G11[4][3] = (1. - gamma/Pr)*v3*mu_by_rho
    G11[4][4] = gamma/Pr*mu_by_rho
  
    G21[1][0] = G11[2][0]#-v2*mu_by_rho
    G21[1][2] = mu_by_rho
    G21[2][0] = 2./3.*v1*mu_by_rho
    G21[2][1] = -2./3.*mu_by_rho
    G21[4][0] = -1./3.*v1*v2*mu_by_rho
    G21[4][1] = -2./3.*v2*mu_by_rho
    G21[4][2] = v1*mu_by_rho

    G31[1][0] = G11[3][0]#-v3*mu_by_rho
    G31[1][3] = mu_by_rho
    G31[3][0] = G21[2][0]#2./3.*v1*mu_by_rho
    G31[3][1] = G21[2][1]#-2./3.*mu_by_rho
    G31[4][0] = -1./3.*v1*v3*mu_by_rho
    G31[4][1] = 2./3.*G11[3][0]#-2./3.*v3*mu_by_rho
    G31[4][3] = G21[4][2]#v1*mu_by_rho



    G12[1][0] = 2./3.*v2*mu_by_rho
    G12[1][2] = -2./3.*mu_by_rho
    G12[2][0] = -v1*mu_by_rho
    G12[2][1] = mu_by_rho
    G12[4][0] = -1./3.*v1*v2*mu_by_rho
    G12[4][1] = v2*mu_by_rho
    G12[4][2] = -2./3.*v1*mu_by_rho

    G22[1][0] = G12[2][0]#-v1*mu_by_rho
    G22[1][1] = mu_by_rho
    G22[2][0] = -4./3.*v2*mu_by_rho
    G22[2][2] = 4./3.*mu_by_rho
    G22[3][0] = -v3*mu_by_rho
    G22[3][3] = mu_by_rho
    G22[4][0] = -(v1**2 + 4./3.*v2**2 + v3**2 + gamma/Pr*(E - vsqr) )*mu_by_rho
    G22[4][1] = (1. - gamma/Pr)*v1*mu_by_rho
    G22[4][2] = (4./3. - gamma/Pr)*v2*mu_by_rho
    G22[4][3] = (1. - gamma/Pr)*v3*mu_by_rho
    G22[4][4] = gamma/Pr*mu_by_rho

    G32[2][0] = G22[3][0]#-v3*mu_by_rho
    G32[2][3] = mu_by_rho
    G32[3][0] = G12[1][0]#2./3.*v2*mu_by_rho
    G32[3][2] = G12[1][2]#-2./3.*mu_by_rho
    G32[4][0] = -1./3.*v2*v3*mu_by_rho
    G32[4][2] = -2./3.*v3*mu_by_rho
    G32[4][3] = G12[4][1]#v2*mu_by_rho


    G13[1][0] = 2./3.*v3*mu_by_rho
    G13[1][3] = -2./3.*mu_by_rho
    G13[3][0] = -v1*mu_by_rho
    G13[3][1] = mu_by_rho
    G13[4][0] = -1./3.*v1*v3*mu_by_rho
    G13[4][1] = v3*mu_by_rho
    G13[4][3] = -2./3.*v1*mu_by_rho

    G23[2][0] = G13[1][0]#2./3.*v3*mu_by_rho
    G23[2][3] = -2./3.*mu_by_rho
    G23[3][0] = -v2*mu_by_rho
    G23[3][2] = mu_by_rho
    G23[4][0] = -1./3.*v2*v3*mu_by_rho
    G23[4][2] = G13[4][1]#v3*mu_by_rho
    G23[4][3] = -2./3.*v2*mu_by_rho

    G33[1][0] = -v1*mu_by_rho
    G33[1][1] = mu_by_rho
    G33[2][0] = -v2*mu_by_rho
    G33[2][2] = mu_by_rho
    G33[3][0] = -4./3.*v3*mu_by_rho
    G33[3][3] = 4./3.*mu_by_rho
    G33[4][0] = -(v1**2 + v2**2 + 4./3.*v3**2 + gamma/Pr*(E - vsqr) )*mu_by_rho
    G33[4][1] = (1. - gamma/Pr)*v1*mu_by_rho
    G33[4][2] = (1. - gamma/Pr)*v2*mu_by_rho
    G33[4][3] = (4./3. - gamma/Pr)*v3*mu_by_rho
    G33[4][4] = gamma/Pr*mu_by_rho

    GC[0][0] = G11
    GC[0][1] = G12
    GC[0][2] = G13
    GC[1][0] = G21
    GC[1][1] = G22
    GC[1][2] = G23
    GC[2][0] = G31
    GC[2][1] = G32
    GC[2][2] = G33
    return GC


  def evalF_Viscous(self,Q,Qx):
    gamma = 1.4
    Pr = 0.72
    ## ->  v_x = 1/rho d/dx(rho v) - rho v /rho^2 rho_x
    ux = 1./Q[0]*(Qx[1] - Q[1]/Q[0]*Qx[0])
    vx = 1./Q[0]*(Qx[2] - Q[2]/Q[0]*Qx[0])
    wx = 1./Q[0]*(Qx[3] - Q[3]/Q[0]*Qx[0])
    ## ->  u_y = 1/rho d/dy(rho u) - rho u /rho^2 rho_y
    uy = 1./Q[0]*(Uy[1] - Q[1]/Q[0]*Uy[0])
    vy = 1./Q[0]*(Uy[2] - Q[2]/Q[0]*Uy[0])
    ## ->  u_z = 1/rho d/dz(rho u) - rho u /rho^2 rho_z
    uz = 1./Q[0]*(Uz[1] - Q[1]/Q[0]*Uz[0])
    wz = 1./Q[0]*(Uz[3] - Q[3]/Q[0]*Uz[0])
    ## -> (kT)_x =d/dx[ (mu gamma)/Pr*(E - 1/2 v^2 ) ]
    ## ->        =mu gamma/Pr *[ dE/dx - 0.5 d/dx(v1^2 + v2^2) ]
    ## ->        =mu gamma/Pr *[ dE/dx - (v1 v1_x + v2 v2_x) ]
    ## ->  E_x = 1/rho d/x(rho E) - rho E /rho^2 rho_x
    kTx =( 1./Q[0]*(Qx[4] - Q[4]/Q[0]*Qx[0] - (Q[1]*ux + Q[2]*vx + Q[3]*wx)  ))*mu*gamma/Pr
    fx = [0]*5 
    v1 = Q[1]/Q[0]
    v2 = Q[2]/Q[0]
    v3 = Q[3]/Q[0]
    fx[1] = 2./3.*mu*(2.*ux - vy - wz) #tau11
    fx[2] = mu*(uy + vx)  #tau11
    fx[3] = mu*(uz + wx) #tau13
    fx[4] = fx[1]*v1 + fx[2]*v2 + fx[3]*v3 + kTx
    return fx
 

