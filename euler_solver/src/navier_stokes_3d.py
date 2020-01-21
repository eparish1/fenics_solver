from euler_3d import  *
import numpy as np
from list_container import *
class navierStokesEqns3D:
  eulerEq = eulerEqns3D()
  evalF = eulerEq.evalF
  evalF_Strong = eulerEq.evalF_Strong
  applyJ = eulerEq.applyJ
  applyJT = eulerEq.applyJT
  nvars = 5
  gamma = eulerEq.gamma
  def __init__(self,mu = 1./100.*0.2,Pr=0.72):
    self.mu = mu
    self.Pr = Pr
    self.params = {"mu":self.mu , "gamma":self.gamma, "Pr":self.Pr }
  def evalGsGradients(self,Q,Qx,Qy,Qz):
    gamma = 1.4
    Pr = 0.72
    ## initialize containers
    GC = listContainer_2d(3,3) 
    G11x = listContainer_2d(self.nvars,self.nvars)
    G21x = listContainer_2d(self.nvars,self.nvars)
    G31x = listContainer_2d(self.nvars,self.nvars)
    G12y = listContainer_2d(self.nvars,self.nvars)
    G22y = listContainer_2d(self.nvars,self.nvars)
    G32y = listContainer_2d(self.nvars,self.nvars)
    G13z = listContainer_2d(self.nvars,self.nvars)
    G23z = listContainer_2d(self.nvars,self.nvars)
    G33z = listContainer_2d(self.nvars,self.nvars)


    u = Q[1]/Q[0]
    v = Q[2]/Q[0]
    w = Q[3]/Q[0]
    E = Q[4]/Q[0]

    ## viscosity, in the future update this to be a field
    mu = self.mu
    mu_x ,mu_y, mu_z = 0,0,0
    vsqr = u**2 + v**2 + w**2

    mu_by_rho = self.mu/Q[0]
    rho_inv = 1./Q[0]

    u_x = rho_inv*(Qx[1] - Qx[0]*Q[1]*rho_inv)    
    v_x = rho_inv*(Qx[2] - Qx[0]*Q[2]*rho_inv)    
    w_x = rho_inv*(Qx[3] - Qx[0]*Q[3]*rho_inv)    
    E_x = rho_inv*(Qx[4] - Qx[0]*Q[4]*rho_inv)    
    u_y = rho_inv*(Qy[1] - Qy[0]*Q[1]*rho_inv)    
    v_y = rho_inv*(Qy[2] - Qy[0]*Q[2]*rho_inv)    
    w_y = rho_inv*(Qy[3] - Qy[0]*Q[3]*rho_inv)    
    E_y = rho_inv*(Qy[4] - Qy[0]*Q[4]*rho_inv)    
    u_z = rho_inv*(Qz[1] - Qz[0]*Q[1]*rho_inv)    
    v_z = rho_inv*(Qz[2] - Qz[0]*Q[2]*rho_inv)    
    w_z = rho_inv*(Qz[3] - Qz[0]*Q[3]*rho_inv)    
    E_z = rho_inv*(Qz[4] - Qz[0]*Q[4]*rho_inv)    
    vsqr_x = 2*(u*u_x + v*v_x + w*w_x)
    vsqr_y = 2*(u*u_y + v*v_y + w*w_y)
    vsqr_z = 2*(u*u_z + v*v_z + w*w_z)


    G11x[1][0] = 4./3.*u_x
    G11x[1][1] = 0.
    G11x[2][0] = -v_x 
    G11x[2][2] = 0.
    G11x[3][0] = -w_x
    G11x[3][3] = 0.
    G11x[4][0] = -gamma/Pr*( E_x - vsqr_x) - 2*w*w_x - 2*v*v_x - 8*u*u_x/3.
    G11x[4][1] = (4./3. - gamma/Pr)*u_x
    G11x[4][2] = (4./3. - gamma/Pr)*v_x
    G11x[4][3] = (4./3. - gamma/Pr)*w_x
    G11x[4][4] = 0
    G21x[1][0] = G11x[2][0]
    G21x[1][2] = 0.
    G21x[2][0] = 2./3.*u_x
    G21x[2][1] = 0. 
    G21x[4][0] = -1./3.*(u*v_x + v*u_x)
    G21x[4][1] = -2./3.*v_x
    G21x[4][2] = u_x
    G31x[1][0] = G11x[3][0]#-w*mu_by_rho
    G31x[1][3] = 0.
    G31x[3][0] = G21x[2][0]
    G31x[3][1] = G21x[2][1]
    G31x[4][0] = -1./3.*(u_x*w + w_x*u)
    G31x[4][1] = 2./3.*G11x[3][0]
    G31x[4][3] = G21x[4][2]


    G12y[1][0] = 2./3.*v_y
    G12y[2][0] = -u_y
    G12y[4][0] = -1./3.*(u_y*v + v_y*u)
    G12y[4][1] = v_y
    G12y[4][2] = -2./3.*u_y
    G22y[1][0] = G12y[2][0] 
    G22y[2][0] = -4./3.*v_y
    G22y[3][0] = -w_y
    G22y[4][0] = -gamma/Pr*(E_y - vsqr_y) - 2*w*w_y - 8./3.*v*v_y - 2.*u*u_y
    G22y[4][1] = (1. - gamma/Pr)*u_y
    G22y[4][2] = (4./3. - gamma/Pr)*v_y
    G22y[4][3] = (4./3. - gamma/Pr)*w_y
    G32y[2][0] = G22y[3][0]
    G32y[3][0] = G12y[1][0]
    G32y[3][2] = G12y[1][2]
    G32y[4][0] = -1./3.*(v_y*w + w_y*v)
    G32y[4][2] = -2./3.*w_y
    G32y[4][3] = G12y[4][1]

    G13z[1][0] = 2./3.*w_z
    G13z[3][0] = -u_z
    G13z[4][0] = -1./3.*(u_z*w + w_z*u)
    G13z[4][1] = w_z
    G13z[4][3] = -2./3.*u_z
    G23z[2][0] = G13z[1][0]
    G23z[3][0] = -v_z
    G23z[4][0] = -1./3.*(v_z*w + w_z*v)
    G23z[4][2] = G13z[4][1]
    G23z[4][3] = -2./3.*v_z
    G33z[1][0] = -u_z
    G33z[2][0] = -v_z
    G33z[3][0] = -4./3.*w_z
    G33z[4][0] = -gamma/Pr*(E_z - vsqr_z) - 8./3.*w*w_z - 2*v*v_z - 2.*u*u_z
    G33z[4][1] = (1. - gamma/Pr)*u_x
    G33z[4][2] = (1. - gamma/Pr)*v_x
    G33z[4][3] = (4./3. - gamma/Pr)*w_x

    # add mu_by_rho contribution via chain rule
    mu_by_rho_x = mu_x * rho_inv - mu*rho_inv**2*Qx[0]
    mu_by_rho_y = mu_y * rho_inv - mu*rho_inv**2*Qy[0]
    mu_by_rho_z = mu_z * rho_inv - mu*rho_inv**2*Qz[0]
    for i in range(0,5):
      for j in range(0,5):
        G11x[i][j] *= mu_by_rho_x 
        G21x[i][j] *= mu_by_rho_x 
        G31x[i][j] *= mu_by_rho_x
        G12y[i][j] *= mu_by_rho_y 
        G22y[i][j] *= mu_by_rho_y 
        G32y[i][j] *= mu_by_rho_y
        G13z[i][j] *= mu_by_rho_z 
        G23z[i][j] *= mu_by_rho_z 
#        G33z[i][j] *= mu_by_rho_z

    GC[0][0] = G11x
    GC[0][1] = G12y
    GC[0][2] = G13z
    GC[1][0] = G21x
    GC[1][1] = G22y
    GC[1][2] = G23z
    GC[2][0] = G31x
    GC[2][1] = G32y
    GC[2][2] = G33z
    return GC



  def evalGs(self,Q):
    gamma = 1.4
    Pr = 0.72
    ## initialize containers
    GC = listContainer_2d(3,3) 
    G11 = listContainer_2d(self.nvars,self.nvars)
    G21 = listContainer_2d(self.nvars,self.nvars)
    G31 = listContainer_2d(self.nvars,self.nvars)
    G12 = listContainer_2d(self.nvars,self.nvars)
    G22 = listContainer_2d(self.nvars,self.nvars)
    G32 = listContainer_2d(self.nvars,self.nvars)
    G13 = listContainer_2d(self.nvars,self.nvars)
    G23 = listContainer_2d(self.nvars,self.nvars)
    G33 = listContainer_2d(self.nvars,self.nvars)

#    G11 = [[0. for i in range(self.nvars)] for j in range(self.nvars)] 
#    G21 = [[0. for i in range(self.nvars)] for j in range(self.nvars)] 
#    G31 = [[0. for i in range(self.nvars)] for j in range(self.nvars)] 
#    G12 = [[0. for i in range(self.nvars)] for j in range(self.nvars)] 
#    G22 = [[0. for i in range(self.nvars)] for j in range(self.nvars)] 
#    G32 = [[0. for i in range(self.nvars)] for j in range(self.nvars)] 
#    G13 = [[0. for i in range(self.nvars)] for j in range(self.nvars)] 
#    G23 = [[0. for i in range(self.nvars)] for j in range(self.nvars)] 
#    G33 = [[0. for i in range(self.nvars)] for j in range(self.nvars)] 

    u = Q[1]/Q[0]
    v = Q[2]/Q[0]
    w = Q[3]/Q[0]
    E = Q[4]/Q[0]
    vsqr = u**2 + v**2 + w**2

    mu_by_rho = self.mu/Q[0]

    G11[1][0] = -4./3.*u*mu_by_rho
    G11[1][1] = 4./3.*mu_by_rho
    G11[2][0] = -v*mu_by_rho
    G11[2][2] = mu_by_rho
    G11[3][0] = -w*mu_by_rho
    G11[3][3] = mu_by_rho
    G11[4][0] = -(4./3.*u**2 + v**2 + w**2 + gamma/Pr*(E - vsqr) )*mu_by_rho
    G11[4][1] = (4./3. - gamma/Pr)*u*mu_by_rho
    G11[4][2] = (1. - gamma/Pr)*v*mu_by_rho
    G11[4][3] = (1. - gamma/Pr)*w*mu_by_rho
    G11[4][4] = gamma/Pr*mu_by_rho
 
    G21[1][0] = G11[2][0]#-v*mu_by_rho
    G21[1][2] = mu_by_rho
    G21[2][0] = 2./3.*u*mu_by_rho
    G21[2][1] = -2./3.*mu_by_rho
    G21[4][0] = -1./3.*u*v*mu_by_rho

    G21[4][1] = -2./3.*v*mu_by_rho
    G21[4][2] = u*mu_by_rho

    G31[1][0] = G11[3][0]#-w*mu_by_rho
    G31[1][3] = mu_by_rho
    G31[3][0] = G21[2][0]#2./3.*v1*mu_by_rho
    G31[3][1] = G21[2][1]#-2./3.*mu_by_rho
    G31[4][0] = -1./3.*u*w*mu_by_rho
    G31[4][1] = 2./3.*G11[3][0]#-2./3.*w*mu_by_rho
    G31[4][3] = G21[4][2]#v1*mu_by_rho


    G12[1][0] = 2./3.*v*mu_by_rho
    G12[1][2] = -2./3.*mu_by_rho
    G12[2][0] = -u*mu_by_rho
    G12[2][1] = mu_by_rho
    G12[4][0] = -1./3.*u*v*mu_by_rho
    G12[4][1] = v*mu_by_rho
    G12[4][2] = -2./3.*u*mu_by_rho

    G22[1][0] = G12[2][0]#-v1*mu_by_rho
    G22[1][1] = mu_by_rho
    G22[2][0] = -4./3.*v*mu_by_rho
    G22[2][2] = 4./3.*mu_by_rho
    G22[3][0] = -w*mu_by_rho
    G22[3][3] = mu_by_rho
    G22[4][0] = -(u**2 + 4./3.*v**2 + w**2 + gamma/Pr*(E - vsqr) )*mu_by_rho
    G22[4][1] = (1. - gamma/Pr)*u*mu_by_rho
    G22[4][2] = (4./3. - gamma/Pr)*v*mu_by_rho
    G22[4][3] = (1. - gamma/Pr)*w*mu_by_rho
    G22[4][4] = gamma/Pr*mu_by_rho

    G32[2][0] = G22[3][0]#-w*mu_by_rho
    G32[2][3] = mu_by_rho
    G32[3][0] = G12[1][0]#2./3.*v*mu_by_rho
    G32[3][2] = G12[1][2]#-2./3.*mu_by_rho
    G32[4][0] = -1./3.*v*w*mu_by_rho
    G32[4][2] = -2./3.*w*mu_by_rho
    G32[4][3] = G12[4][1]#v*mu_by_rho

    G13[1][0] = 2./3.*w*mu_by_rho
    G13[1][3] = -2./3.*mu_by_rho
    G13[3][0] = -u*mu_by_rho
    G13[3][1] = mu_by_rho
    G13[4][0] = -1./3.*u*w*mu_by_rho
    G13[4][1] = w*mu_by_rho
    G13[4][3] = -2./3.*u*mu_by_rho

    G23[2][0] = G13[1][0]#2./3.*w*mu_by_rho
    G23[2][3] = -2./3.*mu_by_rho
    G23[3][0] = -v*mu_by_rho
    G23[3][2] = mu_by_rho
    G23[4][0] = -1./3.*v*w*mu_by_rho
    G23[4][2] = G13[4][1]#w*mu_by_rho
    G23[4][3] = -2./3.*v*mu_by_rho

    G33[1][0] = -u*mu_by_rho
    G33[1][1] = mu_by_rho
    G33[2][0] = -v*mu_by_rho
    G33[2][2] = mu_by_rho
    G33[3][0] = -4./3.*w*mu_by_rho
    G33[3][3] = 4./3.*mu_by_rho
    G33[4][0] = -(u**2 + v**2 + 4./3.*w**2 + gamma/Pr*(E - vsqr) )*mu_by_rho
    G33[4][1] = (1. - gamma/Pr)*u*mu_by_rho
    G33[4][2] = (1. - gamma/Pr)*v*mu_by_rho
    G33[4][3] = (4./3. - gamma/Pr)*w*mu_by_rho
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
    ## ->        =mu gamma/Pr *[ dE/dx - 0.5 d/dx(v1^2 + v^2) ]
    ## ->        =mu gamma/Pr *[ dE/dx - (v1 v1_x + v v_x) ]
    ## ->  E_x = 1/rho d/x(rho E) - rho E /rho^2 rho_x
    kTx =( 1./Q[0]*(Qx[4] - Q[4]/Q[0]*Qx[0] - (Q[1]*ux + Q[2]*vx + Q[3]*wx)  ))*mu*gamma/Pr
    fx = [0]*5 
    v1 = Q[1]/Q[0]
    v = Q[2]/Q[0]
    w = Q[3]/Q[0]
    fx[1] = 2./3.*mu*(2.*ux - vy - wz) #tau11
    fx[2] = mu*(uy + vx)  #tau11
    fx[3] = mu*(uz + wx) #tau13
    fx[4] = fx[1]*v1 + fx[2]*v + fx[3]*w + kTx
    return fx
 

