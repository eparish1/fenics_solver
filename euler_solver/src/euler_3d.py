from list_container import *
class eulerEqns3D:
  nvars = 5
  def __init__(self,gamma=1.4):
    self.gamma = gamma
  def evalF(self,Q):
    self.gamma = 1.4
    rho = Q[0]
    rhoU = Q[1]
    rhoV = Q[2]
    rhoW = Q[3]
    rhoE = Q[4]
    p = (self.gamma - 1.)*(rhoE - 0.5*rhoU**2/rho - 0.5*rhoV**2/rho - 0.5*rhoW**2/rho)
    fx = listContainer_1d(self.nvars)
    fx[0] = Q[1]*1
    fx[1] = rhoU*rhoU/(rho) + p
    fx[2] = rhoU*rhoV/(rho)
    fx[3] = rhoU*rhoW/(rho)
    fx[4] = (rhoE + p)*rhoU/(rho)
  
    fy = listContainer_1d(self.nvars)
    fy[0] = Q[2]*1
    fy[1] = rhoU*rhoV/(rho)
    fy[2] = rhoV*rhoV/(rho) + p
    fy[3] = rhoV*rhoW/(rho)
    fy[4] = (rhoE + p)*rhoV/(rho)
  
    fz = listContainer_1d(self.nvars)
    fz[0] = Q[3]*1
    fz[1] = rhoU*rhoW/(rho)
    fz[2] = rhoV*rhoW/(rho)
    fz[3] = rhoW*rhoW/(rho) + p
    fz[4] = (rhoE + p)*rhoW/(rho)
    return fx,fy,fz 


  def evalF_Strong(self,Q,Qx,Qy,Qz):
    self.gamma = 1.4
    p = (self.gamma - 1.)*(Q[4] - 0.5*Q[1]**2/Q[0] - 0.5*Q[2]**2/Q[0] - 0.5*Q[3]**2/Q[0])
    px = (self.gamma - 1.)* (Qx[4] - 1./Q[0]*(Q[3]*Qx[3] + Q[2]*Qx[2] + Q[1]*Qx[1]) + 0.5/Q[0]**2*Qx[0]*(Q[3]**2 + Q[2]**2 + Q[1]**2) )
    py = (self.gamma - 1.)* (Qy[4] - 1./Q[0]*(Q[3]*Qy[3] + Q[2]*Qy[2] + Q[1]*Qy[1]) + 0.5/Q[0]**2*Qy[0]*(Q[3]**2 + Q[2]**2 + Q[1]**2) )
    pz = (self.gamma - 1.)* (Qz[4] - 1./Q[0]*(Q[3]*Qz[3] + Q[2]*Qz[2] + Q[1]*Qz[1]) + 0.5/Q[0]**2*Qz[0]*(Q[3]**2 + Q[2]**2 + Q[1]**2) )
    f = listContainer_1d(self.nvars)
    f = listContainer_1d(self.nvars)
    f = listContainer_1d(self.nvars)

    f[0] = 0 + Qx[1]*1  #d/dx(rho Q)
    f[1] = 2.*Q[1]*Qx[1]/Q[0] - Qx[0]*Q[1]**2/Q[0]**2 + px
    f[2] = Q[1]*Qx[2]/Q[0] + Qx[1]*Q[2]/Q[0] - Qx[0]*Q[1]*Q[2]/Q[0]**2
    f[3] = Q[1]*Qx[3]/Q[0] + Qx[1]*Q[3]/Q[0] - Qx[0]*Q[1]*Q[3]/Q[0]**2
    f[4] = Q[1]/Q[0]*(Qx[4] + px) + Qx[1]/Q[0]*(Q[4] + p) - Qx[0]/Q[0]**2*Q[1]*(Q[4] + p)
    
    f[0] += 0 +Qy[2]*1  #d/dx(rho)
    f[1] += Q[1]*Qy[2]/Q[0] + Qy[1]*Q[2]/Q[0] - Qy[0]*Q[1]*Q[2]/Q[0]**2
    f[2] += 2.*Q[2]*Qy[2]/Q[0] - Qy[0]*Q[2]**2/Q[0]**2 + py
    f[3] += Q[2]*Qy[3]/Q[0] + Qy[2]*Q[3]/Q[0] - Qy[0]*Q[2]*Q[3]/Q[0]**2
    f[4] += Q[2]/Q[0]*(Qy[4] + py) + Qy[2]/Q[0]*(Q[4] + p) - Qy[0]/Q[0]**2*Q[2]*(Q[4] + p)

    f[0] += 0 + Qz[3]*1  #d/dx(rho)
    f[1] += Q[1]*Qz[3]/Q[0] + Qz[1]*Q[3]/Q[0] - Qz[0]*Q[1]*Q[3]/Q[0]**2
    f[2] += Q[3]*Qz[2]/Q[0] + Qz[3]*Q[2]/Q[0] - Qz[0]*Q[3]*Q[2]/Q[0]**2
    f[3] += 2.*Q[3]*Qz[3]/Q[0] - Qz[0]*Q[3]**2/Q[0]**2 + pz
    f[4] += Q[3]/Q[0]*(Qz[4] + pz) + Qz[3]/Q[0]*(Q[4] + p) - Qz[0]/Q[0]**2*Q[3]*(Q[4] + p)
    return f


  def applyJT(self,Q0,Qp1,Qp2,Qp3):
    ''' 
    Routine for flux jacobian transpose evaluated at Q0, acting on Qp_i
    inputs: Q0 \in \R^5
            Qp \in \R^5
    returns vector products [ dF_i/dV(Q0) ]^TQp_i, i=1,2,3, where
    F_i is the flux vector in the ith direction, F_i: V -> F(V)
    '''
    self.gamma = 1.4
    u = Q0[1]/Q0[0]
    v = Q0[2]/Q0[0]
    w = Q0[3]/Q0[0]
    qsqr = u**2 + v**2 + w**2
    # compute H in three steps (H = E + p/rho)
    H = (self.gamma - 1.)*(Q0[4] - 0.5*Q0[0]*qsqr) #compute pressure
    H += Q0[4]
    H /= Q0[0]
    fx = listContainer_1d(self.nvars)
    fx[0] = ( (self.gamma - 1.)/2.*qsqr - u**2 )*Qp1[1] - u*v*Qp1[2] - u*w*Qp1[3] + ((self.gamma - 1.)/2.*qsqr - H)*u*Qp1[4]
    fx[1] = Qp1[0] + (3 - self.gamma)*u*Qp1[1] + v*Qp1[2] + w*Qp1[3] + (H + (1. - self.gamma)*u**2)*Qp1[4] 
    fx[2] = (1 - self.gamma)*v*Qp1[1] + u*Qp1[2] + (1 - self.gamma)*u*v*Qp1[4]
    fx[3] = (1 - self.gamma)*w*Qp1[1] + u*Qp1[3] + (1 - self.gamma)*u*w*Qp1[4] 
    fx[4] = (self.gamma - 1.)*Qp1[1] + self.gamma*u*Qp1[4]
  
    fy = listContainer_1d(self.nvars)
    fy[0] = -v*u*Qp2[1] + ( (self.gamma - 1.)/2.*qsqr - v**2 )*Qp2[2] - v*w*Qp2[3] +  ((self.gamma - 1.)/2.*qsqr - H)*v*Qp2[4]
    fy[1] = v*Qp2[1] + (1 - self.gamma)*v*Qp2[2] + (1 - self.gamma)*u*v*Qp2[4]
    fy[2] = Qp2[0] + u*Qp2[1] + (3. - self.gamma)*u*Qp2[2] + w*Qp2[3] + (H + (1. - self.gamma)*v**2 )*Qp2[4] 
    fy[3] = (1. - self.gamma)*w*Qp2[2] + v*Qp2[3] + (1. - self.gamma)*v*w*Qp2[4] 
    fy[4] = (self.gamma - 1.)*Qp2[2] + self.gamma*v*Qp2[4] 
  
    fz = listContainer_1d(self.nvars)
    fz[0] = -u*w*Qp3[1] - v*w*Qp3[2] + ( (self.gamma - 1.)/2.*qsqr - w**2)*Qp3[3] + ((self.gamma - 1.)/2.*qsqr - H)*w*Qp3[4] 
    fz[1] = w*Qp3[1] + (1. - self.gamma)*u*Qp3[3] + (1. - self.gamma)*u*v*Qp3[4]
    fz[2] = w*Qp3[2] + (1. - self.gamma)*v*Qp3[3] + (1. - self.gamma)*v*w*Qp3[4]
    fz[3] = Qp3[0] + u*Qp3[1] + v*Qp3[2] + (3. - self.gamma)*w*Qp3[3] + (H + (1. - self.gamma)*w**2 )*Qp3[4]
    fz[4] = (self.gamma - 1.)*Qp3[3] + self.gamma*w*Qp3[4]
    return fx,fy,fz
  
  
  def applyJ(self,Q0,Qp1,Qp2,Qp3):
    ''' 
    Routine for flux jacobian transpose evaluated at Q0, acting on Qp_i
    inputs: Q0 \in \R^5
            Qp \in \R^5
    returns vector products [ dF_i/dV(Q0) ]Qp_i, i=1,2,3, where
    F_i is the flux vector in the ith direction, F_i: V -> F(V)
    '''
    gamma = self.gamma 
    u = Q0[1]/Q0[0]
    v = Q0[2]/Q0[0]
    w = Q0[3]/Q0[0]
    qsqr = u**2 + v**2 + w**2
    # compute H in three steps (H = E + p/rho)
    H = (gamma - 1.)*(Q0[4] - 0.5*Q0[0]*qsqr) #compute pressure
    H += Q0[4]
    H /= Q0[0]
    fx = listContainer_1d(self.nvars)
    fx[0] = Qp1[1]
    fx[1] = ( (gamma - 1.)/2.*qsqr - u**2)*Qp1[0] + (3. - gamma)*u*Qp1[1] + (1. - gamma)*v*Qp1[2] + \
           (1. - gamma)*w*Qp1[3] + (gamma - 1.)*Qp1[4]
    fx[2] = -u*v*Qp1[0] + v*Qp1[1] + u*Qp1[2]
    fx[3] = -u*w*Qp1[0] + w*Qp1[1] + u*Qp1[3]
    fx[4] = ((gamma - 1.)/2.*qsqr - H)*u*Qp1[0] + (H + (1. - gamma)*u**2)*Qp1[1] + (1. - gamma)*u*v*Qp1[2] + \
           (1. - gamma)*u*w*Qp1[3] + gamma*u*Qp1[4]
  
    fy = listContainer_1d(self.nvars)
    fy[0] = Qp2[2]
    fy[1] = -v*u*Qp2[0] + v*Qp2[1] + u*Qp2[2]
    fy[2] = ( (gamma - 1.)/2.*qsqr - v**2)*Qp2[0] + (1. - gamma)*u*Qp2[1] + (3. - gamma)*v*Qp2[2] + \
           (1. - gamma)*w*Qp2[3] + (gamma - 1.)*Qp2[4]
    fy[3] = -v*w*Qp2[0] + w*Qp2[2] + v*Qp2[3]
    fy[4] = ((gamma - 1.)/2.*qsqr - H)*v*Qp2[0] + (1. - gamma)*u*v*Qp2[1] + (H + (1. - gamma)*v**2)*Qp2[2] + \
           (1. - gamma)*v*w*Qp2[3] + gamma*v*Qp2[4]
  
    fz = listContainer_1d(self.nvars)
  
    fz[0] = Qp3[3]
    fz[1] = -u*w*Qp3[0] + w*Qp3[1] + u*Qp3[3]
    fz[2] = -v*w*Qp3[0] + w*Qp3[2] + v*Qp3[3]
    fz[3] = ( (gamma - 1.)/2.*qsqr - w**2)*Qp3[0] + (1. - gamma)*u*Qp3[1] + (1. - gamma)*v*Qp3[2] + \
           (3. - gamma)*w*Qp3[3] + (gamma - 1.)*Qp3[4]
    fz[4] = ((gamma - 1.)/2.*qsqr - H)*w*Qp3[0] + (1. - gamma)*u*w*Qp3[1] + (1. - gamma)*v*w*Qp3[2] + \
            (H + (1. - gamma)*w**2)*Qp3[3] + gamma*w*Qp3[4]
    return fx,fy,fz
