from dolfin import *

#compute the action of JT on J
class eulerEqns1D:
  def evalF(self,Q):
    Fx = [None]*3
    es = 1.e-30
    gamma = 1.4
    rho = Q[0]
    rhoU = Q[1]
    rhoE = Q[2]
    p = (gamma - 1.)*(rhoE - 0.5*rhoU**2/rho)
    Fx[0] = Q[1]  
    Fx[1] = rhoU*rhoU/(rho) + p 
    Fx[2] = (rhoE + p)*rhoU/(rho) 
    return Fx

  def applyJT(self,Q0,Qp):
    es = 1.e-30
    gamma = 1.4 
    u = Q0[1]/Q0[0]
    qsqr = u**2
    # compute H in three steps (H = E + p/rho)
    H = (gamma - 1.)*(Q0[2] - 0.5*Q0[0]*qsqr) #compute pressure
    H += Q0[2]
    H /= Q0[0]
    f = [None]*3
    f[0] = (  (gamma - 1.)/2.*qsqr - u**2 )   *Qp[1] + \
           (  (gamma- -1.)/2.*qsqr - H )*u    *Qp[2]
  
    f[1] =            1                       *Qp[0] + \
           (  (3 - gamma)*u               )   *Qp[1] + \
           (  H + (1 - gamma)*u**2        )   *Qp[2]
  
    f[2] = (  (gamma - 1.)                )   *Qp[1] + \
           (  gamma * u                   )   *Qp[2]
  
    return f

  def applyJ(self,Q0,Qp):
    es = 1.e-30
    gamma = 1.4 
    u = Q0[1]/Q0[0]
    qsqr = u**2
    # compute H in three steps (H = E + p/rho)
    H = (gamma - 1.)*(Q0[2] - 0.5*Q0[0]*qsqr) #compute pressure
    H += Q0[2]
    H /= Q0[0]
    f = [None]*3
    f[0] = Qp[1]
    f[1] = ( (gamma - 1.)/2.*qsqr - u**2)*Qp[0] + (3. - gamma)*u*Qp[1] +\
           (gamma - 1.)*Qp[2]
    f[2] = ((gamma - 1.)/2.*qsqr - H)*u*Qp[0] + (H + (1. - gamma)*u**2)*Qp[1] +  \
           gamma*u*Qp[2]
    return f

  def evalF_Strong(self,Q,Qx):
    gamma = 1.4
    p =  (gamma - 1.)*(Q[2] - 0.5*Q[1]**2/Q[0])
    px = (gamma - 1.)*(Qx[2] - 1./Q[0]*(Q[1]*Qx[1]) + 0.5/Q[0]**2*Qx[0]*(Q[1]**2) )
    Rx = [None]*3
    Rx[0] = Qx[1]
    Rx[1] = 2.*Q[1]*Qx[1]/Q[0] - Qx[0]*Q[1]**2/Q[0]**2 + px
    Rx[2] = Q[1]/Q[0]*(Qx[2] + px) + Qx[1]/Q[0]*(Q[2] + p) - Qx[0]/Q[0]**2*Q[1]*(Q[2] + p)
    return Rx




