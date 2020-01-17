from dolfin import *

def eulerInviscidFlux(Q):
  Fx = [None]*3
  es = 1.e-30
  gamma = 1.4
  rho = Q[0]
  rhoU = Q[1]
  rhoE = Q[-1]
  p = (gamma - 1.)*(rhoE - 0.5*rhoU**2/rho)
  Fx[0] = Q[1]  
  Fx[1] = rhoU*rhoU/(rho) + p 
  Fx[2] = (rhoE + p)*rhoU/(rho) 
  return Fx

def evalInviscidFluxLinNS(Q0,Qp):
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

def strongFormResidNS(Q,Qx):
  gamma = 1.4
  p =  (gamma - 1.)*(Q[2] - 0.5*Q[1]**2/Q[0])
  px = (gamma - 1.)*(Qx[2] - 1./Q[0]*(Q[1]*Qx[1]) + 0.5/Q[0]**2*Qx[0]*(Q[1]**2) )
  Rx = [None]*3
  Rx[0] = Qx[1]
  Rx[1] = 2.*Q[1]*Qx[1]/Q[0] - Qx[0]*Q[1]**2/Q[0]**2 + px
  Rx[2] = Q[1]/Q[0]*(Qx[2] + px) + Qx[1]/Q[0]*(Q[2] + p) - Qx[0]/Q[0]**2*Q[1]*(Q[2] + p)
  return Rx



def eulerInviscidWeak(Q,Phi):
  es = 1.e-30
  gamma = 1.4
  rho = Q[0]
  rhoU = Q[1]
  #rhoV = Q[2]
  #rhoW = Q[3]
  rhoE = Q[-1]
  p = (gamma - 1.)*(rhoE - 0.5*rhoU**2/rho)# - 0.5*rhoV**2/rho - 0.5*rhoW**2/rho)
  F =  inner( Phi[0].dx(0) , Q[1] )*dx + \
       inner( Phi[1].dx(0) , rhoU*rhoU/(rho) + p )*dx + \
       inner( Phi[-1].dx(0) , (rhoE + p)*rhoU/(rho) )*dx
  return F
  '''
  F =  inner( Phi[0].dx(0) , Q[1] )*dx + \
       inner( Phi[1].dx(0) , rhoU*rhoU/(rho) + p )*dx + \
       inner( Phi[2].dx(0) , rhoU*rhoV/(rho) )*dx + \
       inner( Phi[3].dx(0) , rhoU*rhoW/(rho) )*dx + \
       inner( Phi[-1].dx(0) , (rhoE + p)*rhoU/(rho) )*dx

  fy[0] = Q[2]
  fy[1] = rhoU*rhoV/(rho)
  fy[2] = rhoV*rhoV/(rho) + p
  fy[3] = rhoV*rhoW/(rho)
  fy[4] = (rhoE + p)*rhoV/(rho)

  fz[0] = Q[3]
  fz[1] = rhoU*rhoW/(rho)
  fz[2] = rhoV*rhoW/(rho)
  fz[3] = rhoW*rhoW/(rho) + p
  fz[4] = (rhoE + p)*rhoW/(rho)
  '''
