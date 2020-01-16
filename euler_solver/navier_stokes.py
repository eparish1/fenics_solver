from dolfin import *

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
