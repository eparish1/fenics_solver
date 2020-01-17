def eulerInviscidFlux(Q):
  es = 1.e-30
  gamma = 1.4
  rho = Q[0]
  rhoU = Q[1]
  rhoV = Q[2]
  rhoW = Q[3]
  rhoE = Q[4]
  p = (gamma - 1.)*(rhoE - 0.5*rhoU**2/rho - 0.5*rhoV**2/rho - 0.5*rhoW**2/rho)
  fx = [None]*5
  fx[0] = Q[1]
  fx[1] = rhoU*rhoU/(rho) + p
  fx[2] = rhoU*rhoV/(rho)
  fx[3] = rhoU*rhoW/(rho)
  fx[4] = (rhoE + p)*rhoU/(rho)

  fy = [None]*5
  fy[0] = Q[2]
  fy[1] = rhoU*rhoV/(rho)
  fy[2] = rhoV*rhoV/(rho) + p
  fy[3] = rhoV*rhoW/(rho)
  fy[4] = (rhoE + p)*rhoV/(rho)

  fz = [None]*5
  fz[0] = Q[3]
  fz[1] = rhoU*rhoW/(rho)
  fz[2] = rhoV*rhoW/(rho)
  fz[3] = rhoW*rhoW/(rho) + p
  fz[4] = (rhoE + p)*rhoW/(rho)
  return fx,fy,fz 


def eulerStrongFormResid(Q,Qx,Qy,Qz):
    gamma = 1.4
    p = (gamma - 1.)*(Q[4] - 0.5*Q[1]**2/Q[0] - 0.5*Q[2]**2/Q[0] - 0.5*Q[3]**2/Q[0])
    px = (gamma - 1.)* (Qx[4] - 1./Q[0]*(Q[3]*Qx[3] + Q[2]*Qx[2] + Q[1]*Qx[1]) + 0.5/Q[0]**2*Qx[0]*(Q[3]**2 + Q[2]**2 + Q[1]**2) )
    py = (gamma - 1.)* (Qy[4] - 1./Q[0]*(Q[3]*Qy[3] + Q[2]*Qy[2] + Q[1]*Qy[1]) + 0.5/Q[0]**2*Qy[0]*(Q[3]**2 + Q[2]**2 + Q[1]**2) )
    pz = (gamma - 1.)* (Qz[4] - 1./Q[0]*(Q[3]*Qz[3] + Q[2]*Qz[2] + Q[1]*Qz[1]) + 0.5/Q[0]**2*Qz[0]*(Q[3]**2 + Q[2]**2 + Q[1]**2) )
    f = [None]*5
    f[0] = Qx[1]  #d/dx(rho Q)
    f[1] = 2.*Q[1]*Qx[1]/Q[0] - Qx[0]*Q[1]**2/Q[0]**2 + px
    f[2] = Q[1]*Qx[2]/Q[0] + Qx[1]*Q[2]/Q[0] - Qx[0]*Q[1]*Q[2]/Q[0]**2
    f[3] = Q[1]*Qx[3]/Q[0] + Qx[1]*Q[3]/Q[0] - Qx[0]*Q[1]*Q[3]/Q[0]**2
    f[4] = Q[1]/Q[0]*(Qx[4] + px) + Qx[1]/Q[0]*(Q[4] + p) - Qx[0]/Q[0]**2*Q[1]*(Q[4] + p)
    
    f[0] += Qy[2]  #d/dx(rho)
    f[1] += Q[1]*Qy[2]/Q[0] + Qy[1]*Q[2]/Q[0] - Qy[0]*Q[1]*Q[2]/Q[0]**2
    f[2] += 2.*Q[2]*Qy[2]/Q[0] - Qy[0]*Q[2]**2/Q[0]**2 + py
    f[3] += Q[2]*Qy[3]/Q[0] + Qy[2]*Q[3]/Q[0] - Qy[0]*Q[2]*Q[3]/Q[0]**2
    f[4] += Q[2]/Q[0]*(Qy[4] + py) + Qy[2]/Q[0]*(Q[4] + p) - Qy[0]/Q[0]**2*Q[2]*(Q[4] + p)

    f[0] += Qz[3]  #d/dx(rho)
    f[1] += Q[1]*Qz[3]/Q[0] + Qz[1]*Q[3]/Q[0] - Qz[0]*Q[1]*Q[3]/Q[0]**2
    f[2] += Q[3]*Qz[2]/Q[0] + Qz[3]*Q[2]/Q[0] - Qz[0]*Q[3]*Q[2]/Q[0]**2
    f[3] += 2.*Q[3]*Qz[3]/Q[0] - Qz[0]*Q[3]**2/Q[0]**2 + pz
    f[4] += Q[3]/Q[0]*(Qz[4] + pz) + Qz[3]/Q[0]*(Q[4] + p) - Qz[0]/Q[0]**2*Q[3]*(Q[4] + p)
    return f


def eulerInviscidFluxLin(Q0,Qp):
  #decompose as Q = Q0 + Qp, where Qp is the perturbation
  gamma = 1.4
  u = Q0[1]/Q0[0]
  v = Q0[2]/Q0[0]
  w = Q0[3]/Q0[0]
  qsqr = u**2 + v**2 + w**2
  # compute H in three steps (H = E + p/rho)
  H = (gamma - 1.)*(Q0[4] - 0.5*Q0[0]*qsqr) #compute pressure
  H += Q0[4]
  H /= Q0[0]
  fx = [None]*5
  fx[0] = Qp[1]
  fx[1] = ( (gamma - 1.)/2.*qsqr - u**2)*Qp[0] + (3. - gamma)*u*Qp[1] + (1. - gamma)*v*Qp[2] + \
         (1. - gamma)*w*Qp[3] + (gamma - 1.)*Qp[4]
  fx[2] = -u*v*Qp[0] + v*Qp[1] + u*Qp[2]
  fx[3] = -u*w*Qp[0] + w*Qp[1] + u*Qp[3]
  fx[4] = ((gamma - 1.)/2.*qsqr - H)*u*Qp[0] + (H + (1. - gamma)*u**2)*Qp[1] + (1. - gamma)*u*v*Qp[2] + \
         (1. - gamma)*u*w*Qp[3] + gamma*u*Qp[4]

  fy = [None]*5
  fy[0] = Qp[2]
  fy[1] = -v*u*Qp[0] + v*Qp[1] + u*Qp[2]
  fy[2] = ( (gamma - 1.)/2.*qsqr - v**2)*Qp[0] + (1. - gamma)*u*Qp[1] + (3. - gamma)*v*Qp[2] + \
         (1. - gamma)*w*Qp[3] + (gamma - 1.)*Qp[4]
  fy[3] = -v*w*Qp[0] + w*Qp[2] + v*Qp[3]
  fy[4] = ((gamma - 1.)/2.*qsqr - H)*v*Qp[0] + (1. - gamma)*u*v*Qp[1] + (H + (1. - gamma)*v**2)*Qp[2] + \
         (1. - gamma)*v*w*Qp[3] + gamma*v*Qp[4]

  fz = [None]*5

  fz[0] = Qp[3]
  fz[1] = -u*w*Qp[0] + w*Qp[1] + u*Qp[3]
  fz[2] = -v*w*Qp[0] + w*Qp[2] + v*Qp[3]
  fz[3] = ( (gamma - 1.)/2.*qsqr - w**2)*Qp[0] + (1. - gamma)*u*Qp[1] + (1. - gamma)*v*Qp[2] + \
         (3. - gamma)*w*Qp[3] + (gamma - 1.)*Qp[4]
  fz[4] = ((gamma - 1.)/2.*qsqr - H)*w*Qp[0] + (1. - gamma)*u*w*Qp[1] + (1. - gamma)*v*w*Qp[2] + \
          (H + (1. - gamma)*w**2)*Qp[3] + gamma*w*Qp[4]
  return fx,fy,fz
