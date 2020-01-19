from dolfin import *


def weakform_SUPG(Phi,U,eqns):
  Rx,Ry,Rz = eqn.evalF(U)
  JTPhi_x,JTPhi_y,JTPhi_z = eqn.applyJT(U, Phi.dx(0),Phi.dx(1),Phi.dx(2) )
  R_strong= eqn.evalF_Strong(U,Ux,Uy,Uz)

  R_strongDT = [None]*eqns.nvars
  for i in range(0,5):
    R_strongDT[i] = R_strong[i] + dti*(U[i] - U_n[i])

  for i in range(0,eqns.nvars):
    F +=  inner(Phi[i] ,R_strong[i] )*dx + \
          inner(JTPhi_x[i] + JTPhi_y[i] + JTPhi_z[i],tau*R_strongDT[i] )*dx
  return F
