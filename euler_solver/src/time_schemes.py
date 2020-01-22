from dolfin import *
import numpy as np


def mass_BDF2(Phi,U,auxU_container,dt):
  U_nm1 = auxU_container[0]
  U_nm2 = auxU_container[1]
  dti = 1./dt
  F_BDF2 = inner(Phi,dti*(U - 4./3.*U_nm1 + 1./3.*U_nm2) )*dx
  return F_BDF2

def mass_BDF1(Phi,U,auxU_container,dt):
  U_nm1 = auxU_container[0]
  dti = 1./dt
  F_BDF1 = inner(Phi,dti*(U - U_nm1) )*dx
  return F_BDF1

