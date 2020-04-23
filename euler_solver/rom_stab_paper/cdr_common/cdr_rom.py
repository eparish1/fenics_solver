from dolfin import *
import numpy as np
import os
from postProcessor import *




def executeRom(taus,romProblem,physProblem,fom_sol,dt_fom): 
  if (romProblem.methodDiscrete == 'LSPG'):
    romProblem.dt = taus*1.

  romPP = postProcessor(romProblem,fom_sol,dt_fom)
  Phi = romProblem.Phi
  tau = Constant(taus)
  methodContinuous = romProblem.methodContinuous
  methodDiscrete = romProblem.methodDiscrete
  # Define variational problem
  U_n = Function(romProblem.functionSpace)
  U_nm1 = Function(romProblem.functionSpace)
  U_nm1.interpolate(physProblem.initialConditionFunction())


  f = Constant(physProblem.f_mag)
  nu = Constant(physProblem.nu_mag)
  sigma = Constant(physProblem.sigma_mag)


  dti = Constant(1./romProblem.dt)
  et = romProblem.et

  U = romProblem.U
  V = romProblem.V
  ## Residual
  r =  dti*U +  physProblem.u_field*U.dx(0)  + physProblem.v_field*U.dx(1)   - nu*div(grad(U)) + sigma*U 
  rnm1_f =  - f
  rnm1_u = -dti*U ##we just use this to assemble the matrix, U acts as a dummy for Unm1  
  resid_for_assembly = dti*U_n - dti*U_nm1 +  physProblem.u_field*U_n.dx(0)  + physProblem.v_field*U_n.dx(1)  - nu*div(grad(U_n)) + sigma*U_n
  Mdt_form =  inner(V,dti*(U))*dx
  velocity_form = - ( nu*inner(grad(V), grad(U))*dx - (  (physProblem.u_field*V.dx(0)  + physProblem.v_field*V.dx(1))*U  )*dx + inner(V,sigma*U)*dx )
  RHS =    f*V*dx 
  RHSMAT = dti*V*U*dx

  ## Stabilization
  if methodContinuous == 'SUPG':
    vSUPG = physProblem.u_field*V.dx(0) + physProblem.v_field*V.dx(1)
    velocity_form -=  vSUPG*tau*r*dx 
    RHS -=  vSUPG*tau*rnm1_f*dx
    RHSMAT -= vSUPG*tau*rnm1_u*dx 
  if methodContinuous == 'GLS':
    vGLS = V*dti -  nu*div(grad(V)) + physProblem.u_field*V.dx(0) + physProblem.v_field*V.dx(1) + sigma*V
    velocity_form -=  vGLS*tau*r*dx 
    RHS -=  vGLS*tau*rnm1_f*dx
    RHSMAT -= vGLS*tau*rnm1_u*dx 
  if methodContinuous == 'ADJ':
    vADJ = -V*dti +  nu*div(grad(V)) + physProblem.u_field*V.dx(0) + physProblem.v_field*V.dx(1) - sigma*V
    velocity_form -=  vADJ*tau*r*dx 
    RHS -=  vADJ*tau*rnm1_f*dx
    RHSMAT -= vADJ*tau*rnm1_u*dx 

  A_form = Mdt_form - velocity_form
  AM = assemble(A_form)
  velocity = assemble(velocity_form) 
  RHSv = assemble(RHS)
  Phi_test = Phi*1.

  if methodDiscrete == 'ADJ':
    Phi_test = Phi_test - tau*np.linalg.solve(romProblem.M.array(), np.dot(AM.array().transpose(),Phi_test))
    #Phi_test = np.linalg.solve(M.array(),Phi_test)# np.dot(np.linalg.inv(M.array()),Phi_test)

  if methodDiscrete == 'LSPG':
    Phi_test = np.dot(romProblem.M.array(),Phi_test)*1./romProblem.dt - np.dot(velocity.array(),Phi_test)
    Phi_test = np.dot(romProblem.Minv,Phi_test)  #np.linalg.solve(romProblem.M.array(),Phi_test)

  AMR = np.dot(Phi_test.transpose(),np.dot(AM.array(),Phi)) 
  if methodDiscrete == 'APG':
    Pi = np.dot(Phi,np.dot(Phi.transpose() , romProblem.M.array()))
    #Pi = np.dot(Phi,Phi.transpose() )
    Pihat = np.eye(np.shape(Phi)[0] ) - Pi
    SGS = taus*np.dot(velocity.array(),np.dot(Pihat,np.dot(romProblem.Minv,velocity.array())))
    AMR -= np.dot(Phi_test.transpose(),np.dot(SGS,Phi) )

  RHSvR = np.dot(Phi_test.transpose(),RHSv[:])
  RHSMAT = assemble(RHSMAT)
  RHSMATR = np.dot(Phi_test.transpose(),np.dot(RHSMAT.array(),Phi) )

  
  # Compute solution
  t = 0
  counter = 0
  
  sol = np.zeros(romProblem.K)
  #sol[:] = np.dot(romProblem.Phi.transpose(),np.dot(romProblem.M.array(),U_nm1.vector()[:]) ) 


  while (t <= et - romProblem.dt/2):
    sol = np.linalg.solve(AMR,RHSvR + np.dot(RHSMATR, sol))
    t += romProblem.dt
    romPP.postProcess(sol,counter,t)
    counter += 1

  sol_loc = 'solrom' + romProblem.basis_type + methodContinuous + '_' + methodDiscrete + '_tau_' + str(taus) + '_N_' + str(romProblem.N) + '_p_' + str(romProblem.p) + '/'
  romPP.saveSol(romProblem,sol_loc)

