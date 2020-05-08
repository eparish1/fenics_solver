from dolfin import *
import numpy as np
from femPostProcessor import *

if has_linear_algebra_backend("Epetra"):
    parameters["linear_algebra_backend"] = "Epetra"

def boundaryLR(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS
def boundaryUD(x):
    return x[1] < DOLFIN_EPS or x[1] > 1.0 - DOLFIN_EPS



def grab_sol(u_1_):
    x = np.linspace(0,1,nxSave)
    x,y = np.meshgrid(x,x,indexing='ij')
    Us = np.zeros((nxSave,nxSave))
    for i in range(0,nxSave):
      for j in range(0,nxSave):
        Us[i,j] = u_1_(x[i,j],y[i,j])
    return Us

def executeFom(femProblem,physProblem,femCoarseProblem,sol_loc=None,femPP=None):
  if sol_loc == None:
    sol_loc = 'solfom' + femProblem.methodContinuous + '_tau_' + str(femProblem.tau) + '_N_' + str(femProblem.N) + '_p_' + str(femProblem.p) + '_dt_' + str(femProblem.dt) + '/'
  if femPP == None:
    save_freq = 1 
    femPP = femPostProcessor(femProblem,femCoarseProblem,sol_loc,save_freq)
  # Define variational problem
  U = femProblem.U 
  U_nm1 = femProblem.U_nm1 
  U_nm1.interpolate(physProblem.initialConditionFunction())
  V = femProblem.V 

  u0 = Constant(0.0)
  bclr = DirichletBC(femProblem.functionSpace,u0, boundaryLR)
  bcud = DirichletBC(femProblem.functionSpace,u0, boundaryUD)
  bcs = [bclr,bcud]

  f = Constant(physProblem.f_mag)
  nu = Constant(physProblem.nu_mag)
  sigma = Constant(physProblem.sigma_mag)

  dti = Constant(1./femProblem.dt)
  h = femProblem.mesh.hmin()
  a_mag = sqrt(physProblem.u_field**2 + physProblem.v_field**2)
  #tau = 1.0/( 4.*physProblem.nu_mag / h**2 + 2.*a_mag / h + (physProblem.sigma_mag + 1./femProblem.dt))
  tau = femProblem.tau
  #print(tau)
  ## Residual
  r =  dti*U +  physProblem.u_field*U.dx(0)  + physProblem.v_field*U.dx(1)   - nu*div(grad(U)) + sigma*U 
  rnm1 =  -dti*U_nm1 - f 
  #A = inner(V,dti*(U))*dx + nu*inner(grad(V), grad(U))*dx - (  (physProblem.u_field*V.dx(0)  + physProblem.v_field*V.dx(1))*U  )*dx + inner(V,sigma*U)*dx
  A = inner(V,dti*(U))*dx + nu*inner(grad(V), grad(U))*dx + V*physProblem.u_field*U.dx(0)*dx + V*physProblem.v_field*U.dx(1)*dx  + inner(V,sigma*U)*dx
  RHS =  V*dti*U_nm1*dx + f*V*dx 
  ## Stabilization
  if femProblem.methodContinuous == 'SUPG':
    vSUPG = physProblem.u_field*V.dx(0) + physProblem.v_field*V.dx(1)
    A +=  vSUPG*tau*r*dx 
    RHS -=  tau*vSUPG*rnm1*dx
  if femProblem.methodContinuous == 'GLS':
    vGLS = V*dti -  nu*div(grad(V)) + physProblem.u_field*V.dx(0) + physProblem.v_field*V.dx(1) + sigma*V
    A +=  vGLS*tau*r*dx 
    RHS -=  vGLS*tau*rnm1*dx
  if femProblem.methodContinuous == 'ADJ':
    vADJ = -V*dti +  nu*div(grad(V)) + physProblem.u_field*V.dx(0) + physProblem.v_field*V.dx(1) - sigma*V
    A +=  vADJ*tau*r*dx 
    RHS -=  vADJ*tau*rnm1*dx

  #AM = assemble(A) 
  #for bc in bcs:
  #  bc.apply(AM)

  # Compute solution
  t = 0.
  et = femProblem.et
  counter = 0
  
  while (t <= et - femProblem.dt/2):
    solve(A == RHS, U_nm1,bcs)
    femPP.postProcess(U_nm1,counter,t,femCoarseProblem) 
    t += femProblem.dt
    counter += 1
    print(t,counter)

  femPP.saveSol(femProblem,femCoarseProblem)
'''
  
U_final = grab_sol(UC) 
UTrialC  = TrialFunction(VVC)
VC = TestFunction(VVC)

MC  = assemble( VC * UTrialC * dx )
HC =  assemble( inner(grad(VC), grad(UTrialC))*dx )

K = np.dot( UDOFSave.transpose(), np.dot(MC.array(),UDOFSave)) 
ub,sb,vb = np.linalg.svd(K)
Phi = np.dot(UDOFSave ,1./np.sqrt(sb+1e-30)*ub )

K = np.dot( UDOFSave.transpose(), np.dot(HC.array(),UDOFSave)) 
ub,sb_h,vb = np.linalg.svd(K)
PhiH = np.dot(UDOFSave ,1./np.sqrt(sb_h+1e-30)*ub )


Phi2 = Phi*1.

UDOFSave_project = np.zeros(np.shape(UDOFSave) )
for i in range(0,np.shape(UDOFSave)[1]):
     #UC.vector()[:] = UDOFSave[:,i]
     UDOFSave_project[:,i] = np.dot(Phi2, np.dot(Phi2.transpose(), np.dot(MC.array(),UDOFSave[:,i])) )
UC.vector()[:] = UDOFSave_project[:,-1]*1.
U_final_project = grab_sol(UC)

np.savez(sol_loc + '/pod_basis',UDOFSave=UDOFSave,Phi=Phi,sigma=sb,PhiH=PhiH,sigmaH=sb_h,U_final=U_final,MC=MC.array(),HC=HC.array(),UDOFSave_project=UDOFSave_project,U_final_project=U_final_project)

'''
