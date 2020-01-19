from dolfin import *
import sys
import numpy as np
sys.path.append('../../src/')
from euler_1d import *
if has_linear_algebra_backend("Epetra"):
    parameters["linear_algebra_backend"] = "Epetra"

import time
t0 = time.time()
# Sub domain for Dirichlet boundary condition
#class DirichletBoundary(SubDomain):
#    def inside(self, x, on_boundary):
#        return abs(x[0] - 1.0) < DOLFIN_EPS and on_boundary
def boundary_left(x):
    return x[0] < -5. + 1e-10#DOLFIN_EPS 

def boundary_right(x):
    return x[0] > 5. - 1e-10#DOLFIN_EPS

# Class representing the intial conditions

class ShuInitialConditions(UserExpression):
  def eval(self,Q,x):
    R = 1
    Cv = 5./2.
    mp = -4.  
    press = 10. + 1./3.
    rho = 3.857143
    u = 2.629369
    if (x[0] > mp):
      press = 1.
      rho = (1. + 0.2*sin(5.*x[0]) )
      u = 0 
    T = press/(rho*R)
    E = Cv*T + 0.5*u**2
    Q[0] = rho
    Q[1] = rho*u
    Q[2] = rho*E  
  def value_shape(self):
        return (3,)

# Create mesh and define function space
nx = 400
p = 2
#mesh = UnitIntervalMesh(nx)
L = 10.
mesh = IntervalMesh(nx,-5.,5.)
File("mesh.pvd") << mesh

element = VectorElement("CG", mesh.ufl_cell(), p,dim=3)
V = FunctionSpace(mesh, element)
V2 = FunctionSpace(mesh, element)


# Define boundary condition
#u0L = Constant([1.0,0.0,2.5,0,0,0]) #sod
u0L = Constant([3.857143,10.141852232767,39.1666692650425]) #sod

bcl = DirichletBC(V, u0L, boundary_left)
#u0R = Constant([0.125,0.0,0.25,0,0,0]) #sod
u0R = Constant([0.9475250292592142,0.,2.5])
bcr = DirichletBC(V, u0R, boundary_right)

# Define variational problem
U =  Function(V)
U_n = Function(V)
Phi = TestFunction(V)

Rproject =  Function(V)
PhiR = TestFunction(V)

u_init = ShuInitialConditions()
U.interpolate(u_init)
#Rproject.interpolate(u_init)

U_n.interpolate(u_init)
Rproject.interpolate(u_init)
f = Constant(1)
nu = 1e-3
b = Constant([1.,0.])
dt = 0.0005
dti = 1./dt
et = 2.0
Ux = U.dx(0)
eqn = eulerEqns1D()
Rx = eqn.evalF(U)
R_strong = eqn.evalF_Strong(U_n,U_n.dx(0))
Rortho = [None]*3
for i in range(0,3):
  Rortho[i] = R_strong[i] - Rproject[i]

Rlin = eqn.applyJ(U_n,Rortho) 

tau_rat = 0.5 
tau = tau_rat*dt#(L/nx) / rhoVal 

F = 0
for i in range(0,3):
  F -= inner(Phi[i].dx(0),Rx[i] - tau*Rlin[i])*dx  

for i in range(0,3):
  F = F + inner(Phi[i], dti*(U[i] - U_n[i]) )*dx

aW = 0
for i in range(0,3):
  aW = aW + inner(PhiR[i] , Rproject[i] )*dx 

LW = 0
for i in range(0,3):
#  LW = LW + inner(PhiR[i] , R_strong[i] )*dx
  LW = LW + PhiR[i]*R_strong[i]*dx

#F2 = aW - LW
#F2 = aW - LW
# Compute solution
t = 0
counter = 0
coor = mesh.coordinates()
q_degree = 3
dx = dx(metadata={'quadrature_degree': q_degree})


nxSave = 1000 
sol_save = np.zeros((0,3,nxSave))
def grab_sol(U):
  u_1_, u_2_, u_3_ = U.split()
  x = np.linspace(-L/2,L/2,nxSave)
  U = np.zeros((3,nxSave))
  for i in range(0,nxSave):
    U[0,i] = u_1_(x[i])
    U[1,i] = u_2_(x[i])
    U[2,i] = u_3_(x[i])

  return U 

save_freq = 10
while (t <= et - dt/2):
  file = File("SolShu/PVsol_" + str(counter) + ".pvd")
  #u_1_, u_2_, u_3_,u_4_,u_5_,u_6_ = U.split()
  #file << u_1_
  solve(aW == LW,Rproject)
  #solve(F2 == 0, Rproject, solver_parameters={"newton_solver":{"relative_tolerance": 1e-6,"linear_solver":"lu"}})
  solve(F == 0, U, [bcl,bcr], solver_parameters={"newton_solver":{"relative_tolerance": 1e-6}})
  if (counter % save_freq == 0):
    sol_save = np.append(sol_save,grab_sol(U)[None],axis=0)
  U_n.assign(U)
  t += dt
#  #plot(u,interactive=True)
  counter += 1
  print(t,counter)
#'''
#np.savez('SolShu/mz',U=sol_save)
np.savez('SolShu/mz3_nx_' + str(nx) + '_taurat_' + str(tau_rat) + '_dt_' + str(dt) ,U=sol_save,tau=tau,dt=dt)
print('Final time = ' + str(time.time() - t0))
