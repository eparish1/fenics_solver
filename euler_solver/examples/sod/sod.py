from dolfin import *
import numpy as np
from navier_stokes import *
if has_linear_algebra_backend("Epetra"):
    parameters["linear_algebra_backend"] = "Epetra"

# Sub domain for Dirichlet boundary condition
#class DirichletBoundary(SubDomain):
#    def inside(self, x, on_boundary):
#        return abs(x[0] - 1.0) < DOLFIN_EPS and on_boundary
def boundary_left(x):
    return x[0] < DOLFIN_EPS 

def boundary_right(x):
    return x[0] > 1.0 - DOLFIN_EPS

# Class representing the intial conditions
class InitialConditions(UserExpression):
    def eval(self, values, x):
        if (x < 0.5):
          values[0] = 1.0
          values[2] = 2.5 
        else:
          values[0] = 0.125
          values[2] = 0.25 

        values[1] = 0.0
    def value_shape(self):
        return (3,)


# Create mesh and define function space
nx = 400 
mesh = UnitIntervalMesh(nx)
File("mesh.pvd") << mesh

element = VectorElement("CG", mesh.ufl_cell(), 2,dim=3)
V = FunctionSpace(mesh, element)


# Define boundary condition
u0L = Constant([1.0,0.0,2.5])
bcl = DirichletBC(V, u0L, boundary_left)
u0R = Constant([0.125,0.0,0.25])
bcr = DirichletBC(V, u0R, boundary_right)

# Define variational problem
U =  Function(V)
U_n = Function(V)
Phi = TestFunction(V)

u_init = InitialConditions()
U.interpolate(u_init)
U_n.interpolate(u_init)
f = Constant(1)
nu = 1e-3
b = Constant([1.,0.])
dt = 0.0005
dti = 1./dt
et = 0.2
Ux = U.dx(0)
Rx = eulerInviscidFlux(U)
R_strong = strongFormResidNS(U,Ux)
R_strong += dti*(U - U_n)
Rlin = evalInviscidFluxLinNS(U,R_strong) 
F = 0
#
rho = U[0]
rhoU = U[1]
rhoE = U[2]
u = rhoU/rho
gamma = 1.4
p = (gamma - 1.)*(rhoE - 0.5*rhoU**2/rho)
a = sqrt(gamma*p/rho)
M = ( rhoU/rho ) / a
eig1 = u + a
eig2 = a
eig3 = u - a
rhoVal = sqrt( (u + a)**2 + a**2 + (u - a)**2 )
tau = dt#1.0*(1./nx) / rhoVal 
#
nxSave = 1000 
sol_save = np.zeros((0,3,nxSave))
def grab_sol(U):
  u_1_, u_2_, u_3_ = U.split()
  x = np.linspace(0,1,nxSave)
  U = np.zeros((3,nxSave))
  for i in range(0,nxSave):
    U[0,i] = u_1_(x[i])
    U[1,i] = u_2_(x[i])
    U[2,i] = u_3_(x[i])

  return U 


for i in range(0,3):
  F -= inner(Phi[i].dx(0),Rx[i] - tau*Rlin[i])*dx  
for i in range(0,3):
  F = F + inner(Phi[i], dti*(U[i] - U_n[i]) )*dx
# Compute solution
t = 0
counter = 0
while (t <= et - dt/2):
#  file = File("sol/poisson_" + str(counter) + ".pvd")
#  _u_1, _u_2 = U.split()
#  file << _u_1
  solve(F == 0, U, [bcl,bcr], solver_parameters={"newton_solver":{"relative_tolerance": 1e-6}})
  sol_save = np.append(sol_save,grab_sol(U)[None],axis=0)
  U_n.assign(U)
  t += dt
#  #plot(u,interactive=True)
  counter += 1
  print(t,counter)
#'''

np.savez('Sol/sol_supg',U=sol_save)

