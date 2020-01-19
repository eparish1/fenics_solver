from dolfin import *
import sys
sys.path.append('../../src/')
from euler_3d import *
import numpy as np
import time
#from boundary_conditions import *
if has_linear_algebra_backend("Epetra"):
    parameters["linear_algebra_backend"] = "Epetra"

# Sub domain for Dirichlet boundary condition
class PeriodicDomainXYZ(SubDomain):
    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT on one of the two slave edges
        return bool(( near(x[0], 0) or near(x[1], 0) or near(x[2], 0)) and 
            (not ((near(x[0], Lx) and near(x[2], 0)) or #right front edge 
                  (near(x[0], 0 ) and near(x[2], Lz)) or #left back edge
                  (near(x[1], Ly ) and near(x[2], 0 )) or #top front edge 
                  (near(x[1], 0 ) and near(x[2], Lz )) or #bottom back edge
                  (near(x[0], Lx ) and near(x[1], 0 )) or #bottom right edge
                  (near(x[0], 0 ) and near(x[1], Ly ))    #top left edge
                 )) and on_boundary) 
#                  (near(x[1], 0 ) and near(x[2], 1 ))or #bottom back edge
#                  (near(x[0], 0 ) and near(x[1], 1 ))or #top left edge
#                  (near(x[0], 1 ) and near(x[1], 0 )))) and on_boundary)#right bottom edge
    # Map right boundary (H) to left boundary (G)
    def map(self, x, y):
        if near(x[0],Lx) and near(x[1], Ly) and near(x[2], Lz):
            y[0] = x[0] - Lx 
            y[1] = x[1] - Ly 
            y[2] = x[2] - Lz
        elif near(x[0], Lx ) and near(x[2], Lz ): #back right edge
            y[0] = x[0] - Lx
            y[1] = x[1] 
            y[2] = x[2] - Lz
        elif near(x[1], Ly ) and near(x[2], Lz ): #back top edge
            y[0] = x[0] 
            y[1] = x[1] - Ly
            y[2] = x[2] - Lz
        elif near(x[0], Lx ) and near(x[1], Ly ): #top right edge
            y[0] = x[0] - Lx 
            y[1] = x[1] - Ly
            y[2] = x[2] 
        elif near(x[0], Lx):
            y[0] = x[0] - Lx
            y[1] = x[1]
            y[2] = x[2]
        elif near(x[1], Ly):
            y[0] = x[0]
            y[1] = x[1] - Ly
            y[2] = x[2] 
        elif near(x[2], Lz ):
            y[0] = x[0]
            y[1] = x[1]
            y[2] = x[2] - Lz
        else:
            y[0] = -1000
            y[1] = -1000
            y[2] = -1000


class InitialConditionsVortex(UserExpression):
    def eval(self, Q, x):
      Lx = 2*pi
      Ly = 2*pi
      Lz = 2*pi
      gamma = 1.4
      Minf = 0.2
      T0 = 1./gamma
      R = 1.
      rho = 1.
      p0 = rho*R*T0
      a = sqrt(gamma*R*T0)
      V0 = Minf*a
      Cv = 5./2.*R
      u = V0*sin(x[0])*cos(x[1])*cos(x[2])
      v = -V0*cos(x[0])*sin(x[1])*cos(x[2])
      w = 0
      p = p0 + rho*V0**2/16.*(cos(2.*x[0]) + cos(2.*x[1]) )*(cos(2.*x[2]) + 2.)
      T = p/(rho*R)
      E = Cv*T + 0.5*(u**2 + v**2 + w**2)
      Q[0] = rho
      Q[1] = rho*u
      Q[2] = rho*v
      Q[3] = rho*w
      Q[4] = rho*E

    def value_shape(self):
        return (5,)



# Create mesh and define function space
Lx = 2.*pi
Ly = 2.*pi
Lz = 2.*pi
x0 = Point(0.,0.,0.)
x1 = Point(Lx,Ly,Lz)
mesh = BoxMesh(x0,x1,32,32,32)
File("mesh.pvd") << mesh

element = VectorElement("CG", mesh.ufl_cell(), 1,dim=5)
V = FunctionSpace(mesh, element, constrained_domain=PeriodicDomainXYZ())
V2 = FunctionSpace(mesh, element, constrained_domain=PeriodicDomainXYZ())

#f0 = Expression('sin(x[0])*cos(x)*sin(y)')

x0 = 0.
y0 = 0.
# Define boundary condition

# Define variational problem
U =  Function(V)
U_n = Function(V)
U_n2 = Function(V)
Phi = TestFunction(V)

## for additional MZ projection
Rproject = Function(V2)
PhiR = TestFunction(V2)

u_init = InitialConditionsVortex()
U.interpolate(u_init)
U_n.interpolate(u_init)

dt = 0.05
dti = 1./dt
et = 100.
Ux,Uy,Uz = U.dx(0), U.dx(1),U.dx(2)

eqn = eulerEqns3D()
Rx,Ry,Rz = eqn.evalF(U)

R_strong= eqn.evalF_Strong(U,Ux,Uy,Uz)
R_strong_n = eqn.evalF_Strong(U_n,U_n.dx(0),U_n.dx(1),U_n.dx(2) )

R_ortho = [None]*5
for i in range(0,5):
  R_ortho[i] = R_strong_n[i] - Rproject[i]

Rlin_x,Rlin_y,Rlin_z = eqn.applyJ(U_n,R_ortho,R_ortho,R_ortho)
#JR_ortho = [None]*5
#for i in range(0,5):
#  JR_ortho[i] = Rlin_x[i] + Rlin_y[i] + Rlin_z[i] 


F = 0
tau = 0.5*dt
q_degree = 3
dx = dx(metadata={'quadrature_degree': q_degree})



for i in range(0,5):
  F +=  inner(Phi[i] ,R_strong[i] )*dx + \
        inner(Phi[i].dx(0),tau*Rlin_x[i] )*dx + \
        inner(Phi[i].dx(1),tau*Rlin_y[i] )*dx + \
        inner(Phi[i].dx(2),tau*Rlin_z[i] )*dx 

#F_BDF2 = F

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


auxU_container = [U_n,U_n2]
M_BDF2 = mass_BDF2(Phi,U,auxU_container,dt)
M_BDF1 = mass_BDF1(Phi,U,auxU_container,dt)

F_BDF2 = 2./3.*F + M_BDF2
F_BDF1 = F + M_BDF1

#F2 = 0
#for i in range(0,5):
#  F2 +=  inner(PhiR[i] , Rproject[i] )*dx  - inner(PhiR[i], R_strong_n[i])*dx

a2 = 0
for i in range(0,5):
  a2 +=  inner(PhiR[i] , Rproject[i] )*dx
L2 = 0
for i in range(0,5):
  L2 += inner(PhiR[i],R_strong_n[i])*dx
F2 = a2 - L2
# Compute solution
t = 0
counter = 0
energy = np.zeros(0)
timer_s = time.time()
while (t <= et - dt/2):
  #file = File("sol_sod/sod_" + str(counter) + ".pvd")
#  u_0_, u_1_, u_2_,u_3_,u_4_ = U.split()
#  vtkfile_u_0 <<  (u_0_ , u_1_,t)
  #vtkfile_u_1 << (u_0_ , t)

#  solve(F == 0, U,[],solver_parameters={"linear_solver": "lu"})
  integral = assemble( ( 1./0.2**2*0.5*(U[1]**2 + U[2]**2 + U[3]**2)/U[0]**2)*dx)
  #solve(a2 == L2 , Rproject)
  solve(F2 == 0, Rproject, [], solver_parameters={"newton_solver":{"relative_tolerance": 1e-8, "linear_solver":"gmres"}} )
  if (counter == 0):
    solve(F_BDF1 == 0, U, [], solver_parameters={"newton_solver":{"relative_tolerance": 1e-8, "linear_solver":"gmres"}} )
  else:
    solve(F_BDF2 == 0, U, [], solver_parameters={"newton_solver":{"relative_tolerance": 1e-8, "linear_solver":"gmres"}} )
  U_n2.assign(U_n)
  U_n.assign(U)
  t += dt
  energy = np.append(energy,integral)
  np.savez('energymz_bdf2',energy=energy)
#  #plot(u,interactive=True)
  counter += 1
  sys.stdout.write('===============================' + '\n')
  sys.stdout.write('Energy = ' + str(integral) + '\n')
  sys.stdout.write('t = ' + str(t) + '\n')
  sys.stdout.write('walltime = ' + str(time.time() - timer_s) + '\n')

  sys.stdout.flush()
  #print(t,counter)
