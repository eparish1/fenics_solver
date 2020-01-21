from dolfin import *
import sys
sys.path.append('../../src/')
from navier_stokes_3d import *
import numpy as np
from list_container import *
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
mesh = BoxMesh(x0,x1,16,16,16)
File("mesh.pvd") << mesh
poly_order = 3
element = VectorElement("CG", mesh.ufl_cell(), poly_order,dim=5)
V = FunctionSpace(mesh, element, constrained_domain=PeriodicDomainXYZ())


x0 = 0.
y0 = 0.

# Define variational problem
U =  Function(V)
Utrial = TrialFunction(V)

U_n = Function(V)
Phi = TestFunction(V)

## for additional MZ projection
Rproject = Function(V)


u_init = InitialConditionsVortex()
U.interpolate(u_init)
U_n.interpolate(u_init)
tau = 0.#1./64.
dt = 0.1
dti = 1./dt
et = 50.
Ux,Uy,Uz = U.dx(0), U.dx(1),U.dx(2)

eqn = navierStokesEqns3D(mu=1./1600.*0.2)
#Rx,Ry,Rz = eqn.evalF(U)
GC = eqn.evalGs(U)
GC_grad = eqn.evalGsGradients(U,Ux,Uy,Uz)
JTPhi_x,JTPhi_y,JTPhi_z = eqn.applyJT(U, Phi.dx(0),Phi.dx(1),Phi.dx(2) )
R_strong  = eqn.evalF_Strong(U,Ux,Uy,Uz)
R_strong_n  = eqn.evalF_Strong(U_n,U_n.dx(0),U_n.dx(1),U_n.dx(2))

R_ortho = listContainer_1d(5)
for i in range(0,5):
  R_ortho[i] = R_strong[i] - Rproject[i]
tau = 0.01#0.5*dt
q_degree = 6
dx = dx(metadata={'quadrature_degree': q_degree})

Gx =   containerMatTransposeVec(GC[0][0],Phi.dx(0)) + \
       containerMatTransposeVec(GC[1][0],Phi.dx(1)) + \
       containerMatTransposeVec(GC[2][0],Phi.dx(2))

Gy =   containerMatTransposeVec(GC[0][1],Phi.dx(0)) + \
       containerMatTransposeVec(GC[1][1],Phi.dx(1)) + \
       containerMatTransposeVec(GC[2][1],Phi.dx(2))

Gz =   containerMatTransposeVec(GC[0][2],Phi.dx(0)) + \
       containerMatTransposeVec(GC[1][2],Phi.dx(1)) + \
       containerMatTransposeVec(GC[2][2],Phi.dx(2))


GTX = containerMatTransposeVec(GC_grad[0][0],Phi.dx(0)) + \
      containerMatTransposeVec(GC_grad[1][0],Phi.dx(1)) + \
      containerMatTransposeVec(GC_grad[2][0],Phi.dx(2))

GTY = containerMatTransposeVec(GC_grad[0][1],Phi.dx(0)) + \
      containerMatTransposeVec(GC_grad[1][1],Phi.dx(1)) + \
      containerMatTransposeVec(GC_grad[2][1],Phi.dx(2))

GTZ = containerMatTransposeVec(GC_grad[0][2],Phi.dx(0)) + \
      containerMatTransposeVec(GC_grad[1][2],Phi.dx(1)) + \
      containerMatTransposeVec(GC_grad[2][2],Phi.dx(2))

F = 0
for i in range(0,5):
  F +=  inner(Phi[i] ,R_strong[i] )*dx + \
        inner(JTPhi_x[i] + JTPhi_y[i] + JTPhi_z[i],tau*R_ortho[i] )*dx + \
        inner(Gx[i] , U[i].dx(0))*dx + inner(Gy[i] , Uy[i])*dx + inner(Gz[i],Uz[i])*dx 
for i in range(0,4):
  F +=  inner(GTX[i] + GTY[i] + GTZ[i], tau*R_ortho[i])*dx


for i in range(0,5):
  F = F + inner(Phi[i], dti*(U[i] - U_n[i]) )*dx

## compute MZ projection bilinear form
aproj = inner(Phi,Utrial)*dx
Lproj = 0
for i in range(0,5):
  Lproj += inner(Phi[i],R_strong_n[i])*dx

sol_file = 'Solution_mztau_' + str(tau) + '_p3_dt' + str(dt) 
Hdf = HDF5File(mesh.mpi_comm(),sol_file + '/sol.hdf',"w")
Hdf.write(mesh,"mesh")
# Compute solution
t = 0
counter = 0
energy = np.zeros(0)

save_freq = 10
tsave = np.zeros(0)

info_dict = {'dt':dt,'tau':tau,'polynomial order':poly_order,'quadrature points':q_degree}
fl = open(sol_file + '/settings.dat',"w")
fl.write( str(info_dict) )
fl.close()
#np.savez(sol_file + '/settings',information=info_dict,params = eqn.params)
while (t <= et - dt/2):
  if (counter%save_freq == 0):
    Hdf.write(U,"u",counter)
  
  # comptue TKE
  integral = assemble( ( 1./0.2**2*0.5*(U[1]**2 + U[2]**2 + U[3]**2)/U[0]**2)*dx)
  energy = np.append(energy,integral)
  tsave = np.append(tsave,t)
  np.savez(sol_file + '/energy',energy=energy,t=tsave)
  solve(aproj == Lproj ,Rproject,solver_parameters={"linear_solver":"cg"})
  solve(F == 0, U, [], solver_parameters={"newton_solver":{"relative_tolerance": 1e-8, "linear_solver":"gmres"}} )
  U_n.assign(U)
  t += dt
  counter += 1
  sys.stdout.write('Energy = ' + str(integral) + '\n')
  sys.stdout.write('t = ' + str(t) + '\n')
  sys.stdout.flush()
