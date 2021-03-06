from dolfin import *
from navier_stokes import *
if has_linear_algebra_backend("Epetra"):
    parameters["linear_algebra_backend"] = "Epetra"

# Sub domain for Dirichlet boundary condition
#class DirichletBoundary(SubDomain):
#    def inside(self, x, on_boundary):
#        return abs(x[0] - 1.0) < DOLFIN_EPS and on_boundary
def boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

# Class representing the intial conditions
class InitialConditions(UserExpression):
    def eval(self, values, x):
        values[0] = 1.0
        values[1] = 1.0
        values[2] = 1.0
    def value_shape(self):
        return (3,)


# Create mesh and define function space
mesh = UnitIntervalMesh(64)
File("mesh.pvd") << mesh

element = VectorElement("CG", mesh.ufl_cell(), 1,dim=3)
V = FunctionSpace(mesh, element)


# Define boundary condition
u0 = Constant([0.0,0.,0.])
bc = DirichletBC(V, u0, boundary)
# Define variational problem
U =  Function(V)
U_n = Function(V)
Phi = TestFunction(V)

u_init = InitialConditions()
U.interpolate(u_init)

f = Constant(1)
nu = 1e-3
b = Constant([1.,0.])
tau = 1./64.
dt = 0.01
dti = 1./dt
et = 1.
F = eulerInviscidWeak(U,Phi)
for i in range(0,3):
  F = F + inner(Phi[i], dti*(U[i] - U_n[i]) )*dx
#r = dti*(u - u_nm1) +  dot(b,grad(u) ) - f 
#F = inner(v1,dti*(u - u_nm1))*dx + nu*inner(grad(v1), grad(u))*dx - inner( dot(b, grad(v1)), u  - tau*(r - rp))*dx
#F = F - inner(f,v1)*dx 
#
#F2 = inner( v2 , rp )*dx - inner( v2 , r )*dx
#F = F + F2
#test = Function(V)
# Compute solution
#t = 0
#counter = 0
#while (t <= et - dt/2):
#  file = File("sol/poisson_" + str(counter) + ".pvd")
#  _u_1, _u_2 = U.split()
#  file << _u_1
solve(F == 0, U, bc, solver_parameters={"newton_solver":{"relative_tolerance": 1e-6}})

#  U_n.assign(U)
#  t += dt
#  #plot(u,interactive=True)
#  counter += 1
#  print(t,counter)
#'''
