from dolfin import *
import numpy as np
class femProblemClass:
  def __init__(self,N,p,methodContinuous,dt,et,tau):
    mesh = UnitSquareMesh(N,N)
    self.mesh = mesh
    self.functionSpace = FunctionSpace(mesh, 'CG',p)
    self.methodContinuous = methodContinuous
    self.U =  TrialFunction(self.functionSpace)
    self.U_nm1 = Function(self.functionSpace)
    self.U_working = Function(self.functionSpace)
    self.V = TestFunction(self.functionSpace)
    self.M = assemble( self.V * self.U * dx )
    self.H = assemble( inner(grad(self.V) , grad( self.U ) )*dx )
    self.dt = dt
    self.et = et
    self.N =  np.size(self.U_working.vector()[:]) 
    self.p = p
    self.tau = tau 
