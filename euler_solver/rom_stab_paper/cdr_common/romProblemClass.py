from dolfin import *
import numpy as np
class romProblemClass:
  def __init__(self,functionSpace,p,Phi,basis_type,methodContinuous,methodDiscrete,dt,et):
    self.basis_type = basis_type
    self.methodContinuous = methodContinuous
    self.methodDiscrete = methodDiscrete
    self.functionSpace = functionSpace
    self.U =  TrialFunction(functionSpace)
    self.U_working = Function(functionSpace)
    self.V = TestFunction(functionSpace)
    self.M = assemble( self.V * self.U * dx )
    self.Minv = np.linalg.inv(self.M.array())
    self.H = assemble( inner(grad(self.V) , grad( self.U ) )*dx )
    self.Phi = Phi 
    self.PhiTMPhi = np.dot(Phi.transpose(),np.dot(self.M.array(),Phi)) 
    self.PhiTHPhi = np.dot(Phi.transpose(),np.dot(self.H.array(),Phi))
    self.dt = dt
    self.et = et
    self.K = np.shape(Phi)[1]
    self.N = np.shape(Phi)[0]
    self.p = p 
