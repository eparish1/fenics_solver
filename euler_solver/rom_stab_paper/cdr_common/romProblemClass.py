from dolfin import *
import numpy as np
class romProblemClass:
  def __init__(self,femProblem,Phi,basis_type,methodContinuous,methodDiscrete,dt,et,tau):
    self.femProblem = femProblem
    self.basis_type = basis_type
    self.methodContinuous = methodContinuous
    self.methodDiscrete = methodDiscrete
    self.functionSpace = femProblem.functionSpace
    self.U =  femProblem.U 
    self.U_working = femProblem.U_working 
    self.V = femProblem.V 
    self.M = femProblem.M 
    self.Minv = np.linalg.inv(self.M.array())
    self.H = femProblem.H 
    self.Phi = Phi 
    self.PhiTMPhi = np.dot(Phi.transpose(),np.dot(self.M.array(),Phi)) 
    self.PhiTHPhi = np.dot(Phi.transpose(),np.dot(self.H.array(),Phi))
    self.dt = dt
    self.et = et
    self.K = np.shape(Phi)[1]
    self.N = np.shape(Phi)[0]
    self.p = femProblem.p 
    self.tau = tau
