import numpy as np
import os
from dolfin import *
nxSave = 200
def grab_sol(femProblem,sol):
  femProblem.U_working.vector()[:] = 1.*sol[:]
  x = np.linspace(0,1,nxSave)
  x,y = np.meshgrid(x,x,indexing='ij')
  Us = np.zeros((nxSave,nxSave))
  for i in range(0,nxSave):
    for j in range(0,nxSave):
      Us[i,j] = femProblem.U_working(x[i,j],y[i,j])
  return Us


class femPostProcessor:

  def __init__(self,femProblem,femCoarseProblem,sol_loc,save_freq):
    self.sol_loc = sol_loc
    self.UDOFSave = np.zeros((femCoarseProblem.N,0))
    self.tsave = np.zeros(0)
    self.save_freq = save_freq
    self.save_freq_vtk = 10000
    self.U_coarse = femCoarseProblem.U_working

    def saveToVtk(U,counter):
      file = File(self.sol_loc + "fem_" + str(counter) + ".pvd")
      file << U

    def appendSavedSolution(UDOF,t):
      self.UDOFSave = np.append(self.UDOFSave,UDOF[:,None],axis=1)
      self.tsave = np.append(self.tsave,t)

    def postProcess(U,counter,t,femCoarseProblem):
      if (counter % self.save_freq == 0):
        U_coarse = project(U,femCoarseProblem.functionSpace)
        appendSavedSolution(U_coarse.vector()[:]*1.,t)
        if (counter % self.save_freq_vtk == 0):
          saveToVtk(U,counter)

    self.postProcess = postProcess
  

    def saveSol(femProblem,femCoarseProblem):
      U_final = grab_sol(femCoarseProblem,self.UDOFSave[:,-1])
      if not os.path.exists(self.sol_loc):
        os.mkdir(self.sol_loc)

      skip = 4
      K = np.dot( self.UDOFSave[:,::skip].transpose(), np.dot(femCoarseProblem.M.array(),self.UDOFSave[:,::skip]))
      ub,sb,vb = np.linalg.svd(K)
      Phi = np.dot(self.UDOFSave[:,::skip] ,1./np.sqrt(sb+1e-30)*ub )

      K = np.dot( self.UDOFSave[:,::skip].transpose(), np.dot(femCoarseProblem.H.array(),self.UDOFSave[:,::skip]))
      ub,sb_h,vb = np.linalg.svd(K)
      PhiH = np.dot(self.UDOFSave[:,::skip] ,1./np.sqrt(sb_h+1e-30)*ub )

      np.savez(self.sol_loc + '/pod_basis',UDOFSave=self.UDOFSave,Phi=Phi,sigma=sb,PhiH=PhiH,sigmaH=sb_h,U_final=U_final,MC=femCoarseProblem.M.array(),HC=femCoarseProblem.H.array())
    self.saveSol = saveSol 
