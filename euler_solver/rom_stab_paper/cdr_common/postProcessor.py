import numpy as np
import os
nxSave = 200
def grab_sol(romProblem,sol):
  romProblem.U_working.vector()[:] = 1.*np.dot(romProblem.Phi,sol[:])
  x = np.linspace(0,1,nxSave)
  x,y = np.meshgrid(x,x,indexing='ij')
  Us = np.zeros((nxSave,nxSave))
  for i in range(0,nxSave):
    for j in range(0,nxSave):
      Us[i,j] = romProblem.U_working(x[i,j],y[i,j])
  return Us


class postProcessor:

  def __init__(self,romProblem,fom_sol,dt_fom):
    self.dt_ratio = dt_fom/romProblem.dt
    self.dt_ratio_save = np.fmax(self.dt_ratio,1)
    self.dt_integrate = np.fmax(dt_fom,romProblem.dt)
    self.error_l2 = 0.
    self.error_h1 = 0.
    self.PhiTMPhi = romProblem.PhiTMPhi
    self.PhiTHPhi = romProblem.PhiTHPhi
    self.aDOFSave = np.zeros((romProblem.K,0))
    self.tsave = np.zeros(0)
    if (romProblem.basis_type == 'L2'):
      self.aFOM = np.dot(romProblem.Phi.transpose(),np.dot(romProblem.M.array(),fom_sol )) 
    if (romProblem.basis_type == 'H1'):
      self.aFOM = np.dot(romProblem.Phi.transpose(),np.dot(romProblem.H.array(),fom_sol )) 
    ## Compute orthogonal compliment
    #fom_sol_project = np.dot(romProblem.Phi,self.aFOM)
    #uFOM_SGS     = fom_sol - fom_sol_project
    #self.error_ortho_l2 = dt_fom * np.sqrt( np.sum(uFOM_SGS[:]*np.dot(romProblem.M,uFOM_SGS[:])) )
    #self.error_ortho_h1 = dt_fom * np.sqrt( np.sum(uFOM_SGS[:]*np.dot(romProblem.H,uFOM_SGS[:])) )


    def computeProjectedErrors(aROM,aFOM):
      da =  aROM - aFOM
      self.error_l2 += np.dot(da,np.dot(self.PhiTMPhi,da) )
      self.error_h1 += np.dot(da,np.dot(self.PhiTHPhi,da) )

    def appendSavedSolution(aROM,t):
      self.aDOFSave = np.append(self.aDOFSave,aROM[:,None],axis=1)
      self.tsave = np.append(self.tsave,t)

    def postProcess(aROM,counter,t):
      if (counter % self.dt_ratio_save == 0):
        appendSavedSolution(aROM,t)
        fom_index = int(counter / self.dt_ratio)
        computeProjectedErrors(aROM,self.aFOM[:,fom_index])
    self.postProcess = postProcess
  

    def saveSol(romProblem,sol_loc):
      self.error_l2 = self.dt_integrate*np.sqrt(self.error_l2)
      self.error_h1 = self.dt_integrate*np.sqrt(self.error_h1)

      U_final = grab_sol(romProblem,self.aDOFSave[:,-1])
      if not os.path.exists(sol_loc):
        os.mkdir(sol_loc)
      np.savez(sol_loc + 'usave',K=romProblem.K,xhat=self.aDOFSave,error_l2p=self.error_l2,error_h1p=self.error_h1,dt=romProblem.dt,t=self.tsave,U_final=U_final,tauROM=romProblem.tau,tauFEM=romProblem.femProblem.tau)#,error_l2=(self.error_l2+self.error_ortho_l2),error_h1=(self.error_h1+self.error_ortho_h1))
    self.saveSol = saveSol 
