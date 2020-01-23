import numpy as np

class listContainer_2d:
  def __init__(self,size1,size2):
    self.size1 = size1
    self.size2 = size2
    self.obj = [[0. for i in range(size1)] for j in range(size2)]

  ## overload the [] operator
  def __getitem__(self,i):
      return self.obj[i]
  def __setitem__(self,indx,val):
    self.obj[indx] = val

  def shape(self):
    return self.size1,self.size2

class listContainer_1d:
  def __init__(self,size1):
    self.size1 = size1
    self.obj = [0]*size1
  ## overload the [] operator
  def __getitem__(self,i):
    return self.obj[i]
  def __setitem__(self,indx,val):
    self.obj[indx] = val
    

  def size(self):
    return self.size1

  def __sub__(self,obj):
    c = listContainer_1d(self.size1)
    for i in range(0,self.size1):
      c[i] = self.obj[i] - obj[i]
    return c

  def __add__(self,obj):
    c = listContainer_1d(self.size1)
    for i in range(0,self.size1):
      c[i] = self.obj[i] + obj[i]
    return c

  def dx(self,index):
    cx = listContainer_1d(self.size1)
    for i in range(0,self.size1):
      cx[i] = self.obj[i].dx(index)
    return cx
def containerMatVec(M,Q):
  ''' 
  computes the matrix vector product MQ
  '''
  N,K = M.shape()
  MQ = listContainer_1d(N)
  for i in range(0,N):
    for j in range(0,K):
        MQ[i] = MQ[i] + M[i][j]*Q[j]
  return MQ

def containerMatTransposeVec(M,Q):
  ''' 
  computes the matrix vector product M^T Q
  '''
  N,K = M.shape() 
  MQ = listContainer_1d(K)
  for i in range(0,K):
    for j in range(0,N):
      MQ[i] = MQ[i] + M[j][i]*Q[j]
  return MQ



