import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

####################################################
####################################################
def Phi(X,f):
  n=len(X); d=len(f(1))
  phi=np.zeros((n,d))
  for i in range(0,n):
    varphi_iT=f(X[i])
    for j in range(0,d):
      phi[i][j]=varphi_iT[j]
  return phi

def Estimator(X,Y,f,l):
  phi=Phi(X,f); n=len(Y); d=len(phi[0,:])
  phiT=np.transpose(phi)
  return np.linalg.solve(phiT@phi+n*l*np.identity(d),phiT@Y)

def Prediction(w,X,Y,f,l):
  theta=Estimator(X,Y,f,l)
  return f(w)@theta

def thetaPrediction(w,theta,f): #Prediction with theta but free
  return f(w)@theta

def GraphPrediction(X,Y,f,l,Inf,Sup,smooth,color,label):
  n=len(X);  AxisX=np.linspace(Inf,Sup,num=smooth);  AxisY=np.linspace(Inf,Sup,num=smooth)
  for i in range(0,smooth):
    AxisY[i]=Prediction(AxisX[i],X,Y,f,l)
  return plt.plot(AxisX,AxisY,color,label=label)

def GraphThetaPrediction(theta,f,Inf,Sup,smooth,color,label): #GraphPrediction with theta but free
  AxisX=np.linspace(Inf,Sup,num=smooth);  AxisY=np.linspace(Inf,Sup,num=smooth)
  for i in range(0,smooth):
    AxisY[i]=thetaPrediction(AxisX[i],theta,f)
  return plt.plot(AxisX,AxisY,color,label=label)

def EstimationError(X,Y,f,l):
  n=len(X);  error=np.zeros(n)
  for i in range(0,n):
    # print(Y[i]); print(f'x{Prediction(X[i],X,Y,l)}')
    error[i]=Y[i]-Prediction(X[i],X,Y,f,l)
  return error

####################################################
####################################################
