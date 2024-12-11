import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from linear_regression.linear_regression import *
from scipy.stats import norm

####################################################
####################################################
def MSE(X,Y,f,l):
  n=len(X)
  theta_l=Estimator(X,Y,f,l)
  phi=Phi(X,f)
  return 1/n*np.linalg.norm(Y-phi@theta_l)**2

def GraphMSE(X,Y,f,l_min,l_max,smooth,color,label):
  l=np.linspace(l_min,l_max,num=smooth)
  M=np.zeros(smooth)
  for i in range(0,smooth):
    M[i]=MSE(X,Y,f,l[i])
  plt.plot(l, M,color,label=label)
  plt.legend(loc="upper left");  plt.grid();  plt.show()
  
def GraphErrorDistributionAsDiscrete(X,Y,f,l,n_part,smooth,x_min,x_max):
  error=EstimationError(X,Y,f,l); n=len(error)
  F_error=np.zeros(n_part)
  Error=np.linspace(min(error),max(error),n_part)
  mean=np.mean(error); std=np.std(error)
  error=np.sort(error);  delta=(error[n-1]-error[0])/(n_part-1)
  for i in range(0,n_part):
    if i==0:
      F_error[i]=0
    else:
      c=0
      for j in range(0,n):
        if error[j]<=error[0]+i*delta:
          c=c+1
      F_error[i]=c/n
  error_x=np.zeros(n_part-1)
  for i in range(0,n_part-1):
    error_x[i]=(Error[i+1]+Error[i])/2.0
  P_error=np.zeros(n_part-1)
  for i in range(0,n_part-1):
    P_error[i]=(F_error[i+1]-F_error[i])
  x_axis=np.linspace(x_min,x_max,smooth)
  print('-------------------------------------------------------')
  print(f'Distribución acumulada del error:')
  plt.plot(Error, F_error,'o',color='magenta',label= 'Error')
  plt.plot(x_axis, norm.cdf(x_axis, mean, std),label=f's={round(std,2)}')
  plt.legend(loc="upper left");  plt.grid();  plt.show()
  # print(error_x); print(P_error)
  print('-------------------------------------------------------')
  print(f'Probabilidad del error:')
  plt.plot(error_x, P_error,'o',color='magenta',label='Error')
  plt.plot(x_axis, norm.pdf(x_axis, mean, std),label=f's={round(std,2)}')
  plt.legend(loc="upper left");  plt.grid();  plt.show()
  print(f'Media: {mean}')
  print(f'Desviación: {std}')
  
def GraphErrorDistributionAsContinuous(X,Y,f,l,n_part,smooth,x_min,x_max):
  error=EstimationError(X,Y,f,l); n=len(error)
  F_error=np.zeros(n_part);  P_error=np.zeros(n_part)
  Error=np.linspace(min(error),max(error),n_part)
  mean=np.mean(error); std=np.std(error); error=np.sort(error)
  delta=(error[n-1]-error[0])/(n_part-1)
  for i in range(0,n_part):
    if i==0:
      F_error[i]=0
    else:
      c=0
      for j in range(0,n):
        if error[j]<=error[0]+i*delta:
          c=c+1
      F_error[i]=c/n
  k=0; c=0
  for i in range(0,n_part-1):
    # print(i,F_error[i+1],F_error[i],c); print('i')
    if F_error[i+1] != F_error[i]:
      if i+c+1>=n_part:
        break
      else:
        for j in range(i,i+c+1):
          # print(k,error[k],error[k+1],c)
          P_error[j]=(F_error[i+1]-F_error[i])/(error[k+1]-error[k])
        c=0
        k=k+1
        # print('')
    else:
      c=c+1
  x_axis=np.linspace(x_min,x_max,smooth)
  print('-------------------------------------------------------')
  print(f'Distribución acumulada del error:')
  plt.plot(Error, F_error,color='magenta',label= 'Error')
  plt.plot(x_axis, norm.cdf(x_axis, mean, std),label=f's={round(std,2)}')
  plt.legend(loc="upper left")
  plt.grid()
  plt.show()
  # print(error_x); print(P_error)
  print('-------------------------------------------------------')
  print(f'Probabilidad del error:')
  plt.plot(Error, P_error,'o',color='magenta',label='Error')
  plt.plot(x_axis, norm.pdf(x_axis, mean, std),label=f's={round(std,2)}')
  plt.legend(loc="upper left")
  plt.grid()
  plt.show()
  print(f'Media: {mean}')
  print(f'Desviación: {std}')
  
####################################################
####################################################
def Print_error_table(X,Y,f,l):
    error=EstimationError(X,Y,f,l)
    error=np.sort(error)
    print('——————————————')
    print('Sorted error')
    print('——————————————')
    for i in range(0,len(X)):
        print(f'{round(error[i],5)}')
        print('——————————————')
####################################################
####################################################

def P_n(x,k): #k: degree of polynomial
  P=np.zeros(k+1)
  for i in range(0,k+1):
    if i==0:
      P[k-i]=1
    else:
      P[k-i]=x**i
  return P

def GraphMSE_per_P_n(X,Y,l,k_min,k_max):
  k=np.linspace(k_min,k_max,num=k_max-k_min+1)
  MSE_k=np.zeros(k_max-k_min+1)
  for i in range(0,k_max-k_min+1):
    def f(x):
      return P_n(x,i+k_min)
    MSE_k[i]=MSE(X,Y,f,l)
  plt.plot(k, MSE_k,'o',color='green',label='(k,MSE(P_k))')
  plt.plot(k, MSE_k,color='green',label='(k,MSE(P_k))')
  plt.legend(loc="upper right")
  # plt.gca().set_aspect('equal')
  plt.grid()
  plt.show()