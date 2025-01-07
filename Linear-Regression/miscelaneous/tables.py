import numpy as np
import pandas as pd
from linear_regression.linear_regression import *

####################################################
####################################################
def PrintVector(v):
  n=len(v)
  for i in range(0,n):
    print(f'{v[i]}',end='\t')
  print('')
  
def PrintVectorH(v):
  n=len(v)
  for i in range(0,n):
    if i<n-1:
      print(f'{v[i]}',end='\t')
    else:
      print(f'{v[i]}')
  print('')

def PrimitivePrintTable(X,Y,f,l):
  error=EstimationError(X,Y,f,l)
  hatY=np.zeros(len(X))
  for i in range(0,len(X)):
    hatY[i]=Prediction(X[i],X,Y,f,l)
  print('———————————————————————————————————————————————————————————')
  print('X \t Y \t hat(Y) \t Error')
  print('———————————————————————————————————————————————————————————')
  for i in range(0,len(X)):
    print(f'{X[i]} \t {Y[i]} \t {round(hatY[i],5)} \t {round(error[i],5)}')
  print('———————————————————————————————————————————————————————————')

def PrintTable(X,Y,f,l):
  error=EstimationError(X,Y,f,l)
  hatY=np.zeros(len(X))
  for i in range(0,len(X)):
    hatY[i]=Prediction(X[i],X,Y,f,l)
  Data=np.array([X,Y,hatY,error])
  Data=np.transpose(Data)
  AuxData=np.zeros((len(Data[:,0])*len(Data[0,:]),2)) #Starts the algorithm
  for i in range(0,len(Data[:,0])):
    for j in range(0,len(Data[0,:])):
      AuxData[i*len(Data[0,:])+j,0]=Data[i,j]
      AuxData[i*len(Data[0,:])+j,1]=i #useless?
  DataX=Data[:,0]
  DataX=np.sort(DataX)
  NewData=Data
  NewData[:,0]=DataX
  for i in range(0,len(NewData[:,0])):
    for k in range(0,len(NewData[:,0])):
      if NewData[i,0]==AuxData[k*len(Data[0,:]),0]:
        for j in range(0,len(NewData[0,:])):
          NewData[i,j]=AuxData[k*len(Data[0,:])+j,0]
  Data=NewData
  print('———————————————————————————————————————————————————————————')
  print('X \t Y \t hat(Y) \t Error')
  print('———————————————————————————————————————————————————————————')
  for i in range(0,len(X)):
    print(f'{Data[i][0]} \t {Data[i][1]} \t {round(Data[i][2],5)} \t {round(Data[i][3],5)}')
  print('———————————————————————————————————————————————————————————')
  
def PrintTable_v2(X,Y,f,l,dec):
  error=EstimationError(X,Y,f,l)
  for i in range(len(error)):
    error[i]=round(error[i],dec)
  hatY=np.zeros(len(X))
  for i in range(0,len(X)):
    hatY[i]=round(Prediction(X[i],X,Y,f,l),dec)
  return pd.DataFrame({'X':X,'Y':Y,'hat(Y)':hatY,'error':error})

