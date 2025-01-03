from copy import error
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import statistics

X=np.array([-2.0,0,1.1,1.6,3.1,4.1,4.8,5.2,5.5,7,8,9,9.5,10.2,0.6,9.1,2,2.4,-1.8,-1.4,-0.7,10.1,15.5,13.2,11.2,16,10.5,12.7,14.8,2.6,3.5,-1.6,7.2,7.3,7.4,-3.1,-2.3,-0.8,-0.2,1.8,2.1,2.9,4.0,5.7,6.0,6.3,6.9,8.1,8.2,14])
Y=np.array([-1.1,2,0,1.9,2.8,3.2,4.8,7.1,7.5,6.6,10.3,11.1,8,9.7,1.2,10.0,-5.1,-1.6,-9.0,-7.3,-3.8,8.0,16.2,13,10.3,15.3,12.5,11.8,12.5,1.2,1.9,2,7.2,6.9,6.5,-3.0,-2.0,0.5,0.1,1.5,2.1,2.8,3.7,5.5,6.4,6.2,7.0,8.0,7.5,13.5])

#######################################
l=0.275 #Penalty
MaxX=max(X); MinX=min(X)
#######################################
def P_n(x,k):
  P=np.zeros(k+1)
  for i in range(0,k+1):
    if i==0:  P[k-i]=1
    else: P[k-i]=x**i
  return P

def f(x): #d<=n (to be fullfilled MCO)
  return np.array([x**3,x**2,x,1])

def Phi(X,f):
  n=len(X); d=f(1); d=len(d)
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

smooth=250
def GraphPrediction(X,Y,f,l,Inf,Sup,smooth,color,label):
  n=len(X);  AxisX=np.linspace(Inf,Sup,num=smooth);  AxisY=np.linspace(Inf,Sup,num=smooth)
  for i in range(0,smooth):
    AxisY[i]=Prediction(AxisX[i],X,Y,f,l)
  return plt.plot(AxisX,AxisY,color,label=label)

def EstimationError(X,Y,f,l):
  n=len(X);  error=np.zeros(n)
  for i in range(0,n):
    # print(Y[i]); print(f'x{Prediction(X[i],X,Y,l)}')
    error[i]=Y[i]-Prediction(X[i],X,Y,f,l)
  return error

def PrintVector(v):
  n=len(v)
  for i in range(0,n):
    print(f'{round(v[i],5)}',end='\t')
  print('')

def PrintVectorH(v):
  n=len(v)
  for i in range(0,n):
    print(f'{round(v[i],5)}',end='\t')
  print('')

def PrintTable(X,Y,f,l):
  error=EstimationError(X,Y,f,l);  hatY=np.zeros(len(X))
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
  NewData=Data;  NewData[:,0]=DataX
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

def MSE(X,Y,f,l):
  n=len(X);  theta_l=Estimator(X,Y,f,l);  phi=Phi(X,f)
  return 1/n*np.linalg.norm(Y-phi@theta_l)**2

def CrossValidation(X,Y,f,l,E): #Para el costo norma 2
  E_c=GetVectorComplement(X,E)
  Xi=GetVectorFromIndexes(X,E) #Entrenamiento
  Yi=GetVectorFromIndexes(Y,E) #Entrenamiento
  Xi_complement=GetVectorFromIndexes(X,E_c) #Validacion
  Yi_complement=GetVectorFromIndexes(Y,E_c) #Validacion
  card=len(Xi)
  return 1/card*np.linalg.norm(Yi_complement-Phi(Xi_complement,f)@Estimator(Xi,Yi,f,l))**2

def PrePrintTable(X,Y,f,l):
  error=EstimationError(X,Y,f,l);  hatY=np.zeros(len(X))
  for i in range(0,len(X)):
    hatY[i]=Prediction(X[i],X,Y,f,l)
  print('———————————————————————————————————————————————————————————')
  print('X \t Y \t hat(Y) \t Error')
  print('———————————————————————————————————————————————————————————')
  for i in range(0,len(X)):
    print(f'{X[i]} \t {Y[i]} \t {round(hatY[i],5)} \t {round(error[i],5)}')
  print('———————————————————————————————————————————————————————————')

print('-------------------------------------------------------')
print('----------------------FULL DATA-----------------------')
print('-------------------------------------------------------')
print(f'Los {len(X)} datos y las predicciones son:')
# PrintTable(X,Y,0)
PrintTable(X,Y,f,l)
print('-------------------------------------------------------')
print(f'El estimador Non-Pen (MSE={round(MSE(X,Y,f,0),5)}) es:')
# print(f'{Estimator(X,Y,f,l)}')
PrintVectorH(Estimator(X,Y,f,0))
print('-------------------------------------------------------')
print(f'El estimador Ridge (MSE={round(MSE(X,Y,f,l),5)}) es:')
# print(f'{Estimator(X,Y,f,l)}')
PrintVectorH(Estimator(X,Y,f,l))
print('-------------------------------------------------------')
print('------------------GRAPHING FULL DATA-------------------')
print('-------------------------------------------------------')
Inf=MinX-0.1; Sup=MaxX+0.1
GraphPrediction(X,Y,f,0,Inf,Sup,smooth,'red','Non-Pen')
GraphPrediction(X,Y,f,l,Inf,Sup,smooth,'blue',f'Ridge (l={l})')
plt.plot(X, Y,'o',color='black',label= '$(x_i,y_i)$')
plt.legend(loc="upper left"); plt.grid(); plt.show()

def GetVectorFromIndexes(v,index_vector):
  n=len(index_vector)
  vector=np.zeros(n)
  for i in range(0,n):
    vector[i]=v[int(index_vector[i])]
  return vector

def GetVectorComplement(v,index_vector):
  L=len(v); n=len(index_vector)
  index_vector_complement=np.zeros(L)
  for j in range(0,L):
    index_vector_complement[j]=j
  for j in range(0,n):
    index_vector_complement=np.delete(index_vector_complement,np.where(index_vector_complement==index_vector[j]))
  return index_vector_complement

def SwapVectors(u,v): #Swap vectors
  aux=u; u=v; v=aux; return u,v

#######################################
#Selection of partitions
#######################################

print(''); print('-------------------------------------------------------')
print('-----------------GRAPHING PARTIAL DATA-----------------')
print('-------------------------------------------------------'); print('')

def PrintCrossValidationStats(X,Y,f,l,index_vector,index_vector_complement,i):
    Xi=GetVectorFromIndexes(X,index_vector) #Training
    Yi=GetVectorFromIndexes(Y,index_vector) #Training
    Xi_complement=GetVectorFromIndexes(X,index_vector_complement) #Validation
    Yi_complement=GetVectorFromIndexes(Y,index_vector_complement) #Validation
    print('-------------------------------------------------------')
    print(f'-------------------PART {i+1} DATA-------------------')
    print('-------------------------------------------------------')
    print(f'El estimador Non-Pen (MSE{i+1}={round(MSE(Xi,Yi,f,0),5)}) es:')
    theta_i=Estimator(Xi,Yi,f,0); PrintVectorH(theta_i)
    print('-------------------------------------------------------')
    print(f'El estimador Ridge (MSE{i+1}={round(MSE(Xi,Yi,f,l),5)}) es:')
    theta_i_lambda=Estimator(Xi,Yi,f,l); PrintVectorH(theta_i_lambda)
    print('-------------------------------------------------------')
    print(f'La Validacion Cruzada Ridge (VAL{i+1}) es: {round(CrossValidation(X,Y,f,l,index_vector),5)}')
    print('-------------------------------------------------------')
    print(f'-----------------GRAPHING PART {i+1} DATA-----------------')
    print('-------------------------------------------------------')
    GraphPrediction(Xi,Yi,f,0,Inf,Sup,smooth,'red','Non-Pen')
    plt.plot(Xi, Yi,'o',color='green',label= '$Training$')
    plt.plot(Xi_complement, Yi_complement,'o',color='blue',label= '$Validation$')
    plt.legend(loc="upper left"); plt.grid(); plt.show()

CV_Estimator=0
for i in range(0,10):
  index_vector=np.array([5*i,5*i+1,5*i+2,5*i+3,5*i+4])
  index_vector_complement=GetVectorComplement(X,index_vector)
  index_vector,index_vector_complement=SwapVectors(index_vector,index_vector_complement)
  PrintCrossValidationStats(X,Y,f,l,index_vector,index_vector_complement,i)
  CV=CrossValidation(X,Y,f,l,index_vector)
  CV_Estimator+=CV; print(''); print('');

print('-------------------------------------------------------')
print(f'El estimador de Validacion Cruzada Ridge es: {round(CV_Estimator/10.0,5)}')
print('-------------------------------------------------------')

#######################################
#Printing CV vs Penalty
#######################################

def GraphCrossValVSPenalty(X,Y,f,l_min,l_max,l_step,S):
    print(''); print('-------------------------------------------------------')
    print('--------------------GRAPHING l vs CV-------------------')
    print('-------------------------------------------------------')
    l=l_min; l_axis_len=int((l_max-l_min)/l_step)+1
    l_axis=np.zeros(l_axis_len)
    CV_axis=np.zeros(l_axis_len); j=0
    while j<l_axis_len:
      l_axis[j]=l; CV_Estimator=0
      for i in range(0,S):
          index_vector=np.array([0+i,S+i,2*S+i,3*S+i,4*S+i])
          index_vector_complement=GetVectorComplement(X,index_vector)
          index_vector,index_vector_complement=SwapVectors(index_vector,index_vector_complement)
          CV=CrossValidation(X,Y,f,l,index_vector)
          CV_Estimator+=CV
      CV_axis[j]=CV_Estimator/S; j=j+1; l=l+l_step
    plt.plot(l_axis, CV_axis,color='green',label= '$l$ vs $CV$')
    plt.legend(loc="upper left"); plt.grid(); plt.show()

GraphCrossValVSPenalty(X,Y,f,0,10,0.005,10)
