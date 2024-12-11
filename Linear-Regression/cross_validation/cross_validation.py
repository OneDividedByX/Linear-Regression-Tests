import numpy as np
from linear_regression.linear_regression import *
from miscelaneous.tables import *
from linear_regression.error import *

####################################################
####################################################
def SimpleValidation(X,Y,f,l,E): #Para el costo norma 2
  E_c=GetVectorComplement(X,E)
  Xi=GetVectorFromIndexes(X,E) #Entrenamiento
  Yi=GetVectorFromIndexes(Y,E) #Entrenamiento
  Xi_complement=GetVectorFromIndexes(X,E_c) #Validacion
  Yi_complement=GetVectorFromIndexes(Y,E_c) #Validacion
  card=len(Xi)
  return 1/card*np.linalg.norm(Yi_complement-Phi(Xi_complement,f)@Estimator(Xi,Yi,f,l))**2

def PrintCrossValidationStats(X,Y,f,l,index_vector,index_vector_complement,i,Inf,Sup,smooth):
    # inicio = time.time()
    Xi=GetVectorFromIndexes(X,index_vector) #Entrenamiento
    Yi=GetVectorFromIndexes(Y,index_vector) #Entrenamiento
    Xi_complement=GetVectorFromIndexes(X,index_vector_complement) #Validacion
    Yi_complement=GetVectorFromIndexes(Y,index_vector_complement) #Validacion
    print('-------------------------------------------------------')
    print(f'-------------------PART {i+1} DATA-------------------')
    print('-------------------------------------------------------')
    print(f'El estimador Non-Pen (MSE{i+1}={round(MSE(Xi,Yi,f,0),5)}) es:')
    theta_i=Estimator(Xi,Yi,f,0)
    PrintVectorH(theta_i)
    print('-------------------------------------------------------')
    print(f'El estimador Ridge (l={l}) (MSE{i+1}={round(MSE(Xi,Yi,f,l),5)}) es:')
    theta_i_lambda=Estimator(Xi,Yi,f,l)
    PrintVectorH(theta_i_lambda)
    print('-------------------------------------------------------')
    print(f'El estimador de validación simple Ridge (VAL{i+1}) es: {round(SimpleValidation(X,Y,f,l,index_vector),5)}')
    print('-------------------------------------------------------')
    print(f'-----------------GRAPHING PART {i+1} DATA-----------------')
    print('-------------------------------------------------------')
    # fin = time.time()
    # print(f'Tiempo de ejecución: {fin-inicio}')
    GraphPrediction(Xi,Yi,f,0,Inf,Sup,smooth,'red','Non-Pen') #Grafica del no penalizado (MCO)
    GraphPrediction(Xi,Yi,f,l,Inf,Sup,smooth,'blue','Ridge') #Grafica de Ridge
    plt.plot(Xi, Yi,'o',color='green',label= 'Training')
    plt.plot(Xi_complement, Yi_complement,'o',color='magenta',label= 'Test')
    plt.legend(loc="upper left")
    # plt.gca().set_aspect('equal')
    plt.grid()
    plt.show()
    
#######################################
#Printing CV vs Penalty
#######################################

def GraphCrossValVSPenalty(X,Y,f,l_min,l_max,l_step,S):
    print('')
    print('-------------------------------------------------------')
    print('--------------------GRAPHING l vs CV-------------------')
    print('-------------------------------------------------------')
    l=l_min
    l_axis_len=int((l_max-l_min)/l_step)+1
    l_axis=np.zeros(l_axis_len)
    CV_axis=np.zeros(l_axis_len); j=0
    while j<l_axis_len:
      l_axis[j]=l
      CV_Estimator=0
      for i in range(0,S):
          index_vector=np.array([0+i,S+i,2*S+i,3*S+i,4*S+i])
          index_vector_complement=GetVectorComplement(X,index_vector)
          index_vector,index_vector_complement=SwapVectors(index_vector,index_vector_complement)
          CV=SimpleValidation(X,Y,f,l,index_vector)
          CV_Estimator+=CV
      CV_axis[j]=CV_Estimator/S; j=j+1
      l=l+l_step
    # plt.plot(l_axis, CV_axis,'o',color='green')
    # PrintVectorH(l_axis)
    # PrintVectorH(CV_axis)
    plt.plot(l_axis, CV_axis,color='green',label= '$l$ vs $CV$')
    plt.legend(loc="upper left")
    # plt.gca().set_aspect('equal')
    plt.grid()
    plt.show()
    
####################################################
####################################################   
    
def GraphKFoldsRidgeMCO(X,Y,f,K_fold,l,Inf,Sup,smooth):
  print('')
  print('-------------------------------------------------------')
  print('---------------GRAPHING K-FOLDS TRAINED----------------')
  print('-------------------------------------------------------')
  print('')
  CV_Estimator=0 #Validacion cruzada Ridge (promedio de las validaciones anteriores)
  r=int(len(X)/K_fold)
  for i in range(0,K_fold): #K_fold-fold
    # index_vector=np.array([0+i,10+i,20+i,30+i,40+i]) #Editable    
    index_vector=np.zeros(r)
    for j in range(0,r): #For the moment only works when (len X % K_fold==0)
      index_vector[j]=r*i+j
    index_vector_complement=GetVectorComplement(X,index_vector)
    index_vector,index_vector_complement=SwapVectors(index_vector,index_vector_complement)
    PrintCrossValidationStats(X,Y,f,l,index_vector,index_vector_complement,i,Inf,Sup,smooth) # Inf and sup are the limits on AxisX
    CV=SimpleValidation(X,Y,f,l,index_vector)
    CV_Estimator+=CV
    print(''); print('')
  CV_Estimator=CV_Estimator/float(K_fold)
  if l>0:
    print('-------------------------------------------------------')
    print(f'El estimador de Validacion Cruzada Ridge (l={l},K={K_fold}) es: {round(CV_Estimator,5)}')
    print('-------------------------------------------------------')
  else:
    print('-------------------------------------------------------')
    print(f'El estimador de Validacion Cruzada MCO (K={K_fold}) es: {round(CV_Estimator,5)}')
    print('-------------------------------------------------------')
    
def EstimatorsMeanKFoldsRidgeMCO(X,Y,f,K_fold,l):
  CV_Estimator=0 #Validacion cruzada Ridge (promedio de las validaciones anteriores)
  theta_MCO=0
  theta_Ridge=0
  for i in range(0,K_fold): #K_fold-fold
    # index_vector=np.array([0+i,10+i,20+i,30+i,40+i]) #Editable
    r=int(len(X)/K_fold)
    index_vector=np.zeros(r)
    for j in range(0,r): #For the moment only works when (len X % K_fold==0)
      index_vector[j]=r*i+j
    index_vector_complement=GetVectorComplement(X,index_vector)
    index_vector,index_vector_complement=SwapVectors(index_vector,index_vector_complement)
    Xi=GetVectorFromIndexes(X,index_vector) #Entrenamiento
    Yi=GetVectorFromIndexes(Y,index_vector) #Entrenamiento
    # Xi_complement=GetVectorFromIndexes(X,index_vector_complement) #Validacion
    # Yi_complement=GetVectorFromIndexes(Y,index_vector_complement) #Validacion
    theta_i=Estimator(Xi,Yi,f,0); theta_MCO=theta_MCO+theta_i;  theta_i_lambda=Estimator(Xi,Yi,f,l)
    theta_Ridge=theta_Ridge+theta_i_lambda
    CV=SimpleValidation(X,Y,f,l,index_vector)
    CV_Estimator+=CV
  CV_Estimator=CV_Estimator/float(K_fold)
  theta_MCO=theta_MCO/float(K_fold)
  theta_Ridge=theta_Ridge/float(K_fold)
  return theta_MCO,theta_Ridge,CV_Estimator

####################################################
####################################################  

def MeanSummaryKFoldsRidgeMCO(X,Y,f,K_fold,l):
    theta_MCO_mean,theta_Ridge_mean,CV_Ridge=EstimatorsMeanKFoldsRidgeMCO(X,Y,f,K_fold,l)
    print(f'Estimador MCO promedio:')
    PrintVectorH(theta_MCO_mean)
    print('')
    print(f'Estimador Ridge (l={l}) promedio:')
    PrintVectorH(theta_Ridge_mean)
    print('')
    print(f'Estimador de Validacion Cruzada Ridge (l={l},K={K_fold}): {CV_Ridge}''')
    
def MeanGraphKFoldsRidgeMCO(X,Y,f,K_fold,l,Inf,Sup,smooth):
    theta_MCO_mean,theta_Ridge_mean,CV_Ridge=EstimatorsMeanKFoldsRidgeMCO(X,Y,f,K_fold,l)
    GraphThetaPrediction(theta_MCO_mean,f,Inf,Sup,smooth,'red',f'MCO mean (K={K_fold})')
    GraphThetaPrediction(theta_Ridge_mean,f,Inf,Sup,smooth,'blue',f'Ridge mean (l={l},K={K_fold})')
    plt.plot(X, Y,'o',color='black',label= '$(x_i,y_i)$')
    plt.legend(loc="upper left")
    # plt.gca().set_aspect('equal')
    plt.grid()
    plt.show()