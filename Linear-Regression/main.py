import numpy as np
import pandas as pd
import csv
from scipy.stats import norm
import statistics
from miscelaneous.processing_data import get_sorted_DataFrame
from cross_validation.cross_validation import MeanGraphKFoldsRidgeMCO, GraphKFoldsRidgeMCO, MeanSummaryKFoldsRidgeMCO
from linear_regression.showing_training import Graph_MCO_Ridge_Training
from linear_regression.showing_training import Print_MCO_Ridge_Table

####################################################
####################################################
data=pd.read_csv('DATAXY.csv',sep=';')
Data_XY=get_sorted_DataFrame(data,['X','Y'],'X')

####################################################
X=np.array(Data_XY['X'])
Y=np.array(Data_XY['Y'])
####################################################
MaxX=max(X); MinX=min(X)
smooth=250 #Check GraphPrediction parameters
l=10 #Ridge penalty (l=0 is equivalent to OLS)
####################################################
def f(x): #d<=n
    # return np.array([x,1])
    # return np.array([x**18,x**16,x**15,x**14,x**12,x**10,x**9,x**8,x**7,x**6,x**5,x**4,x**3,x**2,x,1])
    # return np.array([x**18,x**17,x**16,x**15,x**14,x**13,x**12,x**11,x**10,x**9,x**8,x**7,x**6,x**5,x**4,x**3,x**2,x,1])
    # return np.array([1,x,x**2,x**3,x**4,x**5,x**6,x**7,x**8,x**9,x**10,x**11,x**12,x**13,x**14,x**15,x**16,x**17,x**18])
    # return np.array([x**5,x**4,x**3,x**2,x,1])
    return np.array([1,x,x**2,x**3,x**4,x**5])
####################################################

print(Data_XY)
Print_MCO_Ridge_Table(X,Y,f,l,3)

####################################################
K_fold=5
Interval_Lenght=MaxX-MinX; Delta=0.05*Interval_Lenght
Inf=MinX-Delta; Sup=MaxX+Delta
# Inf=MinX-0.1; Sup=MaxX+0.1
Graph_MCO_Ridge_Training(X,Y,f,l,Inf,Sup,smooth)
# MeanSummaryKFoldsRidgeMCO(X,Y,f,K_fold,l)
# GraphKFoldsRidgeMCO(X,Y,f,K_fold,l,Inf,Sup,smooth)
# MeanGraphKFoldsRidgeMCO(X,Y,f,K_fold,l,Inf,Sup,smooth)