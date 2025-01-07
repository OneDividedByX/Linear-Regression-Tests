import numpy as np
import pandas as pd
import csv
from scipy.stats import norm
import statistics

from validation.cross_validation import MeanGraphKFoldsRidgeMCO, GraphKFoldsRidgeMCO, MeanSummaryKFoldsRidgeMCO,GraphCrossValVSPenalty

import miscelaneous.process_data as process
import miscelaneous.vector as vector
import linear_regression.graph as lr_graph
import linear_regression.summary as lr_summary

####################################################
####################################################
data=pd.read_csv('DATAXY.csv',sep=';')
Data_XY=process.DataFrame.sort(data,['X','Y'],'X')

####################################################
X=np.array(Data_XY['X'])
Y=np.array(Data_XY['Y'])
length=len(X)
####################################################
MaxX=max(X); MinX=min(X)
smooth=250 #Check GraphPrediction parameters
l=10 #Ridge penalty (l=0 is equivalent to OLS)
####################################################
def f(x): #d<=n
    # return np.array([x,1])
    # return np.array([x**18,x**16,x**15,x**14,x**12,x**10,x**9,x**8,x**7,x**6,x**5,x**4,x**3,x**2,x,1])
    # return np.array([x**30,x**29,x**28,x**27,x**26,x**25,x**24,x**23,x**22,x**21,x**20,x**19,x**18,x**17,x**16,x**15,x**14,x**13,x**12,x**11,x**10,x**9,x**8,x**7,x**6,x**5,x**4,x**3,x**2,x,1])
    # return np.array([1,x,x**2,x**3,x**4,x**5,x**6,x**7,x**8,x**9,x**10,x**11,x**12,x**13,x**14,x**15,x**16,x**17,x**18])
    # return np.array([x**18,x**17,x**16,x**15,x**14,x**13,x**12,x**11,x**10,x**9,x**8,x**7,x**6,x**5,x**4,x**3,x**2,x,1])
    #return np.array([x**5,x**4,x**3,x**2,x,1])
    # return np.array([1,x,x**2,x**3,x**4,x**5])
    d=5
    varphi=np.zeros(d)
    for i in range(0,d):
        varphi[i]=x**i
    return varphi
####################################################
dec=3
# print(Data_XY)
# Print_MCO_Ridge_Table(X,Y,f,l,dec)
lr_summary.MCO_Ridge.Table(X,Y,f,l,dec)

####################################################
K_fold=5
Interval_Lenght=MaxX-MinX; Delta=0.05*Interval_Lenght
Inf=MinX-Delta; Sup=MaxX+Delta
# Inf=MinX-0.1; Sup=MaxX+0.1
# Graph_MCO_Ridge_Training(X,Y,f,l,Inf,Sup,smooth+250)
lr_graph.MCO_Ridge.Training(X,Y,f,l,Inf,Sup,smooth+250)

# GraphKFoldsRidgeMCO(X,Y,f,K_fold,l,Inf,Sup,smooth,dec)
# MeanSummaryKFoldsRidgeMCO(X,Y,f,K_fold,l,dec)
# MeanGraphKFoldsRidgeMCO(X,Y,f,K_fold,l,Inf,Sup,smooth)
####################################################

left_limit_graph=0; right_limit_graph=20; l_cv_delta=0.05
# GraphCrossValVSPenalty(X,Y,f,left_limit_graph,right_limit_graph,l_cv_delta,K_fold)
# GraphRidgePolynomialBiasVarianceFromKTo_N(X,sigma,1,15,0.055)

####################################################
deg=1; sigma=1; theta_ast=10*np.ones(deg+1); K=1; N=30; l=0.1
# GraphRidgePolynomialBiasVarianceFromKTo_N(X,sigma,K,N,l)
lr_graph.bias_vs_variance.Ridge(X,sigma,1,50,10)