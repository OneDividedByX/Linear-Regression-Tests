import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from miscelaneous.miscelaneous import GaussJordan_Matrix_Inverse, matrix_product_Trace_optimized, get_less_complex_Matrix
####################################################
####################################################
# Necessary functions

def PhiPolynomial(X,d): #d: degree of polynomial, Phi is [n x (d+1)] dimensional
  n=len(X)
  phi=np.zeros((n,d+1))
  for i in range(0,n):
    for j in range(0,d+1):
      phi[i][j]=X[i]**(d-j)
  return phi

def HatSigma(A,n):
    return 1/n*(np.transpose(A)@A)
  
def RidgePolynomialBias_Variance(X,sigma,d,theta_ast,l): #(In-Data,related to error distribution ,pol degree, theorical estimator, ridge penalty)
  n=len(X)
  A=PhiPolynomial(X,d)
  hat_sigma=HatSigma(A,n)
  H=hat_sigma@hat_sigma
  B=hat_sigma+l*np.identity(d+1)
  # B=GaussJordan_Matrix_Inverse(B)
  # C=B@B
  C=B@B
  # C=GaussJordan_Matrix_Inverse(C)
  C=np.linalg.inv(C)
  Bias=(l**2)*(theta_ast@C@hat_sigma@theta_ast)
  # print(f'Matrix inside of trace for variance (d={d})')
  # print(get_less_complex_Matrix(H@C))
  Variance=((sigma**2)/n)*matrix_product_Trace_optimized(H,C)
  # print(f'var={Variance}')
  # print('')
  return Bias, Variance #Check this function to correct calculation mistakes cause big numbers involved in matrices

def GraphRidgePolynomialBiasVarianceFromKTo_N(X,sigma,K,N,l): #(In-Data,related to error distribution,pol degree_min, pol degree_max, ridge penalty)
    AxisX=np.linspace(K,N,N-K+1); AxisYBias=np.zeros(N-K+1); AxisYVariance=np.zeros(N-K+1)
    for i in range(0,N-K+1):
        theta_ast=10*np.ones(i+K+1) #Subject to change before execution and acording to the case (except dimension)
        B,V=RidgePolynomialBias_Variance(X,sigma,i+K,theta_ast,l)
        AxisYBias[i]=B; AxisYVariance[i]=V
    plt.plot(AxisX, AxisYBias,color='green',label= 'Bias')
    plt.plot(AxisX, AxisYVariance,color='orange',label= f'Variance (l={l})')
    plt.legend(loc="upper left")
    # plt.gca().set_aspect('equal')
    plt.grid()
    plt.show()