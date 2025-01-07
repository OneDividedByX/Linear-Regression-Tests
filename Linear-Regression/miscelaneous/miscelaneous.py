import numpy as np
import pandas as pd
import csv

####################################################
####################################################
# Vector
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

####################################################
####################################################
#DataFrame
def get_sorted_DataFrame(data,colum_names,target_colum_name):
    Data_sorted=data.sort_values(by=target_colum_name)
    Data_sorted=pd.DataFrame(Data_sorted)
    x=np.array(Data_sorted)[:,0]; y=np.array(Data_sorted)[:,1]
    return pd.DataFrame({colum_names[0]:x,colum_names[1]:y})

####################################################
####################################################
#MatrixOperator
def abs_matrix(M):
    n=len(M[0,:])
    m=len(M[:,0])
    M_abs=np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            M_abs[i,j]=np.abs(M[i,j])
    return M_abs

####################################################
####################################################
#MatrixMethod
def GaussJordan_Matrix_Inverse(matrix):
    if len(matrix[0])==len(matrix):
        order= len(matrix)
        unity= np.zeros([order,order])
        for l in range(order):
            unity[l,l]= 1.0
        for k in range(order):
            pivot= matrix[k,k]
            for j in range(order):
                if j==k:
                    continue
                else:
                    sec= matrix[j,k]
                    mult= sec/pivot
                    for i in range(order):
                        matrix[j,i] = matrix[j,i] - mult*matrix[k,i]
                        unity[j,i] =  unity[j,i] - mult*unity[k,i]
            matrix[k]= matrix[k]/pivot
            unity[k]= unity[k]/pivot
        return unity

def get_less_complex_Matrix(M):
    M_abs=abs_matrix(M)
    # M_less=M_abs.copy()
    # for i in range(len(M_less)):
    #     for j in range(len(M_less[0])):
    #         if M_less[i,j]<1e-10:
    #             M_less[i,j]=0
    # return M_less
    M_aux=(1/M_abs.max())*M
    return M_aux

def matrix_product_Trace_optimized(A,B): #Both square matrices
    A_abs=abs_matrix(A); B_abs=abs_matrix(B)
    max_A=A_abs.max(); max_B=B_abs.max()
    A_aux=(1/max_A)*A; B_aux=(1/max_B)*B
    trace=np.trace(A_aux@B_aux)
    return max_A*max_B*trace
    # return max_A*max_B*np.einsum('ij,ji->',A_aux, B_aux)