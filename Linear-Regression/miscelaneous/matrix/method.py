from miscelaneous.miscelaneous import abs_matrix
from miscelaneous.miscelaneous import GaussJordan_Matrix_Inverse
from miscelaneous.miscelaneous import matrix_product_Trace_optimized


def GJ_Inverse(matrix):
    return GaussJordan_Matrix_Inverse(matrix)

def reduce_complexity(M):
    M_abs=abs_matrix(M) #Takes absolute value in the matrix
    m=M_abs.max()
    M_aux=(1/m)*M
    return M_aux, m

def optimized_trace(A,B): #Both square matrices
    return matrix_product_Trace_optimized(A,B)
    # return max_A*max_B*np.einsum('ij,ji->',A_aux, B_aux)