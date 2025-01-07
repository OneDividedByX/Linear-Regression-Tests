from validation.cross_validation import PrintCrossValidationStats
def stats(X,Y,f,l,index_vector,index_vector_complement,i,Inf,Sup,smooth,dec):
    return PrintCrossValidationStats(X,Y,f,l,index_vector,index_vector_complement,i,Inf,Sup,smooth,dec)