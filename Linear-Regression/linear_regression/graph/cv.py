from validation.cross_validation import GraphCrossValVSPenalty
from validation.cross_validation import GraphKFoldsRidgeMCO

def Penalty_CV_map(X,Y,f,l_min,l_max,l_step,S):
    return GraphCrossValVSPenalty(X,Y,f,l_min,l_max,l_step,S)
     
def MCO_Ridge_KFolds(X,Y,f,K_fold,l,Inf,Sup,smooth,dec):
    return GraphKFoldsRidgeMCO(X,Y,f,K_fold,l,Inf,Sup,smooth,dec)