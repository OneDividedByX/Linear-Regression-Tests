from linear_regression.showing_training import Graph_MCO_Ridge_Training
from validation.cross_validation import MeanGraphKFoldsRidgeMCO

def Training(X,Y,f,l,InfX,SupX,smooth):
    return Graph_MCO_Ridge_Training(X,Y,f,l,InfX,SupX,smooth)

def KFolds_mean(X,Y,f,K_fold,l,Inf,Sup,smooth):
    return MeanGraphKFoldsRidgeMCO(X,Y,f,K_fold,l,Inf,Sup,smooth)