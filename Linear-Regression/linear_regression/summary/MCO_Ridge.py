from validation.cross_validation import MeanSummaryKFoldsRidgeMCO
from linear_regression.showing_training import Print_MCO_Ridge_Table
def KFolds_mean(X,Y,f,K_fold,l,dec):
    return MeanSummaryKFoldsRidgeMCO(X,Y,f,K_fold,l,dec)

def Table(X,Y,f,l, dec):
    return Print_MCO_Ridge_Table(X,Y,f,l, dec)