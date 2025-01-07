from linear_regression.bias_variance import GraphRidgePolynomialBiasVarianceFromKTo_N

def MCO(X,sigma,min_deg,max_deg):
    return GraphRidgePolynomialBiasVarianceFromKTo_N(X,sigma,min_deg,max_deg,0)

def Ridge(X,sigma,min_deg,max_deg,l):
    return GraphRidgePolynomialBiasVarianceFromKTo_N(X,sigma,min_deg,max_deg,l)