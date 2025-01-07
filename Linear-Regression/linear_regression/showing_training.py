from miscelaneous.tables import *
from linear_regression.error import *
from linear_regression.linear_regression import *

####################################################
####################################################
# def Print_MCO_Ridge_Table(X: np.ndarray,Y: np.ndarray,f: function,l: float, dec: int) -> str:
def Print_MCO_Ridge_Table(X,Y,f,l, dec):
    """
    Prints the table containing the data and their respective predictions and error predictions for Ridge regression. Additionally, it shows the MCO and Ridge estimators.
    
    Parameters
    ----------
        X: array
            Input data (1-dimensional)
        Y:  array
            Output data (1-dimensional)
        f:  funtion
            Predictor function
        l:  float
            Ridge hyperparameter
        dec:    int
            Number of decimal places to round the results
    """
    print('-------------------------------------------------------')
    print('----------------------FULL DATA-----------------------')
    print('-------------------------------------------------------')
    print(f'Los {len(X)} datos y las predicciones Ridge (l={l}) son:')
    print(PrintTable_v2(X,Y,f,l,dec))
    print('-------------------------------------------------------')
    print(f'El estimador Non-Pen (MSE={round(MSE(X,Y,f,0),dec)}) es:')
    v=Estimator(X,Y,f,0)
    for i in range(len(v)):
        v[i]=float(round(v[i],dec))
    # PrintVectorH(v)
    v=pd.DataFrame({'estimator':v})
    print(v.T)
    print('-------------------------------------------------------')
    print(f'El estimador Ridge (l={l}) (MSE={round(MSE(X,Y,f,l),dec)}) es:')
    v=Estimator(X,Y,f,l)
    for i in range(len(v)):
        v[i]=float(round(v[i],dec))
    # PrintVectorH(v)
    v=pd.DataFrame({'estimator':v})
    print(v.T)
    print('-------------------------------------------------------')

def Graph_MCO_Ridge_Training(X,Y,f,l,InfX,SupX,smooth):
    print('-------------------------------------------------------')
    print('------------------GRAPHING FULL DATA-------------------')
    print('-------------------------------------------------------')
    GraphPrediction(X,Y,f,0,InfX,SupX,smooth,'red','Non-Pen')
    GraphPrediction(X,Y,f,l,InfX,SupX,smooth,'blue',f'Ridge (l={l})')
    plt.plot(X, Y,'o',color='black',label= '$(x_i,y_i)$')
    plt.legend(loc="upper left")
    # plt.gca().set_aspect('equal')
    plt.grid()
    plt.show()