from miscelaneous.tables import *
from linear_regression.error import *
from linear_regression.linear_regression import *

####################################################
####################################################
def Print_MCO_Ridge_Table(X,Y,f,l,dec):
    print('-------------------------------------------------------')
    print('----------------------FULL DATA-----------------------')
    print('-------------------------------------------------------')
    print(f'Los {len(X)} datos y las predicciones son:')
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

def Graph_MCO_Ridge_Training(X,Y,f,l,Inf,Sup,smooth):
    print('-------------------------------------------------------')
    print('------------------GRAPHING FULL DATA-------------------')
    print('-------------------------------------------------------')
    GraphPrediction(X,Y,f,0,Inf,Sup,smooth,'red','Non-Pen')
    GraphPrediction(X,Y,f,l,Inf,Sup,smooth,'blue',f'Ridge (l={l})')
    plt.plot(X, Y,'o',color='black',label= '$(x_i,y_i)$')
    plt.legend(loc="upper left")
    # plt.gca().set_aspect('equal')
    plt.grid()
    plt.show()