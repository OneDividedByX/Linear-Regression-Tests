import numpy as np
import pandas as pd
import csv

####################################################
####################################################
def get_sorted_DataFrame(data,colum_names,target_colum_name):
    Data_sorted=data.sort_values(by=target_colum_name)
    Data_sorted=pd.DataFrame(Data_sorted)
    x=np.array(Data_sorted)[:,0]; y=np.array(Data_sorted)[:,1]
    return pd.DataFrame({colum_names[0]:x,colum_names[1]:y})