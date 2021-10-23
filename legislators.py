
import math
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

    
def year(file,year):
    
    N_year = year-1917
    
    N_file = file
    with open(file) as f:
        data = json.load(f)
      
    df = pd.json_normalize(data, record_path =['terms'],meta = [["id","bioguide"]])
    df2 = pd.merge(df, pd.json_normalize(data), on='id.bioguide',how='left')
    df2['start']= pd.to_datetime(df2['start'])
    df2['end']= pd.to_datetime(df2['end'])
    df2['start_year'] = df2['start'].dt.year
    df2['end_year'] = df2['end'].dt.year
    df3 = df2[df2['bio.gender'] == 'F'][['id.bioguide','start_year','end_year']]
    df3['Diff'] = df3.end_year - df3.start_year
    df5 = pd.DataFrame(columns = ['id.bioguide', 'year'])
    for x in range(0,len(df3)):
        for y in range(0,df3.iloc[x,3]+1):
            ID = df3.iloc[x,0]
            Year = df3.iloc[x,1] + y
            New_data = pd.DataFrame({"id.bioguide":[ID],
                    "year":[ Year]
                                })
            df5 = df5.append(New_data, ignore_index = True)
    N_W = df5.groupby('year').size().to_frame('N_count').reset_index()
    df = N_W.set_index(['year'])
    df_log = np.log(df)
    
    #using auto_arima(df_log['N_count'],trace = True,Suppress_warning = True) to find the best model
    #Best model:  ARIMA(2,1,5)(0,0,0)[0] intercept
    
    model = ARIMA(df_log, order=(2,1,5))
    results = model.fit()
    pred2 = results.predict(start=0,end=N_year,type='levels')
    print(math.floor(math.exp(pred2.iloc[N_year])))
                  
    
    
    
if __name__ == "__main__":
    year(str(sys.argv[1]), int(sys.argv[2]))



