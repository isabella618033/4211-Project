import pandas as pd 
import ast
from ast import literal_eval
import numpy as np
from pandas import DataFrame
from IPython.display import display, clear_output
import math

from datetime import datetime
from dateutil.parser import parse
import time
import datetime

from sklearn.metrics import mean_squared_error
from fancyimpute import KNN 
import matplotlib.pyplot as plt

import winsound

col_Mse = ['id', 'budget', 'original_language', 'popularity', 'release_date',
       'runtime', 'production_companies0',
        'production_countries0', 'spoken_languages0']
           
col_Hit = ['id', 'genres0', 'genres1','Keywords0', 'Keywords1', 'Keywords2', 'cast0', 'cast1', 'cast2',
       'crew0', 'crew1', 'crew2']

dataList = [ 'belongs_to_collection',  'genres', 'production_companies', 'production_countries',  'spoken_languages', 'Keywords', 'cast', 'crew']

orgDF = pd.read_csv('save_ORG.csv')

for col in dataList :
    orgDF[[col]] = orgDF[[col]].applymap(literal_eval)     


def worker(x):
    return x*x

def Getting3MostRelated(df,col):
    df_temp = pd.DataFrame()
    print(col)
    #collect all revenue for each crew
    Rev_List = {}
    for x in range(len(df)):
        if not (isinstance(df[col][x], float )):
            for y in range( len(df[col][x])):
                if (df[col][x][y] not in Rev_List):
                    list = []
                    Rev_List.update( {df[col][x][y]: list} )
                Rev_List[df[col][x][y]].append(df["revenue"][x])
                
    #calculate the variance of revenue for each crew
    var = {}
    for id in Rev_List.keys():
        if (len(Rev_List[id]) > 2):
            var.update({id : np.var(Rev_List[id])})
        else : 
            var.update({id : np.nan})

    #creat the 3 new columns
    length = len(df)
    data = {col+str(0):range(length), col+str(1):range(length),col+str(2):range(length)} 
    df_temp = pd.DataFrame(data) 

    # get the value for each 3 columns
    for x in range(len(df)):
        if (True): #lol
            sub_var = []
            #print(x, col, df[col][x], type(df[col][x]),isinstance(df[col][x], float))
            if not isinstance(df[col][x], float): #if the df is not nan for col,x
                #print("len(df[col][x])",len(df[col][x]))
                for id in df[col][x]:
                    sub_var.append(var[id]) # get the list of variance for each crew in this cell
                #print(sub_var)
                min_item = []
                for a in range(3):
                    if(len(sub_var)>0):
                        min_item.append(min(sub_var))
                        sub_var.remove(min_item[a])     
                    else:
                        min_item .append(np.nan)
                
                min_item.sort()
                
                for a in range(3):
                    name = col +str(a)
                    df_temp[name][x] = float(min_item[a])

    return df_temp, col

def job(num):
    data = [['tom', num+1], ['nick', num+2], ['juli', num+3]] 
    df = pd.DataFrame(data, columns = ['Name', 'Age']) 

    return df

def time2num(df, count):
    x = datetime.datetime(1970, 1, 1)
    helo_time=time.time()
    date = df.release_date[count]
    if (isinstance(date, str))and(date[-3] == "/"):
        if(int(date[-2], 10) < 2):
            df.release_date[count] = date[:-2] + '20' + date[-2:]
        else:
            df.release_date[count] = date[:-2] + '19' + date[-2:]
        
    if (isinstance(date, str)):
        df.release_date[count] = datetime.datetime.strptime(df.release_date[count], '%m/%d/%Y')

    if  isinstance(df.release_date[count], datetime.datetime):
        #clear_output(wait=True)
        #display(str(count)+ "           "+str(df.release_date[count]))
        if(df.release_date[count].year >= 1970):
            df.release_date[count] =(df.release_date[count]-x).days + 365.25*15
        else:
            df.release_date[count] = (df.release_date[count].year-1900)*365.25 + (df.release_date[count].month-1)*(365.25/12)+(df.release_date[count].day-1)
    return df.release_date[count]

def hitRate(predDF, std_scale_fullset,col_Hit):
    winsound.Beep(1800, 1000)
    cols = predDF.columns 
    predDF = pd.DataFrame(std_scale_fullset.inverse_transform(predDF), columns = cols) [col_Hit]
    predDF.rename(columns={"id": "id0"}, inplace=True)
    predDF = predDF.astype(int)
    hit = 0
    count = 0
    colist = col_Hit[1:]
    for col in colist:
        for x in range(len(predDF)):
            if predDF[col][x] in orgDF[col[:-1]][orgDF["id"].tolist().index(predDF["id0"][x])]:
                hit +=1
            count +=1
    return hit / count

def PredictingMethodVerfication(df_complete, missing_rate_list,std_scale_fullset,tup):
    winsound.Beep(1800, 1000)
    i = tup[0]
    df_complete_sampled = tup[1]
    tar_miss_rate = missing_rate_list[i]

    rmsDict = pd.DataFrame(columns=['KNN', 'IterativeImputer', 'SoftImpute','mean','median','most_frequent'],index=[tar_miss_rate])
    rmsAllDict = pd.DataFrame(columns=['KNN', 'IterativeImputer', 'SoftImpute','mean','median','most_frequent'],index=[tar_miss_rate])
    hitDict = pd.DataFrame(columns=['KNN', 'IterativeImputer', 'SoftImpute','mean','median','most_frequent'],index=[tar_miss_rate])
    
    rms_list = []
    rmsAll_list = []
    hit_list = []

    for k in range(10):
        df_predicted = pd.DataFrame(KNN(k,verbose=False).fit_transform(df_complete_sampled))
        
        df_predicted.columns  = df_complete.columns 

        rms = np.sqrt(mean_squared_error(df_predicted[col_Mse], df_complete[col_Mse]))
        rmsAll = np.sqrt(mean_squared_error(df_predicted, df_complete))
        hit = hitRate(df_predicted, std_scale_fullset,col_Hit)
        rms_list.append(rms)
        rmsAll_list.append(rmsAll)
        hit_list.append(hit)

    rmsDict["KNN"][tar_miss_rate] = round(min(rms_list), 5) 
    rmsAllDict["KNN"][tar_miss_rate] = round(min(rmsAll_list), 5) 
    hitDict["KNN"][tar_miss_rate] = round(max(hit_list), 5) 

    # Prediction with IterativeImputer (MICE)

    from fancyimpute import IterativeImputer 

    df_predicted = pd.DataFrame(IterativeImputer(verbose=False).fit_transform(df_complete_sampled))
    df_predicted.columns  = df_complete.columns 
    rms = np.sqrt(mean_squared_error(df_predicted[col_Mse], df_complete[col_Mse]))
    rmsAll = np.sqrt(mean_squared_error(df_predicted, df_complete))
    hit = hitRate(df_predicted, std_scale_fullset,col_Hit)

    rmsDict["IterativeImputer"][tar_miss_rate] = round(rms, 5) 
    rmsAllDict["IterativeImputer"][tar_miss_rate] = round(rmsAll, 5)
    hitDict["IterativeImputer"][tar_miss_rate] = round(hit, 5) 

    # Prediction with SoftImpute 

    from fancyimpute import SoftImpute 

    df_predicted = pd.DataFrame(SoftImpute(verbose=False).fit_transform(df_complete_sampled))
    df_predicted.columns  = df_complete.columns 
    rms = np.sqrt(mean_squared_error(df_predicted[col_Mse], df_complete[col_Mse]))
    rmsAll = np.sqrt(mean_squared_error(df_predicted, df_complete))
    hit = hitRate(df_predicted, std_scale_fullset,col_Hit)

    rmsDict["SoftImpute"][tar_miss_rate] = round(rms, 5) 
    rmsAllDict["SoftImpute"][tar_miss_rate] = round(rmsAll, 5) 
    hitDict["SoftImpute"][tar_miss_rate] = round(hit, 5) 

    #Prediction with 'mean','median','most_frequent'

    # https://scikit-learn.org/stable/modules/impute.html#impute
    from sklearn.impute import SimpleImputer

    for stra in ['mean','median','most_frequent']:

        imp = SimpleImputer(missing_values=np.nan, strategy=stra)
        imp.fit(df_complete_sampled)
        df_predicted = imp.transform(df_complete_sampled)
        df_predicted = pd.DataFrame(df_predicted)
        df_predicted.columns  = df_complete.columns
        rms = np.sqrt(mean_squared_error(df_predicted[col_Mse], df_complete[col_Mse]))
        rmsAll = np.sqrt(mean_squared_error(df_predicted, df_complete))
        hit = hitRate(df_predicted, std_scale_fullset,col_Hit)

        rmsDict[stra][tar_miss_rate] = round(rms, 5) 
        rmsAllDict[stra][tar_miss_rate] = round(rmsAll, 5) 
        hitDict[stra][tar_miss_rate] = round(hit, 5) 
        
    return rmsDict, rmsAllDict,hitDict