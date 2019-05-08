import pandas as pd 
import ast
from ast import literal_eval
import numpy as np
from pandas import DataFrame
from IPython.display import display, clear_output
import math
import statistics

from datetime import datetime
from dateutil.parser import parse
import time
import datetime

from sklearn.metrics import mean_squared_error
from fancyimpute import KNN 
import matplotlib.pyplot as plt

from numpy import *

import winsound

col_Mse = ['id', 'budget', 'original_language', 'popularity', 'release_date',
       'runtime', 
       'production_companies0_mean', 'production_companies0_var',
       'production_countries0_mean', 'production_countries0_var',
       'spoken_languages0_mean', 'spoken_languages0_var', ]
           
col_Hit = ['id', 'genres0_mean', 'genres0_var', 'genres1_mean', 'genres1_var','Keywords0_mean',
       'Keywords0_var', 'Keywords1_mean', 'Keywords1_var', 'Keywords2_mean',
       'Keywords2_var', 'cast0_mean', 'cast0_var', 'cast1_mean', 'cast1_var',
       'cast2_mean', 'cast2_var', 'crew0_mean', 'crew0_var', 'crew1_mean',
       'crew1_var', 'crew2_mean', 'crew2_var']

col_Hit_sub = ['genres0', 'genres1','Keywords0',
        'Keywords1',  'Keywords2',
        'cast0', 'cast1', 
       'cast2',  'crew0', 'crew1',
        'crew2']

dataList = [ 'belongs_to_collection',  'genres', 'production_companies', 'production_countries',  'spoken_languages', 'Keywords', 'cast', 'crew']

dataList_expendsize = { 'belongs_to_collection':1,  'genres':3, 'production_companies':3, 'production_countries':3,  'spoken_languages':3, 'Keywords':3, 'cast':3, 'crew':3}

orgDF = pd.read_csv('save_ORG.csv')

for col in dataList :
    orgDF[[col]] = orgDF[[col]].applymap(literal_eval)    
    
PerformanceDF = pd.read_csv('Performance.csv')
PerformanceDF["mean"] = PerformanceDF["mean"].astype(float)
PerformanceDF["var"] = round(PerformanceDF["var"],0)


def worker(x):
    return x*x

def takeSecond(elem):
    return elem[1]

def Getting3MostRelated(df,col):           
    
    Performance = pd.read_csv('Performance.csv')

    #creat the 3 new columns
    length = len(df)
    data = {}
    for a in range(dataList_expendsize[col]):
        data.update({col+str(a)+"_mean":range(length), col+str(a)+"_var":range(length)})
 
    df_temp = pd.DataFrame(data) 

    # get the value for each 3 columns
    SearchList = Performance["id"].tolist()
    for x in range(len(df)):
        if (True): #lol
            sub_var = []
            if not isinstance(df[col][x], float): #if the df is not nan for col,x
                for id in df[col][x]:
                    
                    index = PerformanceDF.loc[(PerformanceDF['col'] == col) & (PerformanceDF['id'] == id) ].index
                    if not (len(index) == 0):
                        
                        sub_var.append([Performance["mean"][index[0]],Performance["var"][index[0]]]) 
                    # get the list of variance for each crew in this cell
                    
                sub_var = sorted(sub_var, key=takeSecond)
                
                min_item = []
                for a in range(dataList_expendsize[col]):
                    if(len(sub_var)>0):
                        min_item.append(sub_var[0])
                        sub_var.remove(sub_var[0])     
                    else:
                        min_item .append([np.nan,np.nan])

                for count in range(len(min_item)):
                    name = col +str(count)

                    df_temp[name+"_mean"][x] = min_item[count][0]
                    df_temp[name+"_var"][x] = min_item[count][1]

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

def hitRate(predDF,df_complete_sampled, std_scale_fullset,col_Hit):
    
    winsound.Beep(1800, 1000)
    cols = predDF.columns 
    predDF = pd.DataFrame(std_scale_fullset.inverse_transform(predDF), columns = cols)

    predDF = predDF[col_Hit]
    predDF.rename(columns={"id": "id0"}, inplace=True)
    predDF = round(predDF,0)
    hit = 0
    count = 0
    colist = col_Hit_sub
    for col in colist:
        for x in range(len(predDF)):
            if np.isnan(df_complete_sampled[col+"_mean"][x]):
                TargetID = PerformanceDF.loc[(PerformanceDF['mean'] < predDF[col+"_mean"][x]+16209729.94) & 
                                             (PerformanceDF['mean'] > predDF[col+"_mean"][x]-16209729.94) &
                                             (PerformanceDF['var'] < predDF[col+"_var"][x]+6707054291177210)&
                                             (PerformanceDF['var'] > predDF[col+"_var"][x]-6707054291177210)&
                                             (PerformanceDF['col']==col[:-1])]["id"]
                flag = 0
                for id in TargetID:
                    cell = orgDF["id"].tolist().index(predDF["id0"][x])
                    if id in orgDF[col[:-1]][cell]:
                        flag = 1
                hit = hit + flag
                count +=1
    return hit / count


def PredictingMethodVerfication(df_complete, missing_rate_list,std_scale_fullset,tup):
    winsound.Beep(1800, 1000)
    i = tup[0]
    df_complete_sampled = tup[1]
    tar_miss_rate = missing_rate_list[i]

    rmsDict = pd.DataFrame(columns=['KNN', 'IterativeImputer', 'SoftImpute','mean','median','most_frequent','constant'],index=[tar_miss_rate])
    rmsAllDict = pd.DataFrame(columns=['KNN', 'IterativeImputer', 'SoftImpute','mean','median','most_frequent','constant'],index=[tar_miss_rate])
    hitDict = pd.DataFrame(columns=['KNN', 'IterativeImputer', 'SoftImpute','mean','median','most_frequent','constant'],index=[tar_miss_rate])
    
    rms_list = []
    rmsAll_list = []
    hit_list = []

    for k in range(5):
        df_predicted = pd.DataFrame(KNN(k,verbose=False).fit_transform(df_complete_sampled))
        
        df_predicted.columns  = df_complete.columns 

        rms = np.sqrt(mean_squared_error(df_predicted[col_Mse], df_complete[col_Mse]))
        rmsAll = np.sqrt(mean_squared_error(df_predicted, df_complete))
        hit = hitRate(df_predicted, df_complete_sampled, std_scale_fullset,col_Hit)
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
    hit = hitRate(df_predicted,df_complete_sampled, std_scale_fullset,col_Hit)

    rmsDict["IterativeImputer"][tar_miss_rate] = round(rms, 5) 
    rmsAllDict["IterativeImputer"][tar_miss_rate] = round(rmsAll, 5)
    hitDict["IterativeImputer"][tar_miss_rate] = round(hit, 5) 

    # Prediction with SoftImpute 

    from fancyimpute import SoftImpute 

    df_predicted = pd.DataFrame(SoftImpute(verbose=False).fit_transform(df_complete_sampled))
    df_predicted.columns  = df_complete.columns 
    rms = np.sqrt(mean_squared_error(df_predicted[col_Mse], df_complete[col_Mse]))
    rmsAll = np.sqrt(mean_squared_error(df_predicted, df_complete))
    hit = hitRate(df_predicted,df_complete_sampled, std_scale_fullset,col_Hit)

    rmsDict["SoftImpute"][tar_miss_rate] = round(rms, 5) 
    rmsAllDict["SoftImpute"][tar_miss_rate] = round(rmsAll, 5) 
    hitDict["SoftImpute"][tar_miss_rate] = round(hit, 5) 

    #Prediction with 'mean','median','most_frequent'

    # https://scikit-learn.org/stable/modules/impute.html#impute
    from sklearn.impute import SimpleImputer

    for stra in ['mean','median','most_frequent','constant']:

        imp = SimpleImputer(missing_values=np.nan, strategy=stra, fill_value = 0)
        imp.fit(df_complete_sampled)
        df_predicted = imp.transform(df_complete_sampled)
        df_predicted = pd.DataFrame(df_predicted)
        df_predicted.columns  = df_complete.columns
        rms = np.sqrt(mean_squared_error(df_predicted[col_Mse], df_complete[col_Mse]))
        rmsAll = np.sqrt(mean_squared_error(df_predicted, df_complete))
        hit = hitRate(df_predicted, df_complete_sampled, std_scale_fullset,col_Hit)

        rmsDict[stra][tar_miss_rate] = round(rms, 5) 
        rmsAllDict[stra][tar_miss_rate] = round(rmsAll, 5) 
        hitDict[stra][tar_miss_rate] = round(hit, 5) 
        
        
    return rmsDict, rmsAllDict,hitDict