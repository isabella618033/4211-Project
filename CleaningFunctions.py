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
        if (True): #check if we have already ran throught this column before
            sub_var = {}
            print(x, col, df[col][x], type(df[col][x]),isinstance(df[col][x], float))
            if not isinstance(df[col][x], float): #if the df is not nan for col,x
                print("len(df[col][x])",len(df[col][x]))
                for id in df[col][x]:
                    sub_var.update({id : var[id]}) # get the list of variance for each crew in this cell
                #print(sub_var)
                min_item = []
                for a in range(3):
                    if(len(sub_var)>0):
                        min_item.append(min(sub_var.keys(), key=(lambda k: sub_var[k])))
                        sub_var.pop(min_item[a])     
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
            df.release_date[count] = (df.release_date[count].year-1955)*365.25 + (df.release_date[count].month-1)*(365.25/12)+(df.release_date[count].day-1)
    return df.release_date[count]