#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Packages

import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import matplotlib
import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import math
from pandas.io import gbq
from sklearn import linear_model
from sklearn.metrics import mean_squared_log_error
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
import datetime
from datetime import timedelta
from time import time
from itertools import groupby
get_ipython().run_line_magic('matplotlib', 'inline')
from __future__ import print_function
from statsmodels.compat import lzip
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
pd.options.mode.chained_assignment = None  # default='warn'

# Functions

def timer_func(func):
    # This function shows the execution time of 
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func

@timer_func
def top_n_SKUs(df, n):
    df=df[['SKU','Price']]
    df_g=df.groupby(['SKU']).sum()
    df_g=df_g.sort_values(by=['Price'], ascending=False )
    return list(df_g.index[:n])


def price_correction(df):
#     df_out=pd.DataFrame(columns=['COUNTRY','Channel','SKU','Transaction_date','Price', 'Qty_sold' ])
    
    df_out=pd.DataFrame()
    dfc=df
#     full_price=max(dfc['Price'])
#     dfc=dfc[dfc['Price']!=full_price]
    
    dates_list=[date for date in sorted(set(dfc['Transaction_date'])) ]
    
    for date in dates_list:
        price_vs_qty={}
        mini_df=pd.DataFrame()
        mini_df=dfc[['Transaction_date','Price','Qty_sold']][dfc['Transaction_date']==date]
        for price in list(set(mini_df['Price'])) :
            price_vs_qty[price]=int(mini_df['Qty_sold'][mini_df['Price']==price].iloc[0])
            keys = sorted([k for k, v in price_vs_qty.items() if v == max(price_vs_qty.values())])
            key=keys[0]
            
            one_line_df=pd.DataFrame()
            ol_dict={'COUNTRY':[dfc['COUNTRY'].iloc[0]], 'Channel':[dfc['Channel'].iloc[0]], 'SKU':[dfc['SKU'].iloc[0]] ,
                    'Transaction_date':[date], 'Price': [key], 'Qty_sold':[int(mini_df['Qty_sold'].sum())]}
            one_line_df=pd.DataFrame(ol_dict)
        df_out=df_out.append(one_line_df)
    return (df_out)

def price_column_addition(df):
    price_column=[]
    
    for i in list(df['Price']):
        price_column.append(i)
        
    df['Price']=price_column
    return df

@timer_func
def full_price_dict(df):
    sku_vs_max_p={}
    for sku in df['SKU'].unique():
        mini_df=df[df['SKU']==sku]
        full_price=max(mini_df['Price'])
        sku_vs_max_p[sku]=full_price
    return sku_vs_max_p
    


@timer_func
def current_price_dict(df):
    sku_vs_current_p={}
    for sku in df['SKU'].unique():
        mini_df=df[df['SKU']==sku]
        max_date=max(mini_df['Transaction_date'])
        current_price=float(max(mini_df['Price'][mini_df['Transaction_date']==max_date]))
        sku_vs_current_p[sku]=current_price
    return sku_vs_current_p
    

def indices_quantum_function(pc_df):
    dfc=pc_df
    l1=list(dfc['Price'])
    l2=list(range(len(l1)))
    l3=([list(j) for i, j in groupby(l1)])
    
    l4=[]
    for i in l3:
        l4.append(len(i))
    
    l5=[]
    for i in l4:
        l5.append(list(l2[0:i]))
        del l2[:i]
    l6=[x for x in l5 if len(x)>1]
    return l6

def MSRP_filter(df_with_full_price, df_msrp):
    msrp_dict = dict(zip(df_msrp.root_SKU, df_msrp.RETAIL_PRICE_USD_SAP))
    MSRP_list=[]
    for i in list(df_with_full_price['SKU']):
        try:
            MSRP_list.append(msrp_dict[i])
        except (KeyError):
            MSRP_list.append('MSRP unavailable')
            continue
    df_with_full_price['MSRP']=MSRP_list
    df_with_full_price['MSRP_updated'] = np.where((df_with_full_price['MSRP'] == 'MSRP unavailable')
                     , df_with_full_price['Full_price'], df_with_full_price['MSRP'])
    
    df_with_full_price['MSRP_STATUS'] = np.where((df_with_full_price['MSRP_updated'] >=  df_with_full_price['Price'])
                     , 'In_range', 'Out_of_range')
    
    df_with_full_price=df_with_full_price[df_with_full_price['MSRP_STATUS']=='In_range']
    
    df_with_full_price['MSRP']=df_with_full_price['MSRP_updated']
    
    df_with_full_price=df_with_full_price.drop(columns=['MSRP_updated', 'MSRP_STATUS'])
    
    return df_with_full_price
    

def selling_rate(df):
    
    pcm_df=df
    pcm_df.reset_index(drop=True, inplace=True)
    indices_quantum=indices_quantum_function(pcm_df)
    
    df1=pcm_df
    lol_for_dataframe_output=[]
    for quanta in indices_quantum:
        mini_output=pd.DataFrame()
        start=quanta[0]
        end=quanta[-1]
        df_small=df1.filter(items=quanta, axis=0)
        
        
        total_qty_sold=0
       
        td=max(df_small['Transaction_date'])-min(df_small['Transaction_date'])
        days_range=td.days
        
        for i in range(start, end+1):
            total_qty_sold+=df_small['Qty_sold'][i]
        
        Qty_sold_per_day=round((total_qty_sold)/(days_range),2)
       
        output_list=[]        
        output_list.append(list(df_small['SKU'])[0])
        output_list.append(list(df_small['Price'])[0])
        output_list.append(total_qty_sold)
        output_list.append(Qty_sold_per_day)
        
        lol_for_dataframe_output.append(output_list)
        
        output_dataframe=pd.DataFrame(data=lol_for_dataframe_output, columns=['SKU','Price','Total_qty_sold','Qty_sold_per_day'])
        
    
   
    output_indices=indices_quantum_function(output_dataframe)
    li1=list(output_dataframe['Price'])
    li2=([list(j) for i, j in groupby(li1)])
    
    li3=[x[0] for x in li2 if len(x)>1]
    
    for i in li3:
        df_mini=output_dataframe[output_dataframe['Price']==i]
        tqs=df_mini['Total_qty_sold'].sum()
        denom=0
        for item in range (len(df_mini)):
            denom+=(df_mini['Total_qty_sold'].iloc[item])/(df_mini['Qty_sold_per_day'].iloc[item])
        new_qty_per_day=tqs/denom
        output_dataframe['Total_qty_sold'][output_dataframe['Price']==i]=tqs
        output_dataframe['Qty_sold_per_day'][output_dataframe['Price']==i]=new_qty_per_day
        
    output=pd.DataFrame()
    
    output=output_dataframe.drop_duplicates()
    output=output.sort_values('Price')
    output.reset_index(drop=True, inplace=True)

    return output
    

def full_price_and_current_price_columns(pe_df, df):
    fpd=full_price_dict(df)
    fpl=[]
    for sku in list(pe_df['SKU']):
        fpl.append(fpd[sku])
    pe_df['Full_price']=fpl
    
    cpd=current_price_dict(df)
    cpl=[]
    for sku in list(pe_df['SKU']):
        cpl.append(cpd[sku])
    pe_df['Current_price']=cpl
    
    return pe_df

@timer_func
def pricing_elasticity_selling_rate(df, n):
    df_out=pd.DataFrame()
    top_SKUs=top_n_SKUs(df,n)
    for sku in top_SKUs:
        df1=df[df['SKU']==sku]
        try:
            f1=price_correction(df1)
        except (KeyError):
            print ('SKU {} is skipped due to insufficient data on price variations'.format(sku))
            continue
            
        try:
            f2=f1
        except (KeyError):
            print ('SKU {} is skipped due to insufficient data on price variations'.format(sku))
            continue
            
        try:
            f3=selling_rate(f2)
        except (TypeError,UnboundLocalError):
            print ('SKU {} is skipped due to insufficient data on price variations'.format(sku))
            continue
            
        f4=price_column_addition(f3)
        X = f4[['Total_qty_sold', 'Qty_sold_per_day']]
        y = f4['Price']

        regr = linear_model.LinearRegression(fit_intercept=True)
        regr.fit(X, y)
        
        Prediction = regr.predict(X)
        Regression_model_score = regr.score(X, y)
        
#         print ("Prediction: {}".format(Prediction))
#         print ("Regression_score: {}".format(Regression_model_score))
        
        
        one_line_df=pd.DataFrame()
        ol_dict={'COUNTRY':[df1['COUNTRY'].iloc[0]], 'Channel':[df1['Channel'].iloc[0]],
                 'SKU':sku , 'watch_cat':[df1['watch_cat'].iloc[0]],  
                 'Qty_coefficient': regr.coef_[0], 'Selling_rate_coefficient': regr.coef_[1], 'Intercept':regr.intercept_,
                 'Regression_score': Regression_model_score}
        one_line_df=pd.DataFrame(ol_dict)
        print ('SKU: {} Done'.format(sku))    
        df_out=df_out.append(one_line_df)
        df_out.set_index(np.arange(len(df_out)), inplace=True)

        
    return df_out

@timer_func
def Regression_comparison(sku_df):
    
    df1=sku_df
    f1=price_correction(df1)
    f2=f1
    f3=selling_rate(f2)
    f4=price_column_addition(f3)
    X = f4[['Total_qty_sold', 'Qty_sold_per_day']]
    y = f4['Price']

    regr = linear_model.LinearRegression(fit_intercept=True)
    regr.fit(X, y)

    Prediction = regr.predict(X)
#     Regression_model_score = regr.score(X, y)
    
    f4['Prediction']=Prediction
    f4=f4[['SKU', 'Price', 'Prediction']]
    
    r2=r2_score(f4['Price'],f4['Prediction'])
    
    print ('R squared value for this SKU is {}'.format(r2))
    
    return f4

