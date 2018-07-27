# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 16:58:46 2018

@author: Subhajeet
"""

def normalize_data(df):
    import numpy as np
    df=df.dropna(axis=0)
    cname=list(df)
    for i in cname:
        if df[i].dtype==float:
            for j in range(len(df[i])):
                if df[i][j]<=np.sum(df[i])/4:
                    df[i][j]=0
                elif df[i][j]>3*(np.sum(df[i]))/4:
                    df[i][j]=3
                elif df[i][j]>(np.sum(df[i])/4) and df[i][j]<=np.sum(df[i])/2:
                    df[i][j]=1
                elif df[i][j]<=3*(np.sum(df[i])/4) and df[i][j]>np.sum(df[i])/2:
                    df[i][j]=2 
    for i in cname:
        if df[i].dtype==float:
            df[i][df[i]==0.0]='a'
            df[i][df[i]==1.0]='b'
            df[i][df[i]==2.0]='c'
            df[i][df[i]==3.0]='d'
    return df
#%%
def list_of_elements(df):
    import pandas as pd
    cname=list(df)
    l=[]
    for i in cname:
        l.append(pd.unique(df[i]))
    return l
#%%
def split(df,split_ratio=0.8):
    import numpy as np
    split=np.random.rand(len(df))<split_ratio
    train=df[split]
    test=df[~split]
    return train,test
#%%
def set_of_class(df,train,dc):
    import pandas as pd
    dv=pd.unique(df[dc])
    seti=[]
    for i in dv:
        seti.append(train.loc[train[dc]==i])
    return seti
#%%
def count_with_class_dictionary(seti,df,dc,l):
    import pandas as pd
    cname=list(df)
    d1=[{} for  _ in range(len(pd.unique(df[dc])))]
    for i in range(len(d1)):
        y=d1[i]
        data=seti[i]
        for k in range(len(cname)-1):
            for j in l[k]:
                y[j]=len(data.loc[data[cname[k]]==j])
    return d1
#%%
def count_class_elements(seti):
    t=[]
    for i in seti:
        t.append(len(i))
    return t
#%%
def prediction(test,train,dc,seti,d1,l,t):
    import numpy as np
    import pandas as pd
    testres=test.drop(dc,axis=1)
    resy=[[] for _ in range(len(seti))]
    m=2
    for i in range(len(seti)):
        for j in testres.values:
            r=[]
            for k in j:
                nc=d1[i][k]
                y=len(seti[i])
                if k not in d1[i]:
                    nc=0
                for q in l:
                    for s in range(len(q)):
                        if k in q[s]:
                            z=len(q[s])
                p=(nc+(m*(1/z)))/(m+y)
                r.append(p)
            resy[i].append(np.product(r))
    res=[[] for _ in range(len(seti))]
    for i in range(len(resy)):
        for j in resy[i]:
            res[i].append(j*t[i]/np.sum(t))
    cv=pd.unique(train[dc])
    pred=[]
    dk=pd.DataFrame(res).T
    p1=[list(item) for item in dk.values]
    for i in range(len(p1)):
        pred.append(max(p1[i]))
    pred_value=[]
    for i in range(len(pred)):
        for j in range(len(res)):
            if pred[i] in res[j]:
                pred_value.append(cv[j])
    return pred_value
#%%
def error(pred_value,test,dc):
    import numpy as np
    existr=test[dc]
    return pred_value==np.array(existr)
#%%
def naive_bayes(data,split_ratio,dc):
    df=normalize_data(data)
    l=list_of_elements(df)
    train,test=split(df,split_ratio)
    seti=set_of_class(df,train,dc)
    d1=count_with_class_dictionary(seti,df,dc,l)
    t=count_class_elements(seti)
    pred_value=prediction(test,train,dc,seti,d1,l,t)
    pred_error=error(pred_value,test,dc)
    print('Predictions are :','\n',pred_value)
    print('Errors list is :','\n',pred_error)
#%%
#checking with different datasets
import pandas as pd
df=pd.read_csv('tennis.csv')
df1=pd.read_csv('iris.csv',header=None)
dc='play'
dc1=4
naive_bayes(df,split_ratio=0.8,dc=dc)
naive_bayes(df1,split_ratio=0.85,dc=dc1)