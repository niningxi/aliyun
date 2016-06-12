import pandas as pd
import numpy as np
import collections
from sklearn.ensemble import RandomForestRegressor
import datetime
###read in csv data including feature and Y
dianbo=pd.read_csv(r'E:\data\aliyun_new\dianbo.csv')
shoucang=pd.read_csv(r'E:\data\aliyun_new\shoucang.csv')
download=pd.read_csv(r'E:\data\aliyun_new\download.csv')
seven_mean=pd.DataFrame()
for name in dianbo.columns:
    seven_mean[name]=pd.rolling_mean(dianbo[name], window=7)


DATA_2=pd.read_csv(r'E:\data\aliyun_new\data_2_1.csv'  )
d1=datetime.datetime(2015, 3, 1)
festival=[datetime.datetime(2015,4,5),datetime.datetime(2015,4,6),datetime.datetime(2015,5,1),
          datetime.datetime(2015,5,2),datetime.datetime(2015,5,3),datetime.datetime(2015,6,20),
          datetime.datetime(2015,6,21),datetime.datetime(2015,6,22),datetime.datetime(2015,9,26),
          datetime.datetime(2015,9,27),datetime.datetime(2015,9,28),datetime.datetime(2015,10,1),
          datetime.datetime(2015,10,2),datetime.datetime(2015,10,3),datetime.datetime(2015,10,4),
          datetime.datetime(2015,10,5),datetime.datetime(2015,10,6),datetime.datetime(2015,10,7)]
is_festival_day=[]
for i in festival:
    is_festival_day.append((i-d1).days)
DATA_2['is_festival_day']=np.zeros(244)
for i in is_festival_day:
    DATA_2['is_festival_day'].iloc[i]=1
DATA_2['is_weekend']=np.zeros(244)
for i in range(244):
    if DATA_2['weekday'].iloc[i] in [6,7]:
        DATA_2['is_weekend'].iloc[i]=1
    else:
        DATA_2['is_weekend'].iloc[i]=0


########################
for name in dianbo.columns:
    DATA_2[('_').join([name,'C'])]=np.zeros(244)
    DATA_2[('_').join([name,'C'])].iloc[:183]=shoucang[name]
    DATA_2[('_').join([name,'D'])]=np.zeros(244)
    DATA_2[('_').join([name,'D'])].iloc[:183]=download[name]
    DATA_2[('_').join([name,'m7'])]=np.zeros(244)
    DATA_2[('_').join([name,'m7'])].iloc[:183]=seven_mean[name]
    DATA_2[('_').join([name,'P'])]=np.zeros(244)
    DATA_2[('_').join([name,'P'])].iloc[:183]=dianbo[name]
################
def predict_m7(n_,data_original):
    Final_Predict=pd.DataFrame(index=pd.period_range('20150831','20151030',freq='d'))
    for name in dianbo.columns:
        lag=1
        column=['weekday_1','weekday_2','weekday_3','weekday_4','weekday_5','weekday_6','weekday_7','is_festival_day','is_weekend',('_').join([name,str(1)]),('_').join([name,str(2)]),
       ('_').join([name,str(3)]),('_').join([name,str(4)]),('_').join([name,str(5)])]
        data=data_original[name].copy()
        data=pd.DataFrame(data) 
        X_test_0=[]   
        flag=0
        if min(data.iloc[:,0])==0:
            test=data
            for k in range(1,lag+1):
                X_test_0.append(data.iloc[-k,0])
            flag=1
        else:
            test=np.log(data)
            for k in range(1,lag+1):
                X_test_0.append(np.log(data.iloc[-k,0]))
        for k in range(1,lag+1):
            exec("test['lag%s']=test.iloc[:,0].shift(k)"%k)
        test[column]=DATA_2[column].iloc[:183,:]
        X=test.iloc[7:,1:]
        Y=test.iloc[7:,0]
    
        est = RandomForestRegressor(n_estimators=n_).fit(X, Y)
        feature=DATA_2[column].copy()
        feature_predict=[]
        for k in range(61):
            feature_predict.append(np.array(feature.iloc[183+k,:]))
        X__test=[]
        for k in X_test_0:
            X__test.append(k)
        for k in range(61):
            Z=X__test[-lag:].copy()
            for element in feature_predict[k]:
                Z.append(element)
    
            X__test.append(float(est.predict(Z)))
        predict=X__test[-61:]
        if flag==0:
            predict=np.exp(predict)
        Final_Predict[name]=predict 
    return Final_Predict


def predict(n_,data_original):
    Final_Predict=pd.DataFrame(index=pd.period_range('20150831','20151030',freq='d'))
    for name in dianbo.columns:
        lag=1
        column=['weekday_1','weekday_2','weekday_3','weekday_4','weekday_5','weekday_6','weekday_7','is_festival_day','is_weekend',('_').join([name,str(1)]),('_').join([name,str(2)]),
       ('_').join([name,str(3)]),('_').join([name,str(4)]),('_').join([name,str(5)])]
        data=data_original[name].copy()
        data=pd.DataFrame(data) 
        X_test_0=[]   
        flag=0
        if min(data.iloc[:,0])==0:
            test=data
            for k in range(1,lag+1):
                X_test_0.append(data.iloc[-k,0])
            flag=1
        else:
            test=np.log(data)
            for k in range(1,lag+1):
                X_test_0.append(np.log(data.iloc[-k,0]))
        for k in range(1,lag+1):
            exec("test['lag%s']=test.iloc[:,0].shift(k)"%k)
        test[column]=DATA_2[column].iloc[:183,:]
        X=test.iloc[lag:,1:]
        Y=test.iloc[lag:,0]
    
        est = RandomForestRegressor(n_estimators=n_).fit(X, Y)
        feature=DATA_2[column].copy()
        feature_predict=[]
        for k in range(61):
            feature_predict.append(np.array(feature.iloc[183+k,:]))
        X__test=[]
        for k in X_test_0:
            X__test.append(k)
        for k in range(61):
            Z=X__test[-lag:].copy()
            for element in feature_predict[k]:
                Z.append(element)
    
            X__test.append(float(est.predict(Z)))
        predict=X__test[-61:]
        if flag==0:
            predict=np.exp(predict)
        Final_Predict[name]=predict 
    return Final_Predict
###################
Final_Predict_C=predict(100,shoucang)
Final_Predict_D=predict(100,download)
Final_Predict_P=predict(100,dianbo)
Final_Predict_m7=predict_m7(100,seven_mean)
Final_c=Final_Predict_C
Final_d=Final_Predict_D
Final_m=Final_Predict_m7
Final_p=Final_Predict_P

#####################
def predict_IRA_M(n_,Final_Predict_1,Final_Predict_2,Final_Predict_3):
    Final_Predict=pd.DataFrame(index=pd.period_range('20150831','20151030',freq='d'))
    for name in dianbo.columns:
        lag=1
        column=['weekday_1','weekday_2','weekday_3','weekday_4','weekday_5','weekday_6','weekday_7','is_festival_day','is_weekend',('_').join([name,str(1)]),('_').join([name,str(2)]),
       ('_').join([name,str(3)]),('_').join([name,str(4)]),('_').join([name,str(5)]),('_').join([name,'C']),('_').join([name,'D']),('_').join([name,'P'])]
        column_=['weekday_1','weekday_2','weekday_3','weekday_4','weekday_5','weekday_6','weekday_7','is_festival_day','is_weekend',('_').join([name,str(1)]),('_').join([name,str(2)]),
       ('_').join([name,str(3)]),('_').join([name,str(4)]),('_').join([name,str(5)])]
        data=seven_mean[name].copy()
        data=pd.DataFrame(data) 
        X_test_0=[]   
        flag=0
        if min(data.iloc[:,0])==0:
            test=data
            for k in range(1,lag+1):
                X_test_0.append(data.iloc[-k,0])
            flag=1
        else:
            test=np.log(data)
            for k in range(1,lag+1):
                X_test_0.append(np.log(data.iloc[-k,0]))
        for k in range(1,lag+1):
            exec("test['lag%s']=test.iloc[:,0].shift(k)"%k)
        test[column]=DATA_2[column].iloc[:183,:]
        X=test.iloc[7:,1:]
        Y=test.iloc[7:,0]
    
        est = RandomForestRegressor(n_estimators=n_).fit(X, Y)
        feature=DATA_2[column_].copy()
        feature_predict=[]
        for k in range(61):
            feature_predict.append(np.array(feature.iloc[183+k,:]))
        X__test=[]
        for k in X_test_0:
            X__test.append(k)
        for k in range(61):
            Z=X__test[-lag:].copy()
            for element in feature_predict[k]:
                Z.append(element)
            Z.append(Final_Predict_1[name].iloc[k])
            Z.append(Final_Predict_2[name].iloc[k])
            Z.append(Final_Predict_3[name].iloc[k])
            X__test.append(float(est.predict(Z)))
        predict=X__test[-61:]
        if flag==0:
            predict=np.exp(predict)
        Final_Predict[name]=predict 
    return Final_Predict

def predict_IRA_D(n_,Final_Predict_1,Final_Predict_2,Final_Predict_3):
    Final_Predict=pd.DataFrame(index=pd.period_range('20150831','20151030',freq='d'))
    for name in dianbo.columns:
        lag=1
        column=['weekday_1','weekday_2','weekday_3','weekday_4','weekday_5','weekday_6','weekday_7','is_festival_day','is_weekend',('_').join([name,str(1)]),('_').join([name,str(2)]),
       ('_').join([name,str(3)]),('_').join([name,str(4)]),('_').join([name,str(5)]),('_').join([name,'m7']),('_').join([name,'C']),('_').join([name,'P'])]
        column_=['weekday_1','weekday_2','weekday_3','weekday_4','weekday_5','weekday_6','weekday_7','is_festival_day','is_weekend',('_').join([name,str(1)]),('_').join([name,str(2)]),
       ('_').join([name,str(3)]),('_').join([name,str(4)]),('_').join([name,str(5)])]
        data=download[name].copy()
        data=pd.DataFrame(data) 
        X_test_0=[]   
        flag=0
        if min(data.iloc[:,0])==0:
            test=data
            for k in range(1,lag+1):
                X_test_0.append(data.iloc[-k,0])
            flag=1
        else:
            test=np.log(data)
            for k in range(1,lag+1):
                X_test_0.append(np.log(data.iloc[-k,0]))
        for k in range(1,lag+1):
            exec("test['lag%s']=test.iloc[:,0].shift(k)"%k)
        test[column]=DATA_2[column].iloc[:183,:]
        X=test.iloc[7:,1:]
        Y=test.iloc[7:,0]
    
        est = RandomForestRegressor(n_estimators=n_).fit(X, Y)
        feature=DATA_2[column_].copy()
        feature_predict=[]
        for k in range(61):
            feature_predict.append(np.array(feature.iloc[183+k,:]))
        X__test=[]
        for k in X_test_0:
            X__test.append(k)
        for k in range(61):
            Z=X__test[-lag:].copy()
            for element in feature_predict[k]:
                Z.append(element)
            Z.append(Final_Predict_1[name].iloc[k])
            Z.append(Final_Predict_2[name].iloc[k])
            Z.append(Final_Predict_3[name].iloc[k])
            X__test.append(float(est.predict(Z)))
        predict=X__test[-61:]
        if flag==0:
            predict=np.exp(predict)
        Final_Predict[name]=predict 
    return Final_Predict


def predict_IRA_C(n_,Final_Predict_1,Final_Predict_2,Final_Predict_3):
    Final_Predict=pd.DataFrame(index=pd.period_range('20150831','20151030',freq='d'))
    for name in dianbo.columns:
        lag=1
        column=['weekday_1','weekday_2','weekday_3','weekday_4','weekday_5','weekday_6','weekday_7','is_festival_day','is_weekend',('_').join([name,str(1)]),('_').join([name,str(2)]),
       ('_').join([name,str(3)]),('_').join([name,str(4)]),('_').join([name,str(5)]),('_').join([name,'m7']),('_').join([name,'D']),('_').join([name,'P'])]
        column_=['weekday_1','weekday_2','weekday_3','weekday_4','weekday_5','weekday_6','weekday_7','is_festival_day','is_weekend',('_').join([name,str(1)]),('_').join([name,str(2)]),
       ('_').join([name,str(3)]),('_').join([name,str(4)]),('_').join([name,str(5)])]
        data=shoucang[name].copy()
        data=pd.DataFrame(data) 
        X_test_0=[]   
        flag=0
        if min(data.iloc[:,0])==0:
            test=data
            for k in range(1,lag+1):
                X_test_0.append(data.iloc[-k,0])
            flag=1
        else:
            test=np.log(data)
            for k in range(1,lag+1):
                X_test_0.append(np.log(data.iloc[-k,0]))
        for k in range(1,lag+1):
            exec("test['lag%s']=test.iloc[:,0].shift(k)"%k)
        test[column]=DATA_2[column].iloc[:183,:]
        X=test.iloc[7:,1:]
        Y=test.iloc[7:,0]
    
        est = RandomForestRegressor(n_estimators=n_).fit(X, Y)
        feature=DATA_2[column_].copy()
        feature_predict=[]
        for k in range(61):
            feature_predict.append(np.array(feature.iloc[183+k,:]))
        X__test=[]
        for k in X_test_0:
            X__test.append(k)
        for k in range(61):
            Z=X__test[-lag:].copy()
            for element in feature_predict[k]:
                Z.append(element)
            Z.append(Final_Predict_1[name].iloc[k])
            Z.append(Final_Predict_2[name].iloc[k])
            Z.append(Final_Predict_3[name].iloc[k])
            X__test.append(float(est.predict(Z)))
        predict=X__test[-61:]
        if flag==0:
            predict=np.exp(predict)
        Final_Predict[name]=predict 
    return Final_Predict


def predict_IRA_P(n_,Final_Predict_1,Final_Predict_2,Final_Predict_3):
    Final_Predict=pd.DataFrame(index=pd.period_range('20150831','20151030',freq='d'))
    for name in dianbo.columns:
        lag=1
        column=['weekday_1','weekday_2','weekday_3','weekday_4','weekday_5','weekday_6','weekday_7','is_festival_day','is_weekend',('_').join([name,str(1)]),('_').join([name,str(2)]),
       ('_').join([name,str(3)]),('_').join([name,str(4)]),('_').join([name,str(5)]),('_').join([name,'m7']),('_').join([name,'C']),('_').join([name,'D'])]
        column_=['weekday_1','weekday_2','weekday_3','weekday_4','weekday_5','weekday_6','weekday_7','is_festival_day','is_weekend',('_').join([name,str(1)]),('_').join([name,str(2)]),
       ('_').join([name,str(3)]),('_').join([name,str(4)]),('_').join([name,str(5)])]
        data=dianbo[name].copy()
        data=pd.DataFrame(data) 
        X_test_0=[]   
        flag=0
        if min(data.iloc[:,0])==0:
            test=data
            for k in range(1,lag+1):
                X_test_0.append(data.iloc[-k,0])
            flag=1
        else:
            test=np.log(data)
            for k in range(1,lag+1):
                X_test_0.append(np.log(data.iloc[-k,0]))
        for k in range(1,lag+1):
            exec("test['lag%s']=test.iloc[:,0].shift(k)"%k)
        test[column]=DATA_2[column].iloc[:183,:]
        X=test.iloc[7:,1:]
        Y=test.iloc[7:,0]
    
        est = RandomForestRegressor(n_estimators=n_).fit(X, Y)
        feature=DATA_2[column_].copy()
        feature_predict=[]
        for k in range(61):
            feature_predict.append(np.array(feature.iloc[183+k,:]))
        X__test=[]
        for k in X_test_0:
            X__test.append(k)
        for k in range(61):
            Z=X__test[-lag:].copy()
            for element in feature_predict[k]:
                Z.append(element)
            Z.append(Final_Predict_1[name].iloc[k])
            Z.append(Final_Predict_2[name].iloc[k])
            Z.append(Final_Predict_3[name].iloc[k])
            X__test.append(float(est.predict(Z)))
        predict=X__test[-61:]
        if flag==0:
            predict=np.exp(predict)
        Final_Predict[name]=predict 
    return Final_Predict

##############
 while(1):
    Final_C=predict_IRA_C(100,Final_m,Final_d,Final_p)
    Final_D=predict_IRA_D(100,Final_m,Final_c,Final_p)
    Final_P=predict_IRA_P(100,Final_m,Final_c,Final_d)
    Final_M=predict_IRA_M(100,Final_c,Final_d,Final_p)
    sum_=0
    a=Final_P-Final_p
    for i in range(100):
        sum_+=sum(abs(a.iloc[:,i]))
    print(sum_)
    Final_c=Final_C
    Final_d=Final_D
    Final_p=Final_P
    Final_m=Final_M




##########
dic={}
for name in dianbo.columns:
    dic[name]=0
for name in Final_P.columns:
    ar=np.array(Final_P[name])
    total=0
    totoal_2=0
    for i in ar:
        total=1/i+total
        totoal_2=totoal_2+1/(i*i)
    dic[name]=round(total/totoal_2)
Final_P___=pd.DataFrame()
for name in dianbo.columns:
    Final_P___[name]=np.zeros(60)
for name in Final_P.columns:
    for i in range(60):
        Final_P___[name].iloc[i]=dic[name]

#########
date_transform=[]
for i in range(60):
    date_transform.append(('').join(str(pd.period_range('20150901','20151030',freq='d')[i]).split('-')))
for i in range(60):
    date_transform[i]=int(date_transform[i])

result=pd.DataFrame(index=range(6000))
itemer=0
result['artist_id']=0
result['dianbo']=0
result['date']=0
for name in dianbo.columns:
    result['artist_id'].iloc[itemer*60:(itemer+1)*60]=name
    for i in range(60):
        result['dianbo'].iloc[itemer*60+i]=Final_P___[name].iloc[i]
        result['date'].iloc[itemer*60+i]=date_transform[i]
    itemer=1+itemer
result.to_csv(r'E:\data\aliyun_new\result_ira.csv',header=None,index=None,encoding='UTF-8')
