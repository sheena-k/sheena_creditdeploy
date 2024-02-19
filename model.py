import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
import pickle
import gzip

data= pd.read_csv("credit.csv")
rows_drop = data[data["Credit_Score"]=="Standard"].head(25000).index
data =data.drop(rows_drop)
rows_drop2 = data[data["Credit_Score"]=="Poor"].head(10000).index
data =data.drop(rows_drop2)

#outliers for Annual Income columns
q1_ai = np.percentile(data['Annual_Income'],25,method="midpoint")
q2_ai = np.percentile(data['Annual_Income'],50,method="midpoint")
q3_ai = np.percentile(data['Annual_Income'],75,method="midpoint")
iqr = q3_ai- q1_ai
outlier_ai=[]
up_lim_ai = q3_ai+ 1.5*iqr
low_lim_ai = q1_ai-1.5*iqr
for x in data['Annual_Income']:
    if (x<low_lim_ai) or (x>up_lim_ai):
        outlier_ai.append(x)
ind1 = data["Annual_Income"] > up_lim_ai
index_list1 = data.loc[ind1].index.tolist()
data =data.drop(index_list1)
#outliers for Age columns
q1_ag = np.percentile(data["Age"],25,method="midpoint")
q2_ag = np.percentile(data["Age"],50,method="midpoint")
q3_ag = np.percentile(data['Age'],75,method="midpoint")
iqr_ag = q3_ag- q1_ag
outlier_ag=[]
up_lim_ag = q3_ag+ 1.5*iqr_ag
low_lim_ag= q1_ag-1.5*iqr_ag
for x in data['Age']:
    if (x<low_lim_ag) or (x>up_lim_ag):
        outlier_ag.append(x)
ind2 = data["Age"] > up_lim_ag
index_list2 = data.loc[ind2].index.tolist()
data =data.drop(index_list2)
#outliers for Monthly Balance columns
q1_mb = np.percentile(data["Monthly_Balance"],25,method="midpoint")
q2_mb = np.percentile(data['Monthly_Balance'],50,method="midpoint")
q3_mb = np.percentile(data['Monthly_Balance'],75,method="midpoint")
iqr_mb = q3_mb- q1_mb
outlier_mb=[]
up_lim_mb = q3_mb+ 1.5*iqr_mb
low_lim_mb= q1_mb-1.5*iqr_mb
for x in data['Monthly_Balance']:
    if (x<low_lim_mb) or (x>up_lim_mb):
        outlier_mb.append(x)
ind3 = data["Monthly_Balance"] > up_lim_mb
index_list3 = data.loc[ind3].index.tolist()
data =data.drop(index_list3)
data["Age"] = data["Age"].astype(int)
data["Num_of_Loan"] = data["Num_of_Loan"].astype(int)
data["Num_Bank_Accounts"] =data["Num_Bank_Accounts"].astype(int)
data["Credit_History_Age"] = data["Credit_History_Age"].astype(int)
data["Num_Credit_Inquiries"] = data["Num_Credit_Inquiries"].astype(int)
data["Num_Credit_Card"] = data["Num_Credit_Card"].astype(int)
data["Delay_from_due_date"] = data["Delay_from_due_date"].astype(int)
data["Num_of_Delayed_Payment"] = data["Num_of_Delayed_Payment"].astype(int)
data["Changed_Credit_Limit"] = data["Changed_Credit_Limit"].astype(int)

train=data.drop(["ID","Customer_ID","SSN","Month","Name","Type_of_Loan","Occupation",
                 "Amount_invested_monthly","Credit_Utilization_Ratio","Monthly_Inhand_Salary"],axis=1)
le= LabelEncoder()
train["Credit_Score"] = le.fit_transform(train["Credit_Score"])
train["Payment_Behaviour"] = le.fit_transform(train["Payment_Behaviour"])
train["Payment_of_Min_Amount"] = le.fit_transform(train["Payment_of_Min_Amount"])
train["Credit_Mix"] = le.fit_transform(train["Credit_Mix"])

X = train.drop(["Credit_Score"],axis=1)
Y = pd.DataFrame(train["Credit_Score"])

smote = SMOTE()
X,Y = smote.fit_resample(X,Y)



scaler_x = MinMaxScaler()
X_scale= scaler_x.fit_transform(X)

Y= np.squeeze(Y)

x_train,x_test,y_train,y_test = train_test_split(X_scale,Y, test_size = 0.3, random_state=42)

rf_cls =RandomForestClassifier()

model_rf = rf_cls.fit(x_train,y_train)

y_pred_rf = model_rf.predict(x_test)

pipeline = make_pipeline(MinMaxScaler(),RandomForestClassifier())

model = pipeline.fit(x_train, y_train)
y_pred = model.predict(x_test)

#filename="model.pickle"
#with gzip.open(filename,"wb") as file:
   #pickle.dump(model,file)