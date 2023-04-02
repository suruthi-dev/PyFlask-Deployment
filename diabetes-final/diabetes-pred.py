import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("diabetes.csv")
df.head()
df = df.loc[:,['Glucose','BloodPressure','Insulin','Age','Outcome']]
sns.heatmap(df.corr(),annot=True,cmap='YlOrRd')

allFeatures = df.drop('Outcome',axis=1)
labels = df['Outcome']

sss = StratifiedShuffleSplit(n_splits = 4 ,test_size = 0.30,random_state =42)

for train_index,test_index in sss.split(allFeatures,labels):
    x_train_all ,x_test_all = allFeatures.iloc[train_index],allFeatures.iloc[test_index]
    y_train_all , y_test_all = labels.iloc[train_index],labels.iloc[test_index]
    

x_train_all.shape


x_test_all.shape


logreg_all  = LogisticRegression()
logreg_all.fit(x_train_all,y_train_all)
y_pred_all = logreg_all.predict(x_test_all)

accuracy_score(y_test_all,y_pred_all)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_all,y_pred_all)


sns.heatmap(cm,annot=True)


from sklearn.metrics import roc_auc_score

y_pred_proba = logreg_all.predict_proba(x_test_all)[:,1]
auc = roc_auc_score(y_test_all,y_pred_proba)
auc



import pickle


with open("model.pkl",'wb')as f:
    pickle.dump(logreg_all,f)
