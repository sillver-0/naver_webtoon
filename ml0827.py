#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

#데이터 불러오는 함수
def load_data(filename):
    return pd.read_csv(filename)


# In[2]:


train_mcomment = load_data("0827trn.csv")
test_mcomment = load_data("0827tst.csv")
train_pcomment = load_data("0810trn.csv")
test_pcomment = load_data("0810tst.csv")


# In[3]:


from sklearn.base import BaseEstimator, TransformerMixin

# 사이킷런이 DataFrame을 바로 사용하지 못하므로
# 수치형이나 범주형 컬럼을 선택하는 클래스를 만듭니다.
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]


# In[4]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.svm import SVC
#수치형 변수 처리
imputer = Imputer(strategy="median")

num_mpipeline = Pipeline([
        ("select_numeric", DataFrameSelector(["좋아요", "평점", "댓글수 평균", "1화 댓글수"])),
        ("imputer", Imputer(strategy="median")),
        ('scaler', StandardScaler()),
    ])


# In[5]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.svm import SVC
#수치형 변수 처리
imputer = Imputer(strategy="median")

num_ppipeline = Pipeline([
        ("select_numeric", DataFrameSelector(["좋아요", "평점", "댓글수 평균", "1화 댓글수", "긍정"])),
        ("imputer", Imputer(strategy="median")),
        ('scaler', StandardScaler()),
    ])


# In[6]:


num_mpipeline.fit_transform(train_mcomment)


# In[7]:


num_ppipeline.fit_transform(train_pcomment)


# In[8]:


class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],
                                       index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)


# In[9]:


from sklearn.preprocessing import OneHotEncoder
#범주형 변수 처리
cat_pipeline = Pipeline([
        ("select_cat", DataFrameSelector(["장르"])),
        ("imputer", MostFrequentImputer()),
        ("cat_encoder", OneHotEncoder(sparse=False)),
    ])


# In[10]:


cat_pipeline.fit_transform(train_mcomment)


# In[11]:


cat_pipeline.fit_transform(train_pcomment)


# In[12]:


from sklearn.pipeline import FeatureUnion
preprocess_mpipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_mpipeline),
        ("cat_pipeline", cat_pipeline),
    ])


# In[13]:


from sklearn.pipeline import FeatureUnion
preprocess_ppipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_ppipeline),
        ("cat_pipeline", cat_pipeline),
    ])


# In[14]:


X_mtrain = preprocess_mpipeline.fit_transform(train_mcomment)
X_mtrain


# In[15]:


X_ptrain = preprocess_ppipeline.fit_transform(train_pcomment)
X_ptrain


# In[16]:


y_mtrain = train_mcomment["성공"]
X_mtest = preprocess_mpipeline.transform(test_mcomment)
y_mtest = test_mcomment["성공"]


# In[17]:


y_ptrain = train_pcomment["성공"]
X_ptest = preprocess_ppipeline.transform(test_pcomment)
y_ptest = test_pcomment["성공"]


# In[18]:


from sklearn import linear_model
#로지스틱 회귀
lg_clf = linear_model.LogisticRegression()
lg_clf.fit(X_ptrain, y_ptrain)


# In[19]:


lg_pred = lg_clf.predict(X_ptest)


# In[71]:


from sklearn.ensemble import RandomForestClassifier
#랜덤포레스트
rnd_clf = RandomForestClassifier(n_estimators = 30, n_jobs = -1)
rnd_clf.fit(X_mtrain, y_mtrain)


# In[72]:


rnd_pred = rnd_clf.predict(X_mtest)


# In[20]:


from sklearn.ensemble import AdaBoostClassifier
#아다부스트
from sklearn.tree import DecisionTreeClassifier
ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 1), n_estimators = 30, algorithm = 'SAMME.R', learning_rate = 0.5)
ada_clf.fit(X_mtrain, y_mtrain)


# In[21]:


ada_pred = ada_clf.predict(X_mtest)


# In[25]:


from sklearn import metrics
print("Accuracy", metrics.accuracy_score(y_ptest, lg_pred.round()))
#print("Accuracy", metrics.accuracy_score(y_mtest, rnd_pred.round()))
print("Accuracy", metrics.accuracy_score(y_mtest, ada_pred.round()))


# In[27]:


from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt


from sklearn.metrics import roc_curve

fpr1, tpr1, thresholds1 = roc_curve(y_ptest, lg_clf.decision_function(X_ptest))
#fpr2, tpr2, thresholds2 = roc_curve(y_mtest, rnd_clf.predict_proba(X_mtest)[:, 1])
fpr3, tpr3, thresholds3 = roc_curve(y_mtest, ada_clf.decision_function(X_mtest))
x = np.linspace(0, 1, 100)
y = x

# Print ROC curve
plt.plot(fpr1, tpr1, 'r-', label = 'LR')
#plt.plot(fpr2, tpr2, 'b-', label = 'RND')
plt.plot(fpr3, tpr3, 'b-', label = 'Ada')
plt.plot(x, y, 'k--')
plt.title(r'$ROC\ Graph$')
plt.legend(loc = 'best')
plt.savefig('190827-rnd.png')
plt.show()


# Print AUC
auc = np.trapz(tpr1, fpr1)
print('AUC_logistic: ', auc)
#auc = np.trapz(tpr2, fpr2)
#print('AUC_randomforest: ', auc)
auc = np.trapz(tpr3, fpr3)
print('AUC_adaboost: ', auc)


# In[76]:


import statsmodels.api as sm
logit = sm.Logit(y_ptrain,X_ptrain) #로지스틱 회귀분석 시행
result = logit.fit()
result.summary2()

