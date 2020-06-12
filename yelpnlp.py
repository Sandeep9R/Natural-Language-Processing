# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 20:26:25 2020

@author: sravillu
"""


import nltk
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
nltk.download('stopwords')  # stopwords pakage is a preexisting list of stopwords  (the, is , this, there...)
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer
import re


yelp=pd.read_csv('C:/Users/sravillu.ORADEV/Downloads/original/Refactored_Py_DS_ML_Bootcamp-master/20-Natural-Language-Processing/yelp.csv')

corpus=[]

yelp=yelp.drop(['business_id','date','review_id','type','user_id','cool','useful','funny'],axis=1)

yelp['length']=yelp['text'].apply(len)

sns.boxplot(x='stars',y='length',data=yelp)
plt.show()

sns.countplot(x='stars',data=yelp)

yelp_class=yelp[(yelp['stars']==1) | (yelp['stars']==5)]


X=yelp_class['text']
y=yelp_class['stars']

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

X=cv.fit_transform(X)



# =============================================================================
# tfidf
# =============================================================================
# from sklearn.feature_extraction.text import TfidfTransformer
# tfidf=TfidfTransformer()

# tfidf.fit(X)

# X=tfidf.transform(X)



# =============================================================================
# # Splitting the dataset into the Training set and Test set
# =============================================================================

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)



# =============================================================================
# Model building
# =============================================================================

from sklearn.naive_bayes import MultinomialNB

nb=MultinomialNB()
nb.fit(X_train,y_train)

y_pred=nb.predict(X_test)

# =============================================================================
# 
# =============================================================================
from sklearn.metrics import confusion_matrix,classification_report

print(confusion_matrix(y_test,y_pred)) # to be clarified 

print(classification_report(y_test,y_pred))