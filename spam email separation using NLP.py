# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 15:59:39 2020

@author: sravillu
"""


import pandas as pd
import nltk 
import numpy as np
import matplotlib as plt
import seaborn as sns
nltk.download('stopwords')  # stopwords pakage is a preexisting list of stopwords  (the, is , this, there...)
from nltk.corpus import stopwords 

messages =[line.rstrip() for line in open('C:/Users/sravillu.ORADEV/Downloads/original/Refactored_Py_DS_ML_Bootcamp-master/20-Natural-Language-Processing/smsspamcollection/SMSSpamCollection')]

messages= pd.read_csv('C:/Users/sravillu.ORADEV/Downloads/original/Refactored_Py_DS_ML_Bootcamp-master/20-Natural-Language-Processing/smsspamcollection/SMSSpamCollection',sep='\t',names=['label','message'])


# =============================================================================
# create a column for length of messages
# =============================================================================
messages['length']=messages['message'].apply(len)

# =============================================================================
# plot messages for data vistualisation
# =============================================================================

messages.plot.hist(bins=100)

messages.describe()

# =============================================================================
# data preprocession

# =============================================================================
#from nltk import stopwords
from nltk.stem import PorterStemmer
import re
corpus=[]

for i in range(0,5572):
    message_new= re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    # review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])    
    message_new=message_new.lower()
    message_new=message_new.split()
    
    ps=PorterStemmer()
    
    message_new=[ps.stem(word) for word in message_new if not word in stopwords.words('english')]
    
    message_new=' '.join(message_new)
    
    corpus.append(message_new)

# import string

# m='SandeepR is genius and rich and intelligent '

# def text_preprocess(m):
#     message_new=[char for char in m not in string.punctuation]
#     message_new=' '.join(message_new)
#     return[word for word in message_new.split() if word.lower() not in  stopwords.word('english')]


# print(m)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer() 

X=cv.fit_transform(corpus).toarray()
y=messages['label'].values


from sklearn.feature_extraction.text import TfidfTransformer

tfidf=TfidfTransformer() # log d/t weighs -document devided bynumber of terms in the document
tfidf.fit(X)
X=tfidf.transform(X).toarray()

# =============================================================================
# 
# =============================================================================

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# =============================================================================
# model building
# =============================================================================

from sklearn.naive_bayes import GaussianNB

lm=GaussianNB()

lm.fit(X_train, y_train)

y_pred=lm.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score

cm=confusion_matrix(y_test, y_pred)
ac=accuracy_score(y_test, y_pred)

















