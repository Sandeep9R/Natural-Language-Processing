# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 17:42:01 2020

@author: sravillu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

messages=pd.read_csv("06Restaurant_Reviews.tsv",delimiter='\t',quoting=3)

corpus=[]

for i in range(0, 1000):
        review=re.sub('[^a-zA-Z]', ' ', messages['Review'][i]) # cleaned reviews in variable review, this step will only keep the letters from A-z in the review and  remove  the numbers, puntuation part, exclanmations,question marks
        review=review.lower()
        review=review.split()
        
        stemmer=PorterStemmer()
        review=[stemmer.stem(word) for word in review if not word in stopwords.words('english')]
        
        
        review=''.join(review)
        
        corpus.append(review)
        
from sklearn.feature_extraction.text import CountVectorizer

vectorizer=CountVectorizer()

X=vectorizer.fit_transform(corpus).toarray()

y=messages.iloc[:,1]


# # =============================================================================
# # Tfidf
# # =============================================================================

from sklearn.feature_extraction.text import TfidfTransformer

tfidf=TfidfTransformer()

tfidf.fit(X)
X=tfidf.transform(X).toarray()

# =============================================================================
# Naive Bayes 
# =============================================================================
        
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_pred)
acc= accuracy_score(y_test,y_pred)







