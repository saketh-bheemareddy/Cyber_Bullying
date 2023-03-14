
import pandas as pd
import numpy as np
# ML Packages For Vectorization of Text For Feature Extraction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# Visualization Packages
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
#from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# check = tweet
df = pd.read_csv("new_data.csv")
df_data = df[["tweet","label"]]
df_data.columns
df_x = df_data['tweet']
df_y = df_data['label']

    # #### Feature Extraction From Text
    # + CountVectorizer
    # + TfidfVectorizer

    # Extract Feature With CountVectorizer
corpus = df_x
    #print(corpus)
cv = CountVectorizer()
X = cv.fit_transform(corpus) # Fit the Data
    #print(X)

X.toarray()

    # get the feature names
cv.get_feature_names()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X , df_y, test_size=0.33, random_state=42)
X_train
    # Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
clf1 = MultinomialNB()
clf1.fit(X_train,y_train)
clf1.score(X_test,y_test)
print("Accuracy of Model",clf1.score(X_test,y_test)*100,"%")



clf2 = RandomForestClassifier()
clf2.fit(X_train,y_train)
clf2.score(X_test,y_test)
print("Accuracy of Model",clf1.score(X_test,y_test)*100,"%")



clf=linear_model.LogisticRegression(fit_intercept=False)
clf.fit(X_train,y_train)
clf.score(X_test,y_test)

    # Accuracy of our Model
print("Accuracy of Model",clf.score(X_test,y_test)*100,"%")
    




    ### Save the Model 

import pickle 

naivebayesML = open("Cyberbull.pkl","wb")

pickle.dump(clf,naivebayesML)

naivebayesML.close()


    # Load the model


ytb_model = open("Cyberbull.pkl","rb")

new_model = pickle.load(ytb_model)

new_model    


    # Sample Prediciton 3
comment2 = ["hi how r u"]
vect = cv.transform(comment2).toarray()
    #new_model.predict(vect)   

    
if new_model.predict(vect) == 1:
    result = True
    print("bullying")
else:
    result = False
    print("Not bullying")        
  



