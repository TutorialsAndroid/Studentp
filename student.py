import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder , StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score

a = pd.read_csv("student_performance.csv")

# print(a.isnull().sum())

en = LabelEncoder()

a['grade'] = en.fit_transform(a['grade'])

s = StandardScaler()

# print(a)

X= a.drop(columns='grade')
y= a['grade']

X = s.fit_transform(X)


X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)

log = LogisticRegression(max_iter=10000)
m= log.fit(X_train,y_train)
n = m.predict(X_test)
print(n)

print("Accuracy Score : ",accuracy_score(y_test,n))

o = SVC()
p = o.fit(X_train,y_train)
q = p.predict(X_test)
print("Accuracy Score : ",accuracy_score(q))
