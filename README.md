# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. import necessary libraries
2.load the dataset and define the features and target the variables 
3. split the dataset into training and testing sets, train the model to the dataset
4. make predictions on dataset,evaluate the model
```
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by:somalarajurohini 
RegisterNumber:  24000337
*/
```
```
import chardet
file = "spam.csv"
with open(file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data = pd.read_csv("spam.csv",encoding = 'windows-1252')
data.head()
data.info()
data.isnull().sum()
x = data['v1'].values
y = data['v2'].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)
from sklearn.svm import SVC
svc = SVC(kernel = 'linear')
svc.fit(x_train,y_train)
y_pred = svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy = metrics.accuracy_score(y_pred,y_test)
accuracy
```

## Output:
![SVM For Spam Mail Detection](sam.png)


{'encoding': 'Windows-1252', 'confidence': 0.7270322499829184, 'language': ''}





![Screenshot 2024-12-26 103512](https://github.com/user-attachments/assets/5cc010bb-3632-4603-b912-4683f4c43cdb)





![Screenshot 2024-12-26 103517](https://github.com/user-attachments/assets/57931b91-7da5-4cbe-a8f2-2b6bc7ca2a09)






![Screenshot 2024-12-26 103522](https://github.com/user-attachments/assets/c2e364e5-b308-40f2-aeeb-2f007df50531)







![Screenshot 2024-12-26 103529](https://github.com/user-attachments/assets/75d179a9-d5e9-416d-a7ca-f152126ebe30)


0.003587443946188341




## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
