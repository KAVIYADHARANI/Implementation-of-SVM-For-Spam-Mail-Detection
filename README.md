# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the Program.
2. Import the necessary packages.
3. Read the given csv file and display the few contents of the data.

4. Assign the features for x and y respectively.
5. Split the x and y sets into train and test sets.
6. Convert the Alphabetical data to numeric using CountVectorizer.
7. Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.
8. Find the accuracy of the model.
9. End the program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: KAVIYA D
RegisterNumber:  212223040089
*/
```
```
import pandas as pd

data=pd.read_csv("spam.csv",encoding="Windows-1252")

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
### data.head()
![Image-1](https://github.com/user-attachments/assets/3d2ee89b-b37a-4c95-87f3-7a4eadf4e213)

### data.info()
![Image-2](https://github.com/user-attachments/assets/2a210df9-a473-4bca-bab0-2cf72aec64a9)

### data.isnull().sum()
![Image-3](https://github.com/user-attachments/assets/cf7525bc-c5b1-40a2-b744-055f617988cd)

### Prediction
![Image-4](https://github.com/user-attachments/assets/52dd0435-2eda-476c-b032-95bcd81fac78)


### Accuracy 

![Image-5](https://github.com/user-attachments/assets/c59a434c-ff8e-4f03-9611-0d1fd4532b9e)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
