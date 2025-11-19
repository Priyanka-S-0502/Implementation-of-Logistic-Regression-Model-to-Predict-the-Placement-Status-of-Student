# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the required packages and print the present data.

2.Print the placement data and salary data.

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices.

5.Display the results.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by:PRIYANKA S 
RegisterNumber: 212224040255
*/
```
```
import pandas as pd
data=pd.read_csv(r"C:\Users\admin\Desktop\Placement_Data.csv")
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print(y_pred)
print()
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)
print()
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
print(confusion)
print()
from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
print("NAME : PRIYANKA S")
print("Register No: 212224040255")
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:

Head

<img width="793" height="181" alt="image" src="https://github.com/user-attachments/assets/e6f4ce98-6fe9-40a4-a37b-b4a4845fbfeb" />

Copy

<img width="794" height="190" alt="image" src="https://github.com/user-attachments/assets/47dbfef1-9d52-4a8c-9b2c-2b238f3689b1" />

Fit Transform

<img width="1167" height="601" alt="image" src="https://github.com/user-attachments/assets/ffc0e180-c171-4e1f-8172-3934301e4130" />

X

<img width="1096" height="600" alt="image" src="https://github.com/user-attachments/assets/c0aaa7ca-b81b-462c-b6df-1adaf7e6a6a5" />


Y

<img width="482" height="322" alt="image" src="https://github.com/user-attachments/assets/eea1614d-4c27-4a9c-acec-3430433c9ef7" />


Logistic Regression

<img width="808" height="56" alt="image" src="https://github.com/user-attachments/assets/d3e71169-1130-4f79-adb1-930e3f7c6209" />

Accuracy


<img width="230" height="51" alt="image" src="https://github.com/user-attachments/assets/9ad56c17-9983-4b23-b965-2094ecceb8ac" />


Confusion

<img width="186" height="74" alt="image" src="https://github.com/user-attachments/assets/ded0047e-76ce-4a05-8f1b-da241c4b3bd3" />


Classification Report

<img width="620" height="216" alt="image" src="https://github.com/user-attachments/assets/01f5c35b-a190-4e3e-b6e6-58e2f41d210b" />

Prediction


<img width="820" height="175" alt="image" src="https://github.com/user-attachments/assets/ae1100a2-1039-47e6-998c-50b4a0000ad5" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
