# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the required library and read the dataframe.
2.Write a function computeCost to generate the cost function.
3.Perform iterations og gradient steps with learning rate.
4.Plot the Cost function using Gradient Descent and generate the required graph.
```

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Sowmiya R
RegisterNumber: 212225040420 
*/


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("50_Startups.csv")
x=data["R&D Spend"].values
y=data["Profit"].values

x_mean=np.mean(x)
x_std=np.std(x)
x=(x-x_mean)/x_std

w=0.0
b=0.0
alpha=0.01
epochs=100
n=len(x)

losses=[]

for i in range(epochs):
    y_hat=w*x+b
    loss=np.mean((y_hat-y)**2)
    losses.append(loss)
    
    dw=(2/n)*np.sum((y_hat-y)*x)
    db=(2/n)*np.sum(y_hat-y)
    
    w-=alpha*dw
    b-=alpha*db

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(losses)
plt.xlabel("Iterations")
plt.ylabel("Loss(MSE)")
plt.title("Loss vs Iterations")

plt.subplot(1,2,2)
plt.scatter(x,y)
x_sorted=np.argsort(x)
plt.plot(x[x_sorted],(w*x+b)[x_sorted],color="red")
plt.xlabel("R&D Spend (scaled)")
plt.ylabel("Profit")
plt.title("Linear Regression Fit")
plt.tight_layout()
plt.show()

print(f"Final weight (w): {w}")
print(f"Final bias (b): {b}")
```

## Output:
<img width="1385" height="600" alt="image" src="https://github.com/user-attachments/assets/f1dfbf5d-68c2-4227-af8b-865bc88f286c" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
