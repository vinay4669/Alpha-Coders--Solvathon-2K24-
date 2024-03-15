
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


df = pd.read_csv('50_Startups.csv')


df.head()


df.drop('State', axis=True, inplace=True)


df.head()


df.shape


df.corr()


# sns.heatmap(df.corr(), annot=True)
# plt.show()


df.drop('Administration', axis=True, inplace=True)


df.head()


x = df.drop('Profit', axis=True)
y = df['Profit']


x.head()


# data standardization
sc = StandardScaler()
x = sc.fit_transform(x)


x


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)


x_train.shape, x_test.shape, y_train.shape, y_test.shape


model = LinearRegression()
model.fit(x_train, y_train)


y_pred = model.predict(x_test).round(1)


calculation = pd.DataFrame(np.c_[y_test, y_pred], columns=['Original Salary', 'Predict Salary'])
calculation.head()


mse = math.sqrt(mean_squared_error(y_test, y_pred))
mse


print("Training accuracy : ", model.score(x_train, y_train))
print("Testing accuracy : ", model.score(x_test, y_test))


model.coef_


model.intercept_


x1 = x[:, 0]
x2 = x[:, 1]


x1.shape, x2.shape, y.shape


b0 = 0
b1 = 0
b2 = 0

L = 0.001
epochs = 2000

n = float(len(x1))

for i in range(epochs):
    y_pred2 = b0 + b1*x1 + b2*x2
    D_b0 = (-2/n) * sum(y - y_pred2)
    D_b1 = (-2/n) * sum(x1 * (y - y_pred2))
    D_b2 = (-2/n) * sum(x2 * (y - y_pred2))
    b0 -= L * D_b0
    b1 -= L * D_b1
    b2 -= L * D_b2

(b0, b1, b2)
    


y_pred2 = b0 + b1*x1 + b2*x2


calculation = pd.DataFrame(np.c_[y, y_pred2], columns=['Original Salary', 'Predict Salary'])
calculation.head()


mse = math.sqrt(mean_squared_error(y, y_pred2))
mse





