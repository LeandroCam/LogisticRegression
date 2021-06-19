# Chance of admission

Simple models are better for understanding the impact & importance of each feature on a response variable.
The target of this analyse is linear, linear model, linear regression triumphs over all other machine learning methods.
With this dataset we'll predict the chance of Graduate Admissions from an Indian perspective. The dataset can be find in https://www.kaggle.com/mohansacharya/graduate-admissions
Fist we load and analyse the data we have.




```python
import pandas as pd

df = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')
df.head()

```

[head](https://user-images.githubusercontent.com/83521233/122256502-8159be00-cea5-11eb-9ac5-49af270a9a20.png)

As we can see they are all numericall values so we don't need to use and encoding thecnics such as ordinal encoder
or one-hot-encoder. 

Response Variable = Chance of Admit

We have seven features to predict the response variable. Based on the permutation feature importances shown in figure, GCPA is the most important feature, and TOFL Score is the last important feature.

[Importance](https://user-images.githubusercontent.com/83521233/117573956-33d18080-b0b1-11eb-8ef3-62eeee7425d8.png)

Code for permutation feature importance


```python
import pandas as pd
import rfpimp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# Load data
file = 'Admission_Predict_Ver1.1.csv'
df = pd.read_csv(file)
features = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP','LOR' , 'CGPA', 'Research','Chance' ]

# Train/tst split
df_train, df_test = train_test_split(df)
df_train = df_train[features]
df_test = df_test[features]

df_train = df_train[features]
df_test = df_test[features]

X_train, y_train = df_train.drop('Chance',axis=1), df_train['Chance']
X_test, y_test = df_test.drop('Chance',axis=1), df_test['Chance']

# Train
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train,y_train)

# Permutation feature importance
imp = rfpimp.importances(rf, X_test, y_test)

#Plot
fig, ax = plt.subplots()
ax.barh(imp.index, imp['Importance'],  facecolor='blue',  edgecolor='k')
ax.set_xlabel('Importance score')
ax.set_title('Permutation feature importance')
plt.gca().invert_yaxis()
fig.tight_layout()
plt.show()
```

Now we do linear regression with chance Vs CGPA a simple linear regression that can be ploted in 2d so we can star to understand better the model



[Chance vs CGPA](https://user-images.githubusercontent.com/83521233/117573960-3df37f00-b0b1-11eb-8569-a10cd20cd0d3.png)

CGPA is the most important feature regarding Chance of addimission but CGPA alone captured only 78% of variance of the data.

Code for chance vs CGPA


```python
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt


# Load data
file = Admission_Predict_Ver1.1.csv'
df = pd.read_csv(file)

X = df['CGPA'].values.reshape(-1,1)
Y = df['Chance'].values

# Train

adm = linear_model.LinearRegression()
model = adm.fit(X, Y)
response = model.predict(X)

# Evaluate

r2 = model.score(X, Y)

#Plot

plt.style.use('default')
plt.style.use('ggplot')

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(X, response, color='k', label='Regression model')
ax.scatter(X, Y, facecolor='blue', alpha=0.7, label='Sample data', marker='+' )
ax.set_ylabel('Chance of Admit', fontsize=14)
ax.set_xlabel('CGPA', fontsize=14)
ax.legend(facecolor='white', fontsize=11)
ax.set_title('$R^2= %.2f$' % r2, fontsize=15)
```

We can evaluate our model performance in a form of R-squared, with model.score(X, y). X is the features, and y is the response variable used to fit the model. model.score(X, y)
We can make future prediction using model.predict(x_pred),  
lm.predict[[8.87]] 
When your CGPA is 8.87 you have a chance of addmission of 78%.


Now we will use all the features we have.



```python
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Load data
file = pd.read_csv('Admission_Predict_Ver1.1.csv')
features = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP','LOR' , 'CGPA', 'Research', ]
target = 'Chance'

x = np.array(file[features])
y = np.array(file[target])

# Split the data into training/testing sets
X_train, X_test, Y_train, Y_test = train_test_split(x,y)

# Create linear model and train it
lm = linear_model.LinearRegression()
lm.fit(X_train, Y_train)

# Use model to predict on Test data
lmp = lm.predict(X_test)

# Intercept
print('Intercept_:', lm.intercept_)
# Coefficient
print('Coefficient:', lm.coef_)
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(Y_test, lmp))
```

figure results

We can see that we improve the results to 0.83, lets try without TOEFL SCORE.

figure Results

Note.value of individual regression coefficient may not be reliable under multicollinearity, it does not undermine the prediction power of the mode

TOFL score  variable dont improve our model, using her is simply add random noise. If we add it as a predictor to our model, we will most likely notice that our accuracy drops. This is because, while it is added information, it does not correlate well with the other information you have provided or dont have relation with the target(some cases add a feature can even drop the score).
We have ways to improve the model but for this model we stop here.



```python

```
