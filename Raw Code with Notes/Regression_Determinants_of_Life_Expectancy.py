# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 21:18:57 2018

@author: hmzmu
"""
import pandas as pd
import os
import numpy as np
import statsmodels.api as sm

# set working directory 
os.chdir('c:\\Users\hmzmu\desktop\python practice\data')

# import data
pd.read_csv('gapminder.tsv', sep='\t')
df = pd.read_csv('gapminder.tsv', sep='\t')

# inspect the data
df.head()
df.shape
df.columns
df.dtypes

# create column subsets
country_df = df['country']
subset = df[['country', 'continent', 'year']]
subset.head()
# you can also subset by using columns numbers
# you can also use the range function to select a range of columns

#subset rows, inspect row type
df.loc[0]
row_100 = df.loc[99]
type(row_100)

# df.shape gives [rows, columns], the bottom code selects rows
df.shape[0]
# find the last row by translating number of rows to index
df.loc[df.shape[0] - 1]

# the above code shows that the last row is 1703

# pull the row name from the row index
df.iloc[0]

# notice the difference between loc and iloc
df.head()

# specify multiple rows, using nested brackets

df.iloc[[0, 99, 99]]

# subset by rows and columns using df.iloc[row, column]

df.iloc[0, 1]

df.iloc[[0, 99, 999], [0, 1, 2]]

# preliminary analysis and exploration

df.groupby('year')['lifeExp'].mean()
df.groupby(['year', 'continent'])['lifeExp'].mean()

# since this is a series, a python object, it has plot methods associated with it
df.groupby('year')['lifeExp'].mean().plot()




# Regression analysis: determinants of life expectancy 
# Select explanatory variables: year, population, and gdpPerCap
# Select p-value: Our threshold will be significance at the 5% level
# We predict 'year' will not have a strong coefficient but will remain in the model
# Time may not directly affect life expectancy, but medical advancement over time does.
# We predict population will also have a weak coefficient but it will remain in the model 
# A larger country will likely have more economic resources to use for improving quality of life
# gdpPercap will have the strongest coefficient, we predict it will be in the positive direction. 
x = df.iloc[:, [2, 4, 5]]
y = df.iloc[:, 3]


y = y.apply(pd.to_numeric, errors='ignore')



from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/5, random_state = 0)

#check our model for multicollinearity 

x_test.corr()





# fitting simple linear regression to the training set

from sklearn.linear_model import LinearRegression

# predicting the test set results

regressor = LinearRegression()
regressor.fit(x_train, y_train)
regressor.coef_

y_pred = regressor.predict(x_test)

# Test our model 

np.append(arr = np.ones((1704, 1)).astype(int), values = x, axis = 1)
x_opt = x[:, [0, 1, 2, 3]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

# adjusted R-squared is only .11%. This model just barely explains 1% of the variation in life expectancy. 
# remove variable with the highest p-value (least significance)
# keep in mind we are aiming for significance at a p-value of .05
# our first model shows that variable in index 1 is least significant, that variable is 'year'

x_opt = x[:, [0, 2, 3]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

# after excluding the year varialbe, we find that the 'pop' variable doesn't meet the significance threshold
# this model explains only .09% of the variation in life expectancy.

x_opt = x[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

# the final model includes the remaining variable, 'gdpPercap'. The adj. R-squared remains at 0.09%
# the all three models exhibit auto-correlation in the positive direction. 
# none of our predictions of the coefficients were correct.
# based on the number of variables that could not meet the perscribed level of significance..
# coupled with the low adj. R value and evidence of autocorrelation, these models were not well specified. 
# the likely explanation for these low-performing models is omitted variable bias. 






