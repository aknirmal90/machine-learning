import pandas as pd
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt


df = pd.read_csv(filename)
# do_preprocessing_operations()

X_train, X_test, y_train, y_test

regressor = LinearRegression()

# this works for any number of independent variables in X
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


# visualizing results for LR with only one Independent variable
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X_test), color='blue')
# this does a scatter plot of original values and plots
# also draws a straight line between independent variable and the
# straight line fit by the predictor. gives an estimate of good the prediction model is



#########################################################################################
# in case multiple independent variables exist, it is a good idea to reduce their number
# and retain only vars with significant prediction power
#########################################################################################
import statsmodels.formula.api as sm
cols = [0:-1]
ols_fit = sm.OLS(X[cols], y)
ols.summary()

# chose a threshold p_value (say 0.05), any IV whose p_value > 0.05 could be dropped
# use backward elimination to remove IVs with negligible prediction power
# i) ignore the column with highest p_value, and generate ols_fit after ignoring this column
# ii) remove next highest p_value IV (as long as p_value > 5%) and generate another ols_fit

# repeat i) and ii) until none of the IVs have p_value > 5%
# i) after this, fit your linear regression over only the reduced IV set
