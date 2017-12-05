import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.cross_validation import train_test_split

df = pd.read_csv(filename)
X = df.iloc[:, :-1].values  # create an np.array of independent variables
y = df.iloc[:, -1].values  # create an np.array of dependent variables

########################################################################
# replace missing values in dataset
########################################################################
# creates an imputer object which will help remove null values by using the mean values of records across axis
imputer = Imputer(missing_values='NaN', strategy='mean',  axis=0)

# tells the imputer which columns have missing values and then replaces missing values
imputer.fit(X[:, [2,3,4])
X[:, [2,3,4]] = imputer.transform(X[:, [2,3,4]])


########################################################################
# replace categorical variables with numerical labels
########################################################################
# transforms categorical variables into labels which are then transposed into
# to create dummy variables
labelencoder = LabelEncoder()
X[:, 1] = labelencoder.fit_transform(X[:, 1])
y = labelencoder.fit_transform(y)

# creates dummy variables
onehotencoder = OneHotEncoder(categorical_values=[0])
X = onehotencoder.fit_transform(X[:,1]).toarray()


########################################################################
# scaling numerical values
########################################################################
# scales numerical fields to a range between -1 and 1.
# this prevents columns with large numbers from being the dominant component
# of model being fit (since most final params of a model are decided by minizining
# euclidean distance from mean for each column)
scaler = StandardScaler()
X = scaler.fit_transform(X)


########################################################################
# split dataset into training and test sets
########################################################################

X_train, X_test, y_train, y_test = train_test_split(X, y)



########################################################################
# LINEAR REGRESSION
########################################################################
from sklearn.linear_model import LinearRegression
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


########################################################################
# POLYNOMIAL REGRESSION
########################################################################
from sklearn.preprocessing import PolynomialFeatures
converter = PolynomialFeature(degree=4)
X_new = converter.fit_transform(X)
X_new_train, X_new_test, y_train, y_test = train_test_split(X_new, y)

regressor = LinearRegression()
regressor.fit(X_new_train, y_train)
y_pred = regressor.predict(X_new_test)


########################################################################
# SUPPORT VECTOR REGRESSION
########################################################################
from sklearn.smv import SVR

sv_X = StandardScaler().fit(X)
sv_y = StandardScaler().fit(y)

X = sv_X.transform(X)
y = sv_X.transform(y)

# support vector regression needs scaling of numerical values to ensure
# accurate results
regressor = SVR(kernel='rbf')
regressor.fit(X, y)

# dont forget to scale the value of `y` for the prediction
y_pred = regressor.predict(sv_x.transform(np.array([[x_dash]])))
result = sy_y.inverse_transform(y_pred)

########################################################################
# DECISION TREES
########################################################################
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor()
regressor.fit(X_test, y_test)
regressor.predict(X_test)


########################################################################
# RANDOM FORESTS
########################################################################
# for random forests, we will need to mention the number of
# trees we want to take the average over
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=k)
regressor.fit(X_train, y_train)
regressor.predict(X_test)
