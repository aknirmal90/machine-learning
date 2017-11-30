import pandas as pd
from sklean.preprocessing import Imputer

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
from sklean.preprocessing import LabelEncoder, OneHotEncoder
# creates an imputer object which will help remove null values by using the mean values of records across axis
labelencoder = LabelEncoder()
X[:, 1] = labelencoder.fit_transform(X[:, 1])
y = labelencoder.fit_transform(y)

# creates dummy variables
onehotencoder = OneHotEncoder(categorical_values=[0])
X = onehotencoder.fit_transform(X[:,1]).toarray()


########################################################################
# scaling numerical values
########################################################################
from sklean.preprocessing import StandardScaler
# scales numerical fields to a range between -1 and 1.
# this prevents columns with large numbers from being the dominant component
# of model being fit (since most final params of a model are decided by minizining
# euclidean distance from mean for each column)
scaler = StandardScaler()
X = scaler.fit_transform(X)


########################################################################
# split dataset into training and test sets
########################################################################
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
