import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

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


from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

########################################################################
# visualization for 2-D X
########################################################################

# For Visualising the dataset and prediction results
def _get_pyplot_object(X, y, classifier, title, xlabel, ylabel):
    X_set, y_set = X, y
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.75, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green'))(i), label = j)
    plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)    
    return plt


########################################################################
# logistic regression
########################################################################
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)

########################################################################
# K Nearest Neighbour
########################################################################
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_2 = knn.predict(X_test)

########################################################################
# support vector classifer - linear kernel
########################################################################
from sklearn.svm import SVC
svc = SVC(kernel='linear')
svc.fit(X_train, y_train)
y_pred_3 = svc.predict(X_test)

########################################################################
# support vector classifer - gaussian kernel
########################################################################
from sklearn.svm import SVC
svc_gauss = SVC(kernel='rbf')
svc_gauss.fit(X_train, y_train)
y_pred_4 = svc_gauss.predict(X_test)


########################################################################
# naive bayes classifer
########################################################################
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_5 = nb.predict(X_test)


########################################################################
# decision tree regressor
########################################################################
from sklearn.tree import DecisionTreeClassifier
dt_class = DecisionTreeClassifier(criterion='entropy')
dt_class.fit(X_train, y_train)
y_pred_6 = dt_class.predict(X_test)


########################################################################
# random forest regressor
########################################################################
from sklearn.ensemble import RandomForestClassifier
rf_class = RandomForestClassifier(n_estimators=10)
rf_class.fit(X_train, y_train)
y_pred_7 = rf_class.predict(X_test)
