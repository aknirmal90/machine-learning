from sklearn.preprocessing import StandardScaler
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
