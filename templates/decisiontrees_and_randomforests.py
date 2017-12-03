from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

regressor = DecisionTreeRegressor()
regressor.fit(X, y)
regressor.predict(y_dash)


# for random forests, we will need to mention the number of
# trees we want to take the average over
regressor = RandomForestRegressor(n_estimators=k)
regressor.fit(X, y)
regressor.predict(y_dash)
