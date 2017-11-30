# very similar to linear regression, only difference is we need to generate a
# dataset which also contains polynomial raised values of IVs

# if X, y = input dataset
from sklearn.preprocessing import PolynomialFeatures
converter = PolynomialFeature(degree=4)
X_new = converter.fit_transform(X)

# X_new can now be used to the standard LinearRegression object in sklearn.linear_model
# follow steps in linear_processing.py
