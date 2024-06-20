import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, explained_variance_score, r2_score
from sklearn.preprocessing import PolynomialFeatures

input_file = 'data_multivar_regr.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

num_training = int(0.8 * len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

polynomial = PolynomialFeatures(degree=10)
X_train_transformed = polynomial.fit_transform(X_train)
X_test_transformed = polynomial.transform(X_test)

poly_linear_model = LinearRegression()
poly_linear_model.fit(X_train_transformed, y_train)

y_test_pred = poly_linear_model.predict(X_test_transformed)

print("Polynomial regression performance:")
print("Mean absolute error =", round(mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error =", round(mean_squared_error(y_test, y_test_pred), 2))
print("Median absolute error =", round(median_absolute_error(y_test, y_test_pred), 2))
print("Explained variance score =", round(explained_variance_score(y_test, y_test_pred), 2))
print("R2 score =", round(r2_score(y_test, y_test_pred), 2))

datapoint = [[7.75, 6.35, 5.56]]
poly_datapoint = polynomial.transform(datapoint)
prediction = poly_linear_model.predict(poly_datapoint)

print("\nPolynomial regression prediction for new datapoint:")
print("Predicted value:", prediction[0])
