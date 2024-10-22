# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from scipy.io import loadmat
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_absolute_error

Xtrain = loadmat('Xtrain.mat')['Xtrain']
Ytrain = loadmat('Ytrain.mat')['Ytrain']

poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(Xtrain)

model_ls = LinearRegression()
# Ytrain is the value we try to predict
model_ls.fit(X_poly, Ytrain)

# Fit the Least Squares regression model
# Since we need to include an intercept term, we add a column of ones to our feature matrix
X_poly_with_intercept = np.hstack([np.ones((X_poly.shape[0], 1)), X_poly])
coeffs_ls = np.linalg.lstsq(X_poly_with_intercept, Ytrain, rcond=None)[0]

# Set up and solve the Lasso problem using linear programming
# We need to minimize the sum of absolute residuals |y - Xw| + lambda*|w|
# This can be transformed into a linear programming problem
n_features = X_poly.shape[1]
n_samples = Ytrain.size

# The objective function coefficients (for w and slack variables)
c = np.concatenate([np.ones(n_samples), np.repeat(0.01, n_features)])

# The inequality constraints matrix
A_ub = np.block([
    [-np.eye(n_samples), -X_poly],
    [-np.eye(n_samples), X_poly]
])

# The inequality constraints vector
b_ub = np.concatenate([-Ytrain, Ytrain])

# The bounds for each variable in the solution vector
x0_bounds = (None, None)
x_bounds = [(x0_bounds)] * (n_samples + n_features)

# Solve the linear programming problem
res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=x_bounds, method='highs')

# Extract the coefficients for the Lasso regression
coeffs_lasso = res.x[n_samples:]


X_plot = np.linspace(Xtrain.min(), Xtrain.max(), 100).reshape(-1, 1)
x_plot_ploy = poly.transform(X_plot)


y_plot_ls = model_ls.predict(x_plot_ploy)
y_plot_lasso = x_plot_ploy.dot(coeffs_lasso)  # For Lass

# mae_ls = mean_absolute_error(Ytrain, y_plot_ls)
# mae_lasso = mean_absolute_error(Ytrain, y_plot_lasso)
#
# print('Mean Absolute Error of Least Squares fit: ', mae_ls)
# print('Mean Absolute Error of Lasso fit: ', mae_ls)

plt.scatter(Xtrain, Ytrain, label='Training data')
plt.plot(X_plot, y_plot_ls, label='Least Squares fit Polynomial')
plt.plot(X_plot, y_plot_lasso, label='Lasso fit Polynomial')
# Annotating the plot with the error metrics
plt.legend()
plt.show()