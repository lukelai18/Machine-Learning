
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

Xtrain = loadmat('Xtrain.mat')['Xtrain']
Ytrain = loadmat('Ytrain.mat')['Ytrain']
Xtest = loadmat('Xtest.mat')['Xtest']
Ytest = loadmat('Ytest.mat')['Ytest']

def DesignMatrix(X, degree):
    return np.hstack([X**i for i in range(degree+1)])

def GetLoss(y, y_prep):
    return ((y - y_prep) ** 2).mean()

train_loss = []
test_loss = []
w = np.array([])
w3 = np.array([])
w10 = np.array([])

for degree in range(1, 11):
    trainMatrix = DesignMatrix(Xtrain, degree)
    testMatrix = DesignMatrix(Xtest, degree)

    # Try to solve (AT*A)w = AT*b, to get the coefficient w
    w= np.linalg.solve(trainMatrix.T @ trainMatrix, trainMatrix.T @ Ytrain)

    if degree == 3:
        w3 = w
    elif degree == 10:
        w10 = w

    YtrainPred = trainMatrix @ w
    YtestPred = testMatrix @ w

    train_loss.append(GetLoss(Ytrain, YtrainPred))
    test_loss.append(GetLoss(Ytest, YtestPred))

# # For part(a), begin to plot the training and test loss
# plt.plot(range(1, 11), train_loss, label="Training Loss")
# plt.plot(range(1, 11), test_loss, label="Test Loss")
# plt.xlabel('Polynomial Degree')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# for part(b): Calculate weights w for degree 3 using the training data
# degree3 = 3
# # Generate a dense set of x values for plotting the polynomial
# x_dense = np.linspace(Xtrain.min(), Xtrain.max(), 1000)[:, np.newaxis]  # Reshape for compatibility
# A_dense = DesignMatrix(x_dense, degree3)
# # Evaluate the polynomial on the dense set of x values
# y_dense = A_dense @ w3
#
# # Plot the training data
# plt.scatter(Xtrain, Ytrain, color='blue', label='Training Data')
# plt.plot(x_dense, y_dense, color='red', label=f'Degree {degree3} Polynomial')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Training Data and Estimated Degree 3 Polynomial')
# plt.legend()
# plt.show()

# for part(c): Calculate weights w for degree 3 using the training data
degree10 = 10
# Change it into 2-dimensional array, so that DesignMatrix can handle it
x_dense10 = np.linspace(Xtrain.min(), Xtrain.max(), 1000)[:, np.newaxis]  # Reshape for compatibility
A_dense10 = DesignMatrix(x_dense10, degree10)
y_dense10 = A_dense10 @ w10

plt.scatter(Xtrain, Ytrain, color='blue', label='Training Data')
plt.plot(x_dense10, y_dense10, color='red', label=f'Degree {degree10} Polynomial')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Training Data and Estimated Degree 10 Polynomial')
plt.legend()
plt.show()