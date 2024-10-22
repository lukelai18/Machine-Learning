import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Load the data from 'digits.mat'
data = loadmat('digits.mat')

# Preprocess training and testing data
X_train = np.vstack([data[f'train{i}'] for i in range(10)])
Y_train = np.hstack([i * np.ones(data[f'train{i}'].shape[0]) for i in range(10)])
X_test = np.vstack([data[f'test{i}'] for i in range(10)])
Y_test = np.hstack([i * np.ones(data[f'test{i}'].shape[0]) for i in range(10)])

# Add a column of ones for the bias term
X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

# Define the SVM objective function and gradient
def svm_objective_grad(W, X, Y, C):
    W = W.reshape(X.shape[1], -1)
    n_samples = X.shape[0]
    scores = X.dot(W)
    correct_class_scores = scores[np.arange(n_samples), Y.astype(int)].reshape(-1, 1)
    margins = np.maximum(0, scores - correct_class_scores + 1)
    margins[np.arange(n_samples), Y.astype(int)] = 0
    loss = np.mean(np.sum(margins, axis=1)) + 0.5 * C * np.sum(W * W)
    binary = margins > 0
    binary[np.arange(n_samples), Y.astype(int)] = -binary.sum(axis=1)
    grad = X.T.dot(binary) / n_samples + C * W
    return loss, grad.ravel()

# Define the gradient descent algorithm
def gradient_descent(X, Y, C, r, T):
    W = np.zeros(X.shape[1] * 10)
    for t in range(T):
        loss, grad = svm_objective_grad(W, X, Y, C)
        W -= r * grad
    return W

# Function to predict labels
def predict(W, X):
    W = W.reshape(X.shape[1], -1)
    scores = X.dot(W)
    return np.argmax(scores, axis=1)

# Define the range of C values and train the model
C_values = [0.01, 0.1, 1, 10, 100]
learning_rate = 1e-3
iterations = 100
best_C = None
best_accuracy = 0
best_confusion_matrix = None

# Dictionary to hold the trained models for each C value
trained_models = {}

# Train and evaluate the models
for C in C_values:
    W = gradient_descent(X_train, Y_train, C, learning_rate, iterations)
    predictions = predict(W, X_test)
    accuracy = np.mean(predictions == Y_test)
    print(f"C={C}, Test Accuracy={accuracy}")

    # Update the best model if current model is better
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_C = C
        best_confusion_matrix = confusion_matrix(Y_test, predictions)

    # Store the trained model
    trained_models[C] = W

# Display the best C value
print(f"The best value of C is {best_C} with accuracy {best_accuracy}.")
print("Confusion Matrix for the best C value:")
print(best_confusion_matrix)

# Plot the confusion matrix
plt.matshow(best_confusion_matrix, cmap=plt.cm.gray)
plt.title(f'Confusion Matrix for C={best_C}')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()