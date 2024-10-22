import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix

# Load data
data = loadmat('digits.mat')

# Assuming that train0, train1, ..., train9 are the variables in digits.mat
X = np.vstack([data[f'train{i}'] for i in range(10)])
# Create labels for each class
Y = np.hstack([i * np.ones(data[f'train{i}'].shape[0]) for i in range(10)])
X_test = np.vstack([data[f'test{i}'] for i in range(10)])
Y_test = np.hstack([i * np.ones(data[f'test{i}'].shape[0]) for i in range(10)])


# Add a column of ones for the bias term
X = np.hstack((np.ones((X.shape[0], 1)), X))
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

# Define the objective function
def svm_objective(W, X, Y, C):
    n_samples = X.shape[0]
    n_classes = 10
    scores = X.dot(W)
    correct_class_scores = scores[np.arange(n_samples), Y.astype(int)].reshape(-1, 1)
    margins = np.maximum(0, scores - correct_class_scores + 1)
    margins[np.arange(n_samples), Y.astype(int)] = 0
    loss = np.sum(margins) / n_samples
    reg_loss = (1/2) * np.sum(W * W)
    return loss + C * reg_loss

# Define the gradient of the objective function
def svm_gradient(W, X, Y, C):
    n_samples = X.shape[0]
    n_features = X.shape[1]
    n_classes = 10
    scores = X.dot(W)
    correct_class_scores = scores[np.arange(n_samples), Y.astype(int)].reshape(-1, 1)
    margins = np.maximum(0, scores - correct_class_scores + 1)
    margins[np.arange(n_samples), Y.astype(int)] = 0
    binary = margins
    binary[margins > 0] = 1
    row_sum = np.sum(binary, axis=1)
    binary[np.arange(n_samples), Y.astype(int)] = -row_sum.T
    dW = np.dot(X.T, binary)
    dW /= n_samples
    dW += C * W
    return dW

# Gradient descent function
def gradient_descent(X, Y, C, r, T):
    n_features = X.shape[1]
    n_classes = 10
    W = np.zeros((n_features, n_classes))
    loss_history = []

    for t in range(T):
        loss = svm_objective(W, X, Y, C)
        grad = svm_gradient(W, X, Y, C)
        W -= r * grad
        loss_history.append(loss)

        print(f'Iteration {t+1}/{T}, Loss: {loss:.4f}')

    return W, loss_history

# Function to predict labels
def predict(W, X):
    W = W.reshape(X.shape[1], -1)
    scores = X.dot(W)
    return np.argmax(scores, axis=1)

# Experiment with different values of C, r, T
C_values = [0.01, 0.1, 1, 10, 100]
r = 0.005  # Start with a learning rate of 0.1 and adjust based on the results
T = 500  # Set a number of iterations and adjust based on the convergence
best_C = None
best_accuracy = 0
best_confusion_matrix = None

# Dictionary to hold the trained models for each C value
trained_models = {}

for C in C_values:
    W, loss_history = gradient_descent(X, Y, C, r, T)
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
    plt.plot(loss_history, label=f'C={C}')

plt.xlabel('Iteration number')
plt.ylabel('Loss value')
plt.title('SVM Objective Function Value Over Time')
plt.legend()
plt.show()

# Plot the confusion matrix
plt.matshow(best_confusion_matrix, cmap=plt.cm.gray)
plt.title(f'Confusion Matrix for C={best_C}')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

curW =trained_models[0.01]
# print("Rows:", trained_models[0.01].shape[0])
# print("Columns:", trained_models[0.01].shape[1])
# Remove the bias term from each weight vector
# Bias term is assumed to be the first element in each column
curW_no_bias = curW[1:, :]  # Now should be (784, 10)

# Reshape weights for visualization, each column into a 28x28 image
W_images = curW_no_bias.reshape(28, 28, 10)  # Reshape into 28x28 for each of the 10 classes

# Visualization
fig, axes = plt.subplots(2, 5, figsize=(10, 5))  # Adjust the size as needed
fig.suptitle('Visualizing SVM Weights for Each Digit Class')

for i, ax in enumerate(axes.flatten()):
    # Extract the i-th weight matrix and normalize it
    W_image = W_images[:, :, i]
    vmax = np.max(np.abs(W_image))  # Maximum absolute value for symmetric scaling
    img = ax.imshow(W_image, cmap='bwr', vmin=-vmax, vmax=vmax)
    ax.set_title(f'Digit {i}')
    ax.axis('off')

# Add a colorbar to the figure
fig.colorbar(img, ax=axes.ravel().tolist(), shrink=0.95)
plt.show()

