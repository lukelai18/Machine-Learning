import scipy.io
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


# Function to calculate the MLE for the Bernoulli distribution of each pixel

# Function to predict the digit using the trained naive Bayes models with Laplace smoothing
def predict_digit_with_smoothing(digit_mle, sample, alpha=1e-10):
    # Calculate the log probability for the sample to belong to each digit class
    # Add a small constant 'alpha' for Laplace smoothing to avoid log(0)
    log_probs = []
    for mle in digit_mle:
        # Add smoothing to the MLE estimates
        smoothed_mle = np.clip(mle, alpha, 1 - alpha)
        # Calculate the log likelihood
        log_likelihood = np.sum(np.log(smoothed_mle) * sample + np.log(1 - smoothed_mle) * (1 - sample))
        # Since the prior p(y) = 1/10 for each class, we can ignore it for the argmax decision
        log_probs.append(log_likelihood)
    # The predicted class is the one with the highest log probability
    return np.argmax(log_probs)

def calculate_mle(training_data):
    # Sum the training data (since it's binary, this gives us the count of '1's for each pixel)
    pixel_on_counts = np.sum(training_data, axis=0)
    # Divide by the total number of samples to get the probability of pixel 'on'
    mle_probabilities = pixel_on_counts / training_data.shape[0]
    return mle_probabilities

# Load the MATLAB file
mat = scipy.io.loadmat('digits.mat')

# Inspect the contents of the MATLAB file to understand the structure of the data
mat.keys()

# Initialize a list to hold the MLE for each digit
digit_mle = []

# Calculate the MLE for each digit and store the result
for digit in range(10):
    train_key = f'train{digit}'
    # Calculate the MLE for the current digit's training data
    mle = calculate_mle(mat[train_key])
    digit_mle.append(mle)

    # Visualize the MLE as a 28x28 image
    plt.imshow(mle.reshape((28, 28)), cmap='gray')
    plt.title(f'Model for digit {digit}')
    plt.colorbar()
    plt.show()

# Reset the lists for the actual and predicted labels
y_true = []
y_pred = []

# Predict the class for each sample in the test data using the smoothed model
for digit in range(10):
    test_key = f'test{digit}'
    test_data = mat[test_key]
    for sample in test_data:
        predicted_digit = predict_digit_with_smoothing(digit_mle, sample)
        y_pred.append(predicted_digit)
        y_true.append(digit)

# Recalculate the confusion matrix
cm_smoothed = confusion_matrix(y_true, y_pred)

# Visualize the new confusion matrix
plt.figure(figsize=(10, 10))
plt.imshow(cm_smoothed, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix with Smoothing')
plt.colorbar()
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.xticks(np.arange(10))
plt.yticks(np.arange(10))
plt.show()

# Return the new confusion matrix data
cm_smoothed