load('digits.mat');
% Example initialization, the dimensions should match your dataset
X = features; % Where 'features' is the variable name in digits.mat
Y = labels;   % Where 'labels' is the variable name in digits.mat

numFeatures = size(X, 1); % Assuming X is the data matrix loaded from digits.mat
numClasses = 10; % Assuming we have 10 classes for digit classification
w = zeros(numFeatures, numClasses); % Initialize w to zeros or small random values
C = 1; % Example value, this might need tuning
[E, gradE] = evaluateSVM(w, X, Y, C);
alpha = 0.01; % Learning rate
maxIter = 1000; % Maximum number of iterations

for iter = 1:maxIter
    w = w - alpha * gradE; % Update rule
    [E, gradE] = evaluateSVM(w, X, Y, C); % Recompute objective and gradient
    fprintf('Iteration %d: E = %f\n', iter, E);
end
