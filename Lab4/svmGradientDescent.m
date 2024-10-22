function w = svmGradientDescent(dataFile, C, r, T)
    % Load the dataset
    data = load(dataFile);
    
    % Concatenate all training examples and labels
    X = [];
    Y = [];
    for i = 0:9
        variableName = sprintf('train%d', i);
        X = [X; data.(variableName)];
        Y = [Y; i * ones(size(data.(variableName), 1), 1)];
    end
    % Transpose X to have examples as columns
    X = X'; % Now X is 784x5000
    % Add a row of 1s for the bias term
    X = [ones(1, size(X, 2)); X];
    
    % Initialize the weights randomly
    w = randn(size(X, 1), 10); % For 10 classes
    
    % Run gradient descent
    for t = 1:T
        % Evaluate the gradient at the current w
        [~, gradE] = evaluateSVM(C, w, X, Y);
        
        % Update the weights
        w = w - r * gradE;
        
        % Optionally, print out the objective function value to monitor progress
        [E, ~] = evaluateSVM(C, w, X, Y);
        fprintf('Iteration %d, Objective Function Value: %f\n', t, E);
    end
end

