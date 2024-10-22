function [E, gradE] = evaluateSVM(C, w)
    % Load the dataset
    data = load('digits.mat');
    
    % Initialize feature matrix X and label vector Y
    X = [];
    Y = [];
    
    % Number of training examples per class
    numExamples = 500; 
    
    % Load data from train0 to train9 and concatenate
    for i = 0:9
        % Construct the variable name for the current class
        variableName = sprintf('train%d', i);
        
        % Concatenate the feature matrix
        X = [X; data.(variableName)];
        
        % Concatenate the labels
        Y = [Y; i * ones(numExamples, 1)];
    end
    
    % Transpose X to have examples as columns
    X = X';
    
    
    % Initialize the objective value and gradient
    E = 0;
    gradE = zeros(size(w));
    
    % Number of classes and number of examples
    k = size(w, 2);
    n = size(X, 2);

    % Regularization term
    for j = 1:k
        E = E + (1/2) * norm(w(:, j))^2;
    end

    % Loss term
    for i = 1:n
        xi = X(:, i);
        yi = Y(i);
        
        % Find the class with the maximum score not equal to yi
        scores = w' * xi;
        scores(yi) = -inf;  % Exclude the true class
        [max_score, ~] = max(scores);
        
        % Hinge loss for the current example
        margin = w(:, yi)' * xi - max_score - 1;
        loss_i = max(0, -margin);
        
        % Accumulate the loss
        E = E + C * loss_i;
        
        % Gradient computation
        if margin < 0
            for j = 1:k
                if j == yi
                    gradE(:, j) = gradE(:, j) - C * xi;
                else
                    if scores(j) == max_score  % The class that contributed to the loss
                        gradE(:, j) = gradE(:, j) + C * xi;
                    end
                end
            end
        end
    end
    
    % Add the regularization term to the gradient
    gradE = gradE + w;
    
end

