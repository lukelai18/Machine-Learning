% Main entry
load Xtrain;
load Ytrain;
load Xtest;
load Ytest;

D=10;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% question 1

d_x = linspace(1,D,D)
TrainLoss = zeros(D,1);
TestLoss = zeros(D,1);
for i=1:D
    w_d = LinearRegPolySquare(Xtrain, Ytrain, i);
    y_pred = Predictor(w_d, Xtrain, i);
    TrainLoss(i) = SquareLoss(y_pred, Ytrain);
    y_test = Predictor(w_d, Xtest, i);
    TestLoss(i) = SquareLoss(y_test, Ytest);
end

% p=plot(d_x, TrainLoss, d_x, TestLoss)
% p(1).Marker='*';
% p(2).Marker='o';
% legend('training loss','test loss')
% xlabel('degree');
% ylabel('loss');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% question 2
d = 10
W = LinearRegPolySquare(Xtrain, Ytrain, d)
Ypred = Predictor(W, Xtrain, d)
Loss = SquareLoss(Ytrain, Ypred)
 scatter(Xtrain, Ytrain,50,'MarkerEdgeColor',[0 .5 .5],...
              'MarkerFaceColor',[0 .7 .7],...
              'LineWidth',1.5)
 hold on
 scatter(Xtrain, Ypred,50)
 legend('training data','prediction result')
 xlabel('input');
 ylabel('output');
 title('Degree 10 data prediction result.')
 hold off
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% question 3
% d = 10
% W = LinearRegPolySquare(Xtrain, Ytrain, d)
% Ypred = Predictor(W, Xtrain, d)
% Loss = SquareLoss(Ytrain, Ypred)
% scatter(Xtrain, Ytrain)
% hold on
% scatter(Xtrain, Ypred)
% hold off
% 

    