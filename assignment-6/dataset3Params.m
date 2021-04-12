function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%
tvals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

x1 = [1 2 1]; 
x2 = [0 4 -1]; 

row = 1;
lenT = length(tvals);
results = zeros(lenT*lenT, 3);

for n = 1:length(tvals)
    for m = 1:length(tvals)
        model= svmTrain(X, y, tvals(n), @(x1, x2) gaussianKernel(x1, x2, tvals(m))); 
        predictions = svmPredict(model, Xval);
        err_val = mean(double(predictions ~= yval));
        results(row, :) = [tvals(n), tvals(m), err_val];
        row = row + 1; 
    end    
end 

[val, i] = min(results(:, 3)); 
C = results(35, 1);
sigma = results(35, 2);


% =========================================================================

end
