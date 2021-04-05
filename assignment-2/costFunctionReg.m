function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

for i=1:m
    sigm = sigmoid(theta.'*X(i, :).'); 
    p1 = -y(i)*log(sigm);
    p2 = (1-y(i))*log(1-sigm);
    J = J + (p1 - p2);
    
    grad = grad + (X(i, :)*(sigm-y(i))).'; 
end

reg = 0;

for i=2:length(theta)
    reg = reg + theta(i).^2;
end

J = J/m + reg*lambda/(2*m);

grad = grad/m + [0;theta(2:end).*(lambda/m)]; 




% =============================================================

end
