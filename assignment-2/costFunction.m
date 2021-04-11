function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
    
    grad = grad + [(sigm-y(i))*X(i, 1); (sigm-y(i))*X(i, 2);...
        (sigm-y(i))*X(i, 3);];
    
end

J = J/m;
grad = grad/m; 

%
% Note: grad should have the same dimensions as theta
%








% =============================================================

end
