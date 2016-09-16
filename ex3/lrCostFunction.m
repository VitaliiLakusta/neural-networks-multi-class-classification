function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

% %Compute cost function
sigmoidVector = sigmoid(X * theta);
J = (1.0 / m) * ...
    sum(-y .* log(sigmoidVector) - (1 .- y) .* log(1.0 - sigmoidVector)) + ...
    (lambda / (2.0 * m)) * sum(theta(2:end) .^ 2); % regularization term %


%Compute gradient for theta
accuracyDifferenceVector = sigmoidVector - y;
grad = (1.0 / m) * (X' * accuracyDifferenceVector);

thetaWithZeroBias = theta;
thetaWithZeroBias(1) = 0;
grad = grad + (lambda / m) * thetaWithZeroBias; % regularization term %

end
