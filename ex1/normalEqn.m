function [theta] = normalEqn(X, y)
%NORMALEQN Computes the closed-form solution to linear regression 
%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the code to compute the closed form solution
%               to linear regression and put the result in theta.
%

% ---------------------- Sample Solution ----------------------

% dJ/dt = 0 (for all t)
% X'*(X*theta - y)/m = 0
% X'X*t = X'*y
% t = pinv(X'X)X'*y

XtX = X'*X
XtX_inv = pinv(XtX)
theta = XtX_inv*X'*y
% computeCost(X,y,theta)
% -------------------------------------------------------------


% ============================================================

end
