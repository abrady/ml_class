function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
  a = .01;
  % J = computeCost(X, y, theta)
  
  % h(x^(i)) = x0^(i)*t0 + x1^(i)*t1 
  % J(t0,t1) = 1/2m*sum_i_1_to_m[(h(x^(i)) - y^(i))^2]
  %          = 1/2*(x0^1*t0 + x1^1*t1 + y^1)^2 + 1/2*(x0^2*t0 + x1^2*t1 = y^2)^2 + ... + 1/2*(x0^m*t0 + x1^m*t1 = y^m)^2
  % dJ/dt0   = 2/2*(x0^1*t0 + x1^1*t1 + y^1)*x0^1 + 2/2*(x0^2*t0 + x1^2*t1 + y^2)*x0^2 + ... + 2/2*(x0^m*t0 + x1^m*t1 + y^m)*x0^m
  %          = 1/m*(X*theta - y)*x0
  Err = (X*theta - y)/m;
  dtheta = X'*Err;
  theta -= a*dtheta;
%  computeCost(X,y,theta);
  % ============================================================
  % Save the cost J in every iteration    
  J_history(iter) = computeCost(X, y, theta);

end

end
