function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

# map the 400 features to 25 nodes
X1 = [ones(size(X, 1), 1) X];
z1 = X1*Theta1';
a1 = sigmoid(z1); # necessary?

# map from 26 to 10
X2 = [ones(size(a1, 1), 1) a1];
z2 = X2*Theta2';
a2 = sigmoid(z2);

 
# a2 contains the guesses for 5000 samples. e.g. row 1 contains:
# 0.00  0.00  0.00  0.00  0.01  0.00  0.01  0.00  0.01  1.00
# i.e. the best guess is ten, which is correct
# grab the max element for each row, 
# and use it to build a matrix where there is a 1 for that column
# if that is the guess, and zero otherwise.
# [_,K] = max(a2,[],2);

# for each answer, make a 10 element row with 1 in the answer position
# where the index of 10 is 10 and 1 is 1.
# note:
#   sub2ind creates a single index for a row/column pair
#   so for rows 1 to size(K,1), Y(row,K(row)) = 1
Y = zeros(size(a2));
Y(sub2ind(size(Y),1:size(Y,1),y')) = 1; 

# Y's row one, 1000, 2000, 3000 look like this: 
# 0   0   0   0   0   0   0   0   0   1 <= 10, correct (training answer 10 placed in elt 0)
# 1   0   0   0   0   0   0   0   0   0 <= 1, correct
# 0   0   1   0   0   0   0   0   0   0 <= 3, correct
# 0   0   0   0   1   0   0   0   0   0 <= 4, correct

# recode y to be a 10 element vector with a 1 at the
# index of the correct answer

H = Y.*log(a2) + (1 - Y).*log(1 - a2);
ksum = H*ones(size(H,2),1);
J = -1/m*ksum'*ones(size(ksum,1),1)

# regularize
# - first column of each is the bias, don't regularize it by convention
T1 = Theta1(:,2:end)(:); # note A(:) puts entire matrix in single column
T2 = Theta2(:,2:end)(:); # and sum sums columns (below)
J += lambda/(2*m)*(sum(T1.*T1) + sum(T2.*T2))


% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
