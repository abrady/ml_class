function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
# z =    [1.0000000   2.2873553   0.8908079;
#    1.0000000   2.4717267  -0.6861101;
#    1.0000000   0.3836040  -1.6322217]

for i=1:size(z,1)
  for j=1:size(z,2)
    g(i,j) = 1/(1+e^-z(i,j));
  end
end

% =============================================================

end
