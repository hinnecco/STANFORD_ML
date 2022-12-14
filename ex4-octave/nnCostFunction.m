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
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

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


X = [ones(m,1), X];
z = X*Theta1';
a2 = sigmoid(z);
a2 = [ones(size(a2,1),1), a2];
z2 = a2*Theta2';
a3 = sigmoid(z2);

y_bool = (1:num_labels)==y;

J = (1/m)*sum(sum(-(y_bool.*(log(a3)))-((1-y_bool).*(log(1-a3)))));
Theta1_reg = Theta1(:,2:end);
Theta2_reg = Theta2(:,2:end);

reg = (lambda/(2*m))*(sum(sum(Theta1_reg.^2))+sum(sum(Theta2_reg.^2)));

J = J + reg;

%Backpropagation
#Primeiro devemos calcular o feedforward
zforward2 = X*Theta1';
action2 = sigmoid(zforward2);
action2 = [ones(size(action2,1),1), action2];
zforward3 = action2*Theta2';
action3 = sigmoid(zforward3);
h_forward = action3;

%tranformando y em um vetor de booleanos
y_results = (1:num_labels)==y;

%calculando os deltas
delta3 = h_forward-y_results;
delta2 = (delta3*Theta2).*[ones(size(zforward2,1),1) sigmoidGradient(zforward2)];
delta2 = delta2(:,2:end);

%calculando os gradientes
Theta1_grad = (1/m)*(delta2'*X);
Theta2_grad = (1/m)*(delta3'*action2);

%Calculando o regulariza??o para os gradientes

Grad_reg1 = (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
Grad_reg2 = (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];

%adicionando a regulariza??o
Theta1_grad = Theta1_grad+Grad_reg1;
Theta2_grad = Theta2_grad+Grad_reg2; 





% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
