function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


a1 = [ones(m,1) X]; %incluindo coluna Bias em X
  
z2 = a1 * Theta1';  %definindo o valor de Z
a2 = sigmoid(z2);   %calculando o valor com a função de ativação
 
a2 =  [ones(size(a2,1),1) a2];  %incluindo coluna Bias em a2
  
z3 = a2 * Theta2';  %definindo o valor de z para submeter a função sigmoid
a3 = sigmoid(z3);  %calculando o valor do output-layer

[prob, p] = max(a3,[],2); %utilizando a função max para obter o index da classe de maior probabilidade.

% =========================================================================


end
