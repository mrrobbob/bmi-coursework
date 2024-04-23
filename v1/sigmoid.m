function x = sigmoid(z)
    x = 1.0 ./ (1.0 + exp(-z));
end