function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

m = size(X, 1);
num_labels = size(Theta2, 1);

p = zeros(m, 1);

X = [ones(m, 1) X];

layer2Linear = Theta1 * X';
layer2Features = sigmoid(layer2Linear);
layer2Features = [ones(1, size(layer2Features, 2)); layer2Features];

outputLayerLinear = Theta2 * layer2Features;
outputLayerPredictions = sigmoid(outputLayerLinear);

[maxPredictionValues, predictionClasses] = max(outputLayerPredictions);
p = predictionClasses;

end
