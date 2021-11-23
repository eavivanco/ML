function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 30;
sigma = 0.03;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

current_error = 3000;
posible_values = [0, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]; 

for posible_C = posible_values
    for posible_sigma = posible_values
        model =  svmTrain(X, y, posible_C, @(x1, x2) gaussianKernel(x1, x2, posible_sigma));
		pred = svmPredict(model , Xval);
		error = mean(double(pred ~= yval));
        if error <= current_error
            current_error = error;
            C = posible_C;
            sigma = posible_sigma;
        end
    end 
end 
% =========================================================================

end
