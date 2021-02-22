function [gamma_best, lambda_best] = selectModel_KRLS_CompleteCrossValidation(predictors, responses, lambda_v, gamma_v, verbose)
%
% Finds the best gamma and lambda hyperparameters from those given in
% lambda_v and gamma_v vectors for the model.
% Uses complete k-fold cross validation.
%
% Inputs:
% > predictors                  - training set
% > responses                   - categories binded to training set 
% > lambda_v                    - vector of lambda hyperparameter values
% > gamma_v                     - vector of gamma hyperparameter values
% > verbose                     - true/false
%
% Outputs:
% > gamma_best                  - gamma that gives the best error
% > lambda_best                 - lambda that gives the best error

    n = length(responses);
    nl = round(0.7 * n);
	err_best = +Inf;
    for gamma = gamma_v
        for lambda = lambda_v
            err = 0;
            for mc = 1:10
                i = randperm(n);
                il = i(1:nl);
                iv = i(nl + 1:end);
                model = trainModel_KRLS_MulticlassClassification(predictors(il,:), responses(il), lambda, gamma);
                predictedOutcome = applyModel_KRLS_MulticlassClassification(model, predictors(iv,:));
                err = err + mean(predictedOutcome ~= responses(iv));
            end
            err = err / 10;
            if (err_best > err)
                err_best = err;
                gamma_best = gamma;
                lambda_best = lambda;
            end
            if (verbose)
                fprintf('Gamma: %.e Lambda: %.e Error: %e \tBest Gamma: %.e Best Lambda: %.e Best Error: %e\n',gamma, lambda, err, gamma_best, lambda_best, err_best);
            end
        end
    end
end