function predictedOutcome = applyModel_KRLS_MulticlassClassification(model, predictors)
%
% Applies the given model on the given data and calculates prediction.
%
% Inputs:
% > model                       - trained model to be used
% > predictors                  - test set
%
% Outputs:
% > predictedOutcome            - calculated predictions

    predictedOutcome = [];
    for i = 1:length(model)
        tmp = exp(-model(i).gamma * pdist2(predictors, model(i).X)) * model(i).alpha;
        tmp(tmp > 0) = model(i).p;
        tmp(tmp < 0) = model(i).m;
        predictedOutcome = [predictedOutcome, tmp]; %#ok<AGROW>
    end
    predictedOutcome = mode(predictedOutcome, 2);
end