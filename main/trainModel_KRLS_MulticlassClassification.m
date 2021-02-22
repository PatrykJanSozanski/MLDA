function model = trainModel_KRLS_MulticlassClassification(predictors, responses, lambda, gamma)
%
% Creates the KRLS model with the given lambda and gamma hyperparameters and
% trains it on given data.
%
% Inputs:
% > predictors                  - training set
% > responses                   - categories binded to training set 
% > lambda                      - lambda hyperparameter
% > gamma                       - gamma hyperparameter
%
% Outputs:
% > model                       - trained model 

    c = max(responses);
    j = 0;
    for c1 = 1:c
        for c2 = c1 + 1:c
            j = j + 1;
            flag = responses == c1 | responses == c2; 
            XP = predictors(flag,:);
            YP = responses(flag);
            YP(YP == c1) = -1;
            YP(YP == c2) = +1;
            Q = exp(-gamma * pdist2(XP, XP));
            alpha = (Q + lambda * eye(size(Q))) \ YP;
            model(j).X = XP;        %#ok<AGROW>
            model(j).alpha = alpha; %#ok<AGROW>
            model(j).m = c1;        %#ok<AGROW>
            model(j).p = c2;        %#ok<AGROW>
            model(j).gamma = gamma; %#ok<AGROW>
        end
    end
end