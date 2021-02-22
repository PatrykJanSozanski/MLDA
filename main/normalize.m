function XN = normalize(X)
%
% Normalizes arrays to 0-1 range.
%
% Inputs:
% > X                       - array to be normalized
%   
% Outputs:
% > XN                     - normalized array

    for i = 1:size(X, 2)
        mi = min(X(:, i));
        ma = max(X(:, i));
        di = ma - mi;
        X(:, i) = (X(:, i) - mi) / di;
    end
    XN = X;
end