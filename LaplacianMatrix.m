function [L] = LaplacianMatrix(src_X,tar_X,k)
    options.p = k;             % keep default
    % options.sigma = 0.1;        % keep default
    % options.lambda = 10.0;      % keep default
    % options.gamma = 1.0;        % [0.1,10]
    % options.ker = 'rbf';        % 'rbf' | 'linear'
    X = [src_X,tar_X];
    % Data normalization
    X = X*diag(sparse(1./sqrt(sum(X.^2))));
    n = size(src_X,2);
    m = size(tar_X,2);
    nm = n+m;
    % Construct graph Laplacian
    manifold.k = options.p;
    manifold.Metric = 'Cosine';
    manifold.NeighborMode = 'KNN';
    manifold.WeightMode = 'Cosine';
   
%     W = graph(X',manifold);

    W = graphNew(X',manifold);
    Dw = diag(sparse(sqrt(1./sum(W))));
    L = speye(nm)-Dw*W*Dw;
end