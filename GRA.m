function [hx] = GRA(xx,parameterGRA)
    % xx : dxn input
    xx = full(xx);
    W = [];
    [d, n] = size(xx);
    t = var(xx');
    index = t > 0.000001;
    xx = xx(index,:);

    H = eye(n) - ones(n,n)/n;
    xxc = xx*H;

    s = sum(abs(xxc),2);
    index = ( s > 1e-9 );
    xxc = xxc(index,:);
    xx = xx(index,:);

    C = xxc*xxc';  %%%%%%
    [W, D] = get_W_l21_loss(xxc',diag(C),parameterGRA); % l21 loss
    if sum(sum(isnan(W))) > 0
        error('W is wrong' );
    end
    W = full(W);
    hx = W'*xx;
    hx = tanh(hx*parameterGRA.alpha);
end

function [W, G1] = get_W_l21_loss(X,A,parameterGRA)
   % min |XW - X|_{2,1} + lambda trace(W'*A*W)
    [n, d] = size(X);
    G1 = ones(n,1); 
    G2 = ones(n,1); 

    Y = X';
    Newsrc_X = Y(:,1:parameterGRA.size);
    Newtar_X = Y(:,1+parameterGRA.size:end);
    parameterGRA.L = LaplacianMatrixE(Newsrc_X,Newtar_X,parameterGRA.k);
    [U_w,S_w] = eig(full(parameterGRA.L));
    E=real(S_w^(0.5)*U_w'*X);
    max_iter = 20; % ususllay converge in less than 10 iterations
    obj_old = 1e9;
    for i = 1:max_iter
        W = update_W(X,A, E,G1, G2,parameterGRA);  
        G1 = update_G(X-X*W);
        G2 = update_G(E*W);
        obj_new = get_obj(X,A,W,E,parameterGRA);
        assert(obj_new <= obj_old, ['old: ' num2str(obj_old) ', new: ' num2str(obj_new)]);
        if obj_old - obj_new < obj_new * 1e-5
            disp(['  GRA converged in ' num2str(i) ' iterations']);
            break;
        end
        obj_old = obj_new;
    end
end

function W = update_W(X,DIAG,E,G1,G2,parameterGRA)
    % C = X'*X;
    C = X'* bsxfun(@times,X,G1);
    C2 = E'* bsxfun(@times,E,G2);
    W = inv(C + parameterGRA.lambda*diag(DIAG)+ parameterGRA.beta*C2) * C;
end

function obj = get_obj(X,A,W,E,parameterGRA)
    % A is vector
    R = X - X*W;
    obj1 = sum(sqrt(sum(R.*R, 2)));
    obj2 = sum(sum( W.*bsxfun(@times,W,A)));
    R2 = E*W;
    obj3 = sum(sqrt(sum(R2.*R2, 2)));
    % obj3 = trace(W'*X'*parameter.L*X*W);
    % obj2 = sum(sqrt(sum(W.*W,2)));
    obj = obj1 + parameterGRA.lambda*obj2 + parameterGRA.beta*obj3;
end

function G = update_G(R)
    s = sqrt(sum(R.*R,2));
    s = s + 1e-7;
    index = (s>0);
    % disp(['num of zero: ' num2str( length(index) -  sum(index))]);
    G = zeros(length(s),1);
    G(index) = 1./s(index);
    G = G/2;
end


