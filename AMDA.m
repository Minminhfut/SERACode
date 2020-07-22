function [hx, W] = AMDA(xx,parameterAMDA)
% xx : dxn input
% noise: corruption level
% lambda: regularization

% hx: dxn hidden representation
% W: dx(d+1) mapping
[d, n] = size(xx);
% adding bias
xxb = [xx; ones(1, n)];
% scatter matrix S
S = xxb*xxb';
% corruption vector
q = ones(d+1, 1)*(1-parameterAMDA.noises);
q(end) = 1;

% Q: (d+1)x(d+1)
Q = S.*(q*q');
Q(1:d+2:end) = q.*diag(S);
% P: dx(d+1)
P = S(1:end-1,:).*repmat(q', d, 1);
nbsrc=parameterAMDA.size;
nbtgt=size(xx,2)-nbsrc;
parameterAMDA.MMD=[(1/nbsrc^2)*ones(nbsrc,nbsrc), -1/(nbsrc*nbtgt)*ones(nbsrc,nbtgt); -1/(nbsrc*nbtgt)*ones(nbtgt,nbsrc), (1/nbtgt^2)*ones(nbtgt,nbtgt)];
    
MMD = xxb*parameterAMDA.MMD*xxb';
Qm = MMD.*(q*q');
Qm(1:d+2:end) = q.*diag(MMD);
MMD = Qm;

Newsrc_X = xxb(:,1:parameterAMDA.size);
Newtar_X = xxb(:,1+parameterAMDA.size:end);
parameterAMDA.L = LaplacianMatrix(Newsrc_X,Newtar_X,parameterAMDA.k);
Manifold = xxb*parameterAMDA.L*xxb';
Ql = Manifold.*(q*q');
Ql(1:d+2:end) = q.*diag(Manifold);
Manifold = Ql;

% final W = P*Q^-1, dx(d+1);
lambda = 1E-5;
reg = lambda*eye(d+1);
reg(end,end) = 0;
W = P/(Q+reg +parameterAMDA.theda*MMD +parameterAMDA.gamma*Manifold);
hx = W*xxb;
hx = tanh(hx);


