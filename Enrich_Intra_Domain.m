function [Ws,Wt] = Enrich_Intra_Domain(Sshare,Sspecial,Tshare,Tspecial,parameter)
% Sshare: Ncxd
% Sspecial:Nsxd
% Tshare:Ncxd
% Tspecial:Ntxd
% parameter
    [Nc,d] = size(Sshare);
    Ws = inv(Sshare*Sshare' + parameter.rho*diag(ones(Nc,1))) *(Sshare*Sspecial');
    Wt = inv(Tshare*Tshare' + parameter.rho*diag(ones(Nc,1))) *(Tshare*Tspecial');
    Ws = Ws';
    Wt = Wt';
end
