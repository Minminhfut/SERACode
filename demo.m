
srcStr = {'Caltech10', 'Caltech10', 'Caltech10', 'amazon', 'amazon', 'amazon', 'webcam', 'webcam', 'webcam', 'dslr', 'dslr', 'dslr'};
tgtStr = {'amazon', 'webcam', 'dslr', 'Caltech10', 'webcam', 'dslr', 'Caltech10', 'amazon', 'dslr', 'Caltech10', 'amazon', 'webcam'};
Result_Final = [];
        
for iData = 1:12
    src = char(srcStr{iData});
    tgt = char(tgtStr{iData});
    data = strcat(src, '_vs_', tgt);
    
    benchmark = pwd;
    addpath(genpath(benchmark));

    Datapath1= [benchmark,'/imagedata/',src '_SURF_L10.mat'];
    load(Datapath1);
    Xs = fts ./ repmat(sum(fts, 2), 1, size(fts, 2));
    src_X = Xs';
    src_labels = labels;
    parameter.size = size(src_labels,1);

    Datapath1= [benchmark,'/imagedata/',tgt '_SURF_L10.mat'];
    load(Datapath1);
    Xt = fts ./ repmat(sum(fts, 2), 1, size(fts,2));
    tar_X = Xt';
    tar_labels = labels;
    
    
    
    parameterAMDA.noises = 0.7;
    parameterAMDA.k = 10;
    parameterAMDA.gamma = 0.01;
    parameterAMDA.theda = 100;
    parameterAMDA.size = size(src_labels,1);

    parameterGRA.alpha = 100;
    parameterGRA.lambda = 5;
    parameterGRA.beta = 0.01;
    parameterGRA.k = 10;
    parameterGRA.size = size(src_labels,1);
    fprintf('data = %s\n', data);
    disp('Stage one: Enriching the knowledge of intra-domain features');
    parameter.rho = 0.001;
    [Ws,Wt] = Enrich_Intra_Domain(src_X,src_X,tar_X,tar_X,parameter);
    src_X = tanh(Ws*src_X);
    tar_X = tanh(Wt*tar_X);
    
    disp('Stage two: Extracting the knowledge of inter-domain features');
    disp('           1) AMDA');
    total = [src_X,tar_X];
    [AMDA_allhx, Ws] = AMDA(total,parameterAMDA);  
    
    disp('           2) GRA');
    [GRA_allhx] = GRA(AMDA_allhx,parameterGRA); 
        
    disp('           3) AMDA');
    parameterAMDA.noises = 0.7;
    [allhx, Ws] = AMDA(GRA_allhx,parameterAMDA); 

    xr=allhx(:,1:size(src_X,2));
    xr=xr';
    bestC = 1./mean(sum(xr.*xr,2));
    model = svmtrain(src_labels,xr,['-q -t 0 -c ',num2str(bestC),' -m 3000']);
    xe= allhx(:,size(src_X,2)+1:end);
    xe=xe';
    [label,accuracy] = svmpredict(tar_labels,xe,model);
    Result_Final = [Result_Final;accuracy(1)]
end
Result_Final






























% 
% srcStr = {'Caltech10', 'Caltech10', 'Caltech10', 'amazon', 'amazon', 'amazon', 'webcam', 'webcam', 'webcam', 'dslr', 'dslr', 'dslr'};
% tgtStr = {'amazon', 'webcam', 'dslr', 'Caltech10', 'webcam', 'dslr', 'Caltech10', 'amazon', 'dslr', 'Caltech10', 'amazon', 'webcam'};
% Result = [];
%         
% % parameter.alpha = 150;
% % parameter.lambda = 0.001;
% % parameter.beta = 0.001;
% % parameter.noise = 0.7;
% % parameter.k = 10;
% % parameter.layer = 1;
% 
% for iData = 1:1
% 
%     disp(num2str(iData));
%     src = char(srcStr{iData});
%     tgt = char(tgtStr{iData});
%     data = strcat(src, '_vs_', tgt);
% 
%     benchmark = pwd;
%     addpath(genpath(benchmark));
% 
%     Datapath1= [benchmark,'/imagedata/',src '_SURF_L10.mat'];
%     load(Datapath1);
%     Xs = fts ./ repmat(sum(fts, 2), 1, size(fts, 2));
%     src_X = Xs';
%     src_labels = labels;
%     parameter.size = size(src_labels,1);
% 
%     Datapath1= [benchmark,'/imagedata/',tgt '_SURF_L10.mat'];
%     load(Datapath1);
%     Xt = fts ./ repmat(sum(fts, 2), 1, size(fts,2));
%     tar_X = Xt';
%     tar_labels = labels;
% 
%     fprintf('data=%s\n', data);
%     
%     
% parameter.noises = 0.7;
% parameter.layers = 1;
% parameter.lambda = 1e-5;
% % parameter.beta = (size(src_labels,2)+size(tar_labels,2))*4;
% parameter.beta = 10000;
% parameter.gamma = 0.01;
% parameter.size = size(src_labels,1);
% 
% parameterManifold.alpha = 100;
% parameterManifold.lambda = 20;
% parameterManifold.beta = 0.01;
% parameterManifold.size = size(src_labels,1);
%    parame.beta = 1;
%     disp('Compute...');
% %     src_X = double(src_X>0);
% %     tar_X = double(tar_X>0);
%     [Ws,Wt] = FCA(src_X,src_X,tar_X,tar_X,parame);
% % % %  [Ws,Wt] = FCAL21(src_X,src_X,tar_X,tar_X,parame);
%     src_X = tanh(Ws*src_X);
%     tar_X = tanh(Wt*tar_X);
%     disp('finish...');
%     total = [src_X,tar_X];
%     nbsrc=size(src_X,2);
%     nbtgt=size(tar_X,2);
%     
%     
% %     disp('Computer MMD...')
%     parameter.MMD=[(1/nbsrc^2)*ones(nbsrc,nbsrc), -1/(nbsrc*nbtgt)*ones(nbsrc,nbtgt); -1/(nbsrc*nbtgt)*ones(nbtgt,nbsrc), (1/nbtgt^2)*ones(nbtgt,nbtgt)];
%    
% 
%         parameter.layers = 1;
%     [allhx, Ws] = mSDA(total,parameter);  
% %     [allhx, Ws] = mSDA(allhx,parameter);  
% %     [allhx, Ws] = mSDA(allhx,parameter);  
% 
% %         [allhx, D_cell, W_cell] = myRepresentationLearningManifold(allhx,1,parameterManifold); 
%         %     disp('mDA finish..');
%         [allhx, D_cell, W_cell] = myRepresentationLearningManifold(allhx,1,parameterManifold); 
% 
% %     parameter.noises = 0.7;
% 
% 
% %      [allhx, Ws] = mSDA(allhx2,parameter); 
% 
% %     parameter.noises = 0.7;
% %  
%      [allhx, Ws] = mSDA(allhx,parameter); 
% %     
% %      [allhx, Ws] = mSDA(allhx2,parameter); 
% 
%     xr=allhx(:,1:size(src_X,2));
%     xr=xr';
%     bestC = 1./mean(sum(xr.*xr,2));
%     model = svmtrain(src_labels,xr,['-q -t 0 -c ',num2str(bestC),' -m 3000']);
%     xe= allhx(:,size(src_X,2)+1:end);
%     xe=xe';
%     [label,accuracy] = svmpredict(tar_labels,xe,model);
%     accuracy
% %     Result = [Result;accuracy(1)]
% 
% % source = ones(size(src_X,2),1);
% % target = -1*ones(size(tar_X,2),1);
% % 
% % xr=allhx;
% % % xr=allhx(:,1:size(src_X,2));
% % label=[source;target];
% % % xr=src_X;
% % xr=xr';
% % bestC = 1./mean(sum(xr.*xr,2));
% % model2 = svmtrain(label,xr,['-q -t 0 -c ',num2str(bestC),' -m 3000']);
% % [label2,accuracy] = svmpredict(label,xr,model2);
% %     Result = [Result;accuracy(1)]
% 
% 
% %     total = [src_X,tar_X];
% %     [allhx, Ws] = mSDA(total, parameter.noise,1);
% % 
% %     [allhx, D_cell, W_cell] = myRepresentationLearningM(allhx,parameter);
% %      xr=[src_X; allhx(:,1:size(src_X,2))];
% %     xr=xr';
% %     bestC = 1./mean(sum(xr.*xr,2));
% %     model = svmtrain(src_labels,xr,['-q -t 0 -c ',num2str(bestC),' -m 3000']);
% %     xe=[tar_X; allhx(:,size(src_X,2)+1:end)];
% %     xe=xe';
% %     [label,accuracy] = svmpredict(tar_labels,xe,model);
% % 
% %     accuracy(1)
% %     Result = [Result; accuracy(1)];
%     fprintf('\n');
% end
% Result
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
