
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
