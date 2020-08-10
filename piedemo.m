
clear all;
clc
str_domains = {'PIE05', 'PIE07', 'PIE09', 'PIE27','PIE29'};

fid = fopen(strcat('test','.txt'),'wt');

acc2=[];
options.rho =0.9;
options.p = 2;
options.eta = 0.01;
options.T = 10;
options.knn=10;
options.muu =1.0;
options.k = 200;         
options.alpha= 0.9;  
options.mu =0.1; 
options.beta = 0.01; 
fprintf(fid,'rho  : \t %f\n',options.rho);
fprintf(fid,'p  : \t %f\n',options.p);
fprintf(fid,'eta  : \t %f\n',options.eta);
fprintf(fid,'alpha  : \t %f\n',options.alpha);
fprintf(fid,'mu  : \t %f\n',options.mu);
fprintf(fid,'beta  : \t %f\n',options.beta);
fprintf(fid,'k  : \t %f\n',options.k);
fprintf(fid,'muu  : \t %f\n',options.muu);
for i = 1 : 5
    for j = 1 : 5
        if i == j
            continue;
        end
        src = char(str_domains{i});
        tgt = char(str_domains{j});
        options.data = strcat(src,'-vs-',tgt);
        fprintf('Data=%s \n',options.data);
        load(['data1/pie/' src '.mat']); % source domain
        fea = fea*diag(sparse(1./sqrt(sum(fea.^2))));
        X1 = zscore(fea,1); clear fea
        X1 = normc(X1)';
        Y1 = gnd;           clear gnd
        
        load(['data1/pie/' tgt '.mat']);     % target domain
        fea = fea*diag(sparse(1./sqrt(sum(fea.^2))));
        X2 = zscore(fea,1); clear fea
        X2 = normc(X2)';
        Y2 = gnd;            clear gnd
        
        
        knn_model = fitcknn(X1',Y1,'NumNeighbors',1);
        Cls = knn_model.predict(X2');
        Yt0 = Cls;
        [Acc] = JDSC(X1,X2,Y1,Yt0,Y2,options);
        fprintf('%s --> %s: %.2f accuracy \n\n', src, tgt, Acc * 100);
        fprintf(fid,'%s --> %s: %.2f accuracy \n', src, tgt, Acc * 100);
        acc2=[acc2;Acc*100]; 
    end
end
A=mean(acc2);
fprintf(fid,'mean: %.2f  \n\n', A);
