clear all;
clc
str_domains = {'Caltech10', 'amazon', 'webcam', 'dslr'};

fid = fopen(strcat('test','.txt'),'wt');

options.rho =0.9;
options.p = 5;
options.eta = 0.2;
options.T = 10;
options.knn=10;
options.muu = 1.0;
options.k = 40;
options.alpha=1.0;
options.mu = 0.9;
options.beta = 0.1;

acc2=[];
fprintf(fid,'rho  : \t %f\n',options.rho);
fprintf(fid,'p  : \t %f\n',options.p);
fprintf(fid,'eta  : \t %f\n',options.eta);
fprintf(fid,'alpha  : \t %f\n',options.alpha);
fprintf(fid,'mu  : \t %f\n',options.mu);
fprintf(fid,'beta  : \t %f\n',options.beta);
fprintf(fid,'k  : \t %f\n',options.k);
fprintf(fid,'muu  : \t %f\n',options.muu);

for i = 1 : 4
    for j = 1 : 4
        if i == j
            continue;
        end
        src = char(str_domains{i});
        tgt = char(str_domains{j});
        options.data = strcat(src,'-vs-',tgt);
        fprintf('Data=%s \n',options.data);
        load(['data1/office/' src '_SURF_L10.mat']);     % source domain
        fts = fts ./ repmat(sum(fts,2),1,size(fts,2));
        X1 = zscore(fts); clear fts
        X1 = normr(X1)';
        Y1 = labels;           clear labels
        
        load(['data1/office/' tgt '_SURF_L10.mat']);     % target domain
        fts = fts ./ repmat(sum(fts,2),1,size(fts,2));
        X2 = zscore(fts); clear fts
        X2 = normr(X2)';
        Y2 = labels;            clear labels
        
        knn_model = fitcknn(X1',Y1,'NumNeighbors',1);
        Cls = knn_model.predict(X2');
        Yt0 = Cls;
        [Acc] = JDSC(X1,X2,Y1,Yt0,Y2,options);
        fprintf('%s --> %s: %.2f accuracy \n\n', src, tgt, Acc * 100);
        fprintf(fid,'%s --> %s: %.2f accuracy \n', src, tgt, Acc * 100);
        acc2=[acc2;Acc*100]
        
    end
end
A=mean(acc2);
fprintf(fid,'mean: %.2f  \n\n', A);
