clear all;
clc

fid = fopen(strcat('test','.txt'),'wt');

options.rho = 0.9;
options.p = 2;
options.eta = 0.01;
options.T = 10;
options.knn = 10;
options.muu = 1.0;
options.k = 100;
options.alpha= 0.4;
options.mu = 1.0;
options.beta = 0.01;

fprintf(fid,'rho  : \t %f\n',options.rho);
fprintf(fid,'p  : \t %f\n',options.p);
fprintf(fid,'knn  : \t %f\n',options.knn);
fprintf(fid,'eta  : \t %f\n',options.eta);
fprintf(fid,'alpha  : \t %f\n',options.alpha);
fprintf(fid,'mu  : \t %f\n',options.mu);
fprintf(fid,'beta  : \t %f\n',options.beta);
fprintf(fid,'k  : \t %f\n',options.k);
fprintf(fid,'muu  : \t %f\n',options.muu);

acc2=[];

for dataStr = {'USPS_vs_MNIST','MNIST_vs_USPS'}
    
    % Preprocess data using L2-norm
    data = strcat(char(dataStr));
    options.data = data;
    
    load(strcat('data1/digit/',data));
    X_src = X_src*diag(sparse(1./sqrt(sum(X_src.^2))));
    X_tar = X_tar*diag(sparse(1./sqrt(sum(X_tar.^2))));
    X1 = normc(X_src);
    X2 = normc(X_tar);
    Y1=Y_src;              clear Y_src
    Y2=Y_tar;              clear Y_tar
    
    knn_model = fitcknn(X1',Y1,'NumNeighbors',1);
    Cls = knn_model.predict(X2');
    Yt0 = Cls;
    [Acc] = JDSC(X1,X2,Y1,Yt0,Y2,options);
    fprintf('%s : %.2f accuracy \n\n', options.data, Acc * 100);
    fprintf(fid,'%s: %.2f accuracy \n',options.data, Acc * 100);
    acc2=[acc2;Acc*100];
    
end

A=mean(acc2);
fprintf(fid,'mean: %.2f  \n\n', A);