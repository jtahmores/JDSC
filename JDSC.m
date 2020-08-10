function [Acc] = JDSC(Xs, Xt, Ys, Yt0, Yt, options)

alpha = options.alpha;
mu = options.mu;
beta = options.beta;
k = options.k;
T = options.T;
muu=options.muu;
m = size(Xs,1);
ns = size(Xs,2);
nt = size(Xt,2);

class = unique(Ys);
C = length(class);

% compute LDA
dim = size(Xs,1);
meanTotal = mean(Xs,2);

Sw = zeros(dim, dim);
Sb = zeros(dim, dim);
for i=1:C
    Xi = Xs(:,find(Ys==class(i)));
    meanClass = mean(Xi,2);
    Hi = eye(size(Xi,2))-1/(size(Xi,2))*ones(size(Xi,2),size(Xi,2));
    Sw = Sw + Xi*Hi*Xi'; % calculate within-class scatter
    Sb = Sb + size(Xi,2)*(meanClass-meanTotal)*(meanClass-meanTotal)'; % calculate between-class scatter
end
P = zeros(2*m,2*m);
P(1:m,1:m) = Sb;
Q = Sw;

for t = 1:T
    % Construct MMD matrix
    [Ms, Mt, Mst, Mts] = constructMMD(ns,nt,Ys,Yt0,C,muu);
    
    Ts = Xs*Ms*Xs';
    Tt = Xt*Mt*Xt';
    Tst = Xs*Mst*Xt';
    Tts = Xt*Mts*Xs';
    
    % Construct centering matrix
    Ht = eye(nt)-1/(nt)*ones(nt,nt);
    
    X = [zeros(m,ns) zeros(m,nt); zeros(m,ns) Xt];
    H = [zeros(ns,ns) zeros(ns,nt); zeros(nt,ns) Ht];
    
    Smax = mu*X*H*X'+beta*P;
    Smin = [Ts+alpha*eye(m)+beta*Q, Tst-alpha*eye(m) ; ...
        Tts-alpha*eye(m),  Tt+(alpha+mu)*eye(m)];
    [W,~] = eigs(Smax, Smin+1e-9*eye(2*m), k, 'LM');
    A = W(1:m, :);
    Att = W(m+1:end, :);
    
    Zs = A'*Xs;
    Zt = Att'*Xt;
    if T>1
        %---------------------------------------------------------------------
        Xss = double(Zs);
        Xtt = double(Zt);
        XX = [Xss,Xtt];
        n = size(Xss,2);
        mm = size(Xtt,2);
        YY = [];
        for c = 1 : C
            YY = [YY,Ys==c];
        end
        YY = [YY;zeros(mm,C)];
        XX = XX * diag(sparse(1 ./ sqrt(sum(XX.^2))));
        %% Construct graph Laplacian
        if options.rho > 0
            manifold.k = options.p;
            manifold.Metric = 'Cosine';
            manifold.NeighborMode = 'KNN';
            manifold.WeightMode = 'Cosine';
            W1 = lapgraph(XX',manifold);
            D = diag(sparse(sqrt(1 ./ sum(W1))));
            L = eye(n + mm) - D * W1 * D;
            
        else
            L = 0;
        end
        
        % Construct kernel
        K = kernel('rbf',XX,sqrt(sum(sum(XX .^ 2).^0.5)/(n + mm)));
        E = diag(sparse([ones(n,1);zeros(mm,1)]));
        Beta = ((E + options.rho * L )* K + options.eta * speye(n + mm,n + mm)) \ (E * YY);
        F = K * Beta;
        [~,Cls] = max(F,[],2);
        
        %% Compute accuracy
        Acc = numel(find(Cls(n+1:end)==Yt)) / mm;
        Yt0 = Cls(n+1:end);
        fprintf('Iteration:[%02d]>>mu=%.2f,Acc=%f\n',t,muu,Acc);
        muu = estimate_mu(Xss',Ys,Xtt',Yt0);
    end
end
end
function K = kernel(ker,X,sigma)
switch ker
    case 'linear'
        K = X' * X;
    case 'rbf'
        n1sq = sum(X.^2,1);
        n1 = size(X,2);
        D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*X'*X;
        K = exp(-D/(2*sigma^2));
    case 'sam'
        D = X'*X;
        K = exp(-acos(D).^2/(2*sigma^2));
    otherwise
        error(['Unsupported kernel ' ker])
end
end

