function [Ms, Mt, Mst, Mts] = constructMMD(ns,nt,Ys,Yt0,C, muu)
e = [1/ns*ones(ns,1);-1/nt*ones(nt,1)];
es = 1/ns*ones(ns,1);
et = -1/nt*ones(nt,1);

M =(e*e'*C);
Ms =(1-muu)*(es*es'*C);
Mt =(1-muu)*(et*et'*C);
Mst =(1-muu)*(es*et'*C);
Mts =(1-muu)*(et*es'*C);
if ~isempty(Yt0) && length(Yt0)==nt
    for c = reshape(unique(Ys),1,C)
        es = zeros(ns,1);
        et = zeros(nt,1);
        es(Ys==c) = 1/length(find(Ys==c));
        et(Yt0==c) = -1/length(find(Yt0==c));
        es(isinf(es)) = 0;
        et(isinf(et)) = 0;
        Ms =  Ms + muu*(es*es');
        Mt = Mt + muu*(et*et');
        Mst = Mst + muu*(es*et');
        Mts = Mts + muu*(et*es');
    end
end

Ms = Ms/norm(M,'fro');
Mt = Mt/norm(M,'fro');
Mst = Mst/norm(M,'fro');
Mts = Mts/norm(M,'fro');