function ACz = mytest(X,I1,n_clusters,lambda,rho)

% Generate data
begin = tic;
maxIter = 20;        % maximum iterative steps

DEBUG = 1;
n = size(X,2);
r = 1;  %n_space;
l = 1;

% figure
% colormap(gray)


%% Initial Z
In = eye(n);
Z.U = In(:,1);
Z.V = In(:,1);
Z.S = 0;
Z.Up = In(:,2:end);  %In(:,2:end);
Z.Vp = In(:,2:end);
Z.r = 0;
%Z.mat = Z.U*Z.S*Z.V'

E = zeros(size(X));
U = zeros(size(X));

    function f = cost(Z)
        Zmat = Z.U*Z.S*Z.V';
        f =  0.1*sum(diag(Z.S)) +0.5*rho*norm(X - X*Zmat - E + U, 'fro');%
    end
    function G = egrad(Z)
        % Same comment here about Xmat.
        Zmat = Z.U*Z.S*Z.V';
        G = - rho*X'*(X - X*Zmat - E + U);
    end

ACz = zeros(1, 10*n_clusters);
% This is the major part. We pursue the rank up to 10*n_clusters.
for i = 1:10*n_clusters %rankmax-1 face 10*n_cluster
    %This is to solve the LRR over the variety M_{<=k}. The objective
    %function is
    % 0.5|Z|^2_F + lambda |E|_{2,1} + 0.5 rho |X - X Z - E + U|^2_F
    % That is, we replace |Z|_* with the smooth term 0.5|Z|^2_F for
    % convenience
    %  -------------------------------------------------------------------------------
    [Z, E, U] = riemann_ladmp_lrr(X, Z, E, U, r, lambda, rho, maxIter, DEBUG);
    %   Z.S = diag(max(diag(Z.S)-0.01,0));
    % Display the result
    Zmat = Z.U * Z.S*Z.V';
    %     disp(['rankZ=' num2str(size(Z.S,1))]);
    
    %     imagesc(abs(Zmat)+abs(Zmat'))
    %     drawnow
    
    % doing rank pursuit
    r = r + l;
    problem.M = rank_variety_factory(n, n, r);
    problem.cost = @cost;
    problem.egrad = @egrad;
    negEgrad = - problem.egrad(Z);
    
    %   grad = problem.M.egrad2rgrad(Z, negEgrad);
    G = problem.M.proj(Z, negEgrad);
    %   % Get the initial estimate on the M_{<= k+l}
    
    Z = problem.M.retr(Z, G,1);
    
    % ------------------------------------------------------------------------------------------
    
    Zmatc =10000*Zmat.^2;
    % if i>=2*n_clusters %2 coil 6 extYaleB
    AC = zeros(1,30);
    
    for jj = 1:10
        %  scale_sig = 0.05;
        Zmatc =Zmatc./(repmat(sqrt(sum(Zmatc.^2)),[size(Zmatc,1),1])+1e-4);
        Zmatc = max(Zmatc-(jj-1)*0.01,0);+min(Zmatc+(jj-1)*0.01,0);
        
        [Zclusters2,NcutEigenvectors,NcutEigenvalues] = ncutW(0.5*(abs(Zmatc)+abs(Zmatc')),n_clusters);
        
        Iz = zeros(size(I1,1),1);
        for ii = 1:n_clusters
            idx = find(Zclusters2(:,ii)==1);
            Iz(idx) = ii;
        end
        bestIz = bestMap(I1,Iz);
        ACz(jj) = length(find(I1 == bestIz))/length(I1);
    end
    [ACw(i),rj] = max(ACz);
    AC(i) = max(ACw);
    [ACzmax,rank_max] = max(AC);
    disp(['ACzmax=' num2str(ACzmax) 'rank_max=' num2str(rank_max)]);
    if max(AC(i))==1
        break
    end
    
    if rem(i,10)==  0
        toc(begin)
    end
end
end

