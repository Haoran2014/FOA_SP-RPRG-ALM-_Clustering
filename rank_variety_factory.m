function M = rank_variety_factory(m, n, k)
% Manifold of m-by-n matrices X such that rank(X) <= k.

% function M = rank_variety_factory(m, n, k)
% Based on the geometry from
% T. P. Cason, P.-A. Absil, and P. Van Dooren, "Iter- ative methods for low 
%  rank approximation of graph similarity matrices," Linear Algebra Appl., 
%  vol. 438, pp. 1863?1882, 2013.
% R.SchneiderandA.Uschmajew,"Convergenceresults for projected line-search 
%   methods on varieties of low-rank matrices via Lojasiewicz inequality,"
%   ArXiv e- print, arXiv:1402.5284, 2014.

% Original author: Junbin Gao, 30 May 2015.
% Contributors:  Junbin Gao, 30 May 2015
% Change log:
    
    M.name = @() sprintf('%dx%d matrices with rank less than or equal to %d',m, n, k);
    
    M.dim = @() (m+n-k)*k;  % this dimension is for the manifold of rank = k
                            % I have not 
                            % found any formula for the dimension of this 
                            % variety. 
    
    % Fisher metric as proposed
    % We assume the tangent vector on the tangent cone is defined by
    % xi.M, xi.Up, xi.Vp, xi.TS, xi.TU and xi.TV, i.e.,
    % xi = xi.
    M.inner = @(x, d1, d2) d1.A(:).'*d2.A(:) + d1.B(:).'*d2.B(:) ...
                                             + d1.C(:).'*d2.C(:) ...
                                             + d1.D(:).'*d2.D(:); 
    M.norm = @(X, eta) sqrt(M.inner(X, eta, eta));
    
    M.dist = @(x, y) error('rank_varietyfactory.dist not implemented yet.');
    
    %M.typicaldist = @() m*pi/2; % Not so sure
    
    M.egrad2rgrad = @egrad2rgrad;
    function rgrad = egrad2rgrad(X, egrad)
        PU = X.U * X.U';
        PV = X.V * X.V';
        PpU = eye(m) - PU;
        PpV = eye(n) - PV;
        rgrad.A = X.U' * egrad * X.V;
        rgrad.B = X.U' * egrad * PpV * X.Vp;
        rgrad.C = X.Up' * PpU * egrad * X.V;      
        if X.r == k
            rgrad.D = zeros(m-k, n-k);
        elseif X.r < k
            prj_grad = PU * egrad * PV + PpU * egrad * PV + PU *egrad * PpV;
            err = egrad - prj_grad;
            t = k - X.r;
            [tmpU, tmpS, tmpV] = svd(err, 'econ');
            Xi_ks = tmpU(:,1:t) * tmpS(1:t,1:t) * tmpV(:,1:t)';
            rgrad.D = X.Up' * Xi_ks * X.Vp;
        else
           error('The point X is not on the variety, over the rank!') 
        end
    end
    
    M.ehess2rhess = @ehess2rhess;
    function rhess = ehess2rhess(X, egrad, ehess, eta)
        error('rank_varietyfactory.ehess2rhess not implemented yet.')
    end
    
    % Projection of the vector eta onto the tangent space
    M.proj = @projection;
    function etaproj = projection(X, eta)
        PU = X.U * X.U';
        PV = X.V * X.V';
        PpU = eye(m) - PU;
        PpV = eye(n) - PV;
        etaproj.A = X.U' * eta * X.V;
        etaproj.B = X.U' * eta * PpV * X.Vp;
        etaproj.C = X.Up' * PpU * eta * X.V;      
        if X.r == k
            etaproj.D = zeros(m-k, n-k);
        elseif X.r < k
            prj_grad = PU * eta * PV + PpU * eta * PV + PU * eta * PpV;
            err = eta - prj_grad;
            t = k - X.r;
            [tmpU, tmpS, tmpV] = svd(err, 'econ');
            Xi_ks = tmpU(:,1:t) * tmpS(1:t,1:t) * tmpV(:,1:t)';
            etaproj.D = X.Up' * Xi_ks * X.Vp;
        else
           error('The point X is not on the variety, over the rank!') 
        end
    end
    
    M.tangent = M.proj;
    M.tangent2ambient = @(X, eta) eta;
    
    M.retr = @retraction;
    function Y = retraction(X, eta, t)
        if nargin < 3
            t = 1.0;
        end
        % Although there is a quick way to find the retraction, here I use
        % the simple way. This is inefficient for large matrix
        
        % Work out X + t*eta
        dir = X.U * ( X.S + t*eta.A) * X.V' + t*X.Up*eta.C*X.V' + ...
               t*X.U*eta.B*X.Vp' + t*X.Up * eta.D * X.Vp';
           
        % We use projection onto M_{<=k} as the retraction. 
        [U, S, V] = svd(dir);
        r = nnz(diag(S));
        if r >= k
        Y.r = k;
        r= k;
        Y.U = U(:,1:r);
        Y.S = S(1:r, 1:r);
        Y.V = V(:,1:r);
        Y.Up = U(:,r+1:m);
        Y.Vp = V(:,r+1:n);
        elseif r>0
        Y.r = r;
        Y.U = U(:,1:r);
        Y.S = S(1:r, 1:r);
        Y.V = V(:,1:r);
        Y.Up = U(:,r+1:m);
        Y.Vp = V(:,r+1:n);
        elseif r== 0
        Y.r =0;
        Y.U = U(:,1);
        Y.S = S(1, 1);
        Y.V = V(:,1);
        Y.Up = U(:,r+1:m);
        Y.Vp = V(:,r+1:n); 
        end
       
    end
    
    M.exp = @exponential;
    function Y = exponential(X, eta, t)
        if nargin < 3
            t = 1.0;
        end
        Y = retraction(X, eta, t);
        warning('manopt:rank_variety:exp', ...
            ['Exponential for Rank variety' ...
            'not implemented yet. Used retraction instead.']);
    end
    
    M.hash = @(X) ['z' hashmd5(X.U(:))];
    
    M.rand = @random;
    function X = random()
        % A random point in the Embedding space
        X = rand(m, n);
        s = randi([1,k],1);
        [U, S, V] = svd(X);
        X.U = U(:,1:s);
        X.V = V(:,1:s);
        X.S = S(1:s, 1:s);
        X.Up = U(:,s+1:m);
        X.Vp = V(:,s+1:n);
    end
    
    M.randvec = @randomvec;
    function eta = randomvec(X)
        % A random vector in the Tangent space
        eta = randn(m, n);
        eta = projection(X, eta); 
    end
    
    M.lincomb = @lincomb;
    
    M.zerovec = @(X) struct('U', [eye(k); zeros(m-k,k)], 'S', zeros(k, k), ...
                'V', [eye(k); zeros(n-k,k)], 'Up', [zeros(k, m-k); eye(m-k)], ...
                'Vp', [zeros(k, n-k); eye(n-k)]);
    M.transp = @transport;
    function y = transport(x1, x2, d)
        tmp = x1.U * d.A * x1.V' + x1.Up * d.C * x1.V' + x1.U * d.B * x1.Vp' + x1.Up * d.D * x1.Vp';
        y = projection(x2, tmp);
    end
    % vec and mat are not isometries, because of the unusual inner metric.
    M.vec = @(X, U) [U.A(:); U.B(:); U.C(:); U.D(:)]; 
    M.mat = @(X, u) struct('A', reshape(u(1:(X.r*X.r)), X.r, X.r), ...
        'B', reshape(u((X.r*X.r+1): X.r*X.r + X.r*(n-X.r)), X.r, n-X.r), ...
        'C', reshape(u((X.r*X.r + X.r*(n-X.r) + 1):X.r*X.r + X.r*(n-X.r) + (m-X.r)*X.r), m-X.r, X.r), ...
        'D', reshape(u(X.r*X.r + X.r*(n-X.r) + (m-X.r)*X.r+1:end), m-X.r, n-X.r));
    M.vecmatareisometries = @() false;
end




% Linear combination of tangent vectors
function d = lincomb(X, a1, d1, a2, d2) %#ok<INUSL>
    if nargin == 3
        d.A = a1*d1.A;
        d.B = a1*d1.B;
        d.C = a1*d1.C;
        d.D = a1*d1.D;
    elseif nargin == 5
        d.A = a1*d1.A + a2*d2.A;
        d.B = a1*d1.B + a2*d2.B;
        d.C = a1*d1.C + a2*d2.C;
        d.D = a1*d1.D + a2*d2.D; 
    else
        error('Bad use of  Rank_variety.lincomb.');
    end
end





