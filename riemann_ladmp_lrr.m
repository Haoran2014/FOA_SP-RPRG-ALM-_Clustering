function [Z,E, U] = riemann_ladmp_lrr(X, Z, E, U, r, lambda, rho, maxIter, DEBUG)
% This matlab code implements linearized ADM method for LRR problem
%------------------------------
% min |Z|^2_F+lambda*|E|_2,1
% s.t., X = XZ+E
%--------------------------------
% inputs:
%        X -- D*N data matrix
% outputs:
%        Z -- N*N representation matrix
%        E -- D*N sparse error matrix
%        relChgs --- relative changes
%        recErrs --- reconstruction errors
%
% created by Risheng Liu on 05/02/2011, rsliu0705@gmail.com
%
clear global;
global M;% M=Z_k+X'*(X-X*Z_k-E_{k+1}+Y/mu_k)/eta.

%addpath PROPACK;

if (~exist('DEBUG','var'))
    DEBUG = 0;
end
if nargin < 7
    rho = 1.9;
end
if nargin < 6
    lambda = 0.1;
end

normfX = norm(X,'fro');
tol1 = 1e-4;%threshold for the error in constraint
tol2 = 1e-5;%threshold for the change in the solutions
[d n] = size(X);
opt.tol = tol2;%precision for computing the partial SVD
opt.p0 = ones(n,1);

%max_mu = 1e10;
%norm2X = norm(X,2);
% mu = 1e2*tol2;
%mu = min(d,n)*tol2;

%eta = norm2X*norm2X*1.02;%eta needs to be larger than ||X||_2^2, but need not be too large.

 

%% Start main loop

iter = 0;
%%---------------------------------------------------------------
% if DEBUG
%     disp(['initial,rank(Z)=' num2str(rank(Z.U * Z.S*Z.V'))]);
% end

function f = cost(Z)
     Zmat = Z.U*Z.S*Z.V';
     f = 0.5*rho*norm(X - X*Zmat - E + U, 'fro');%0.1*sum(diag(Z.S)) +
end

function G = egrad(Z)
    % Same comment here about Xmat.
    Zmat = Z.U*Z.S*Z.V';
    G =  - rho*X'*(X - X*Zmat - E + U);
end

problem.M = rank_variety_factory(n, n, r);
problem.cost = @cost;
problem.egrad = @egrad;
 


Zmat = Z.U*Z.S*Z.V';

while iter<maxIter
    iter = iter + 1;
    %copy U, E and Z to compute the change in the solutions
    Ek = E; 
    Uk = U;
    Zmatk = Zmat;     
    %Solve the problem on the manifold starting from the current point Z
    %Please note that here we are using newly updated objective function
   
  
    [Z, zcost, info, options] =FISTA_manfold(problem, Z);
  
   % [Z, zcost, info, options] = steepestdescent_fast(problem, Z);
    Zmat = Z.U*Z.S*Z.V';
    XZ = X*Zmat; 
      
    %Solve for E
    %E = solve_l1(X - XZ + U,lambda, rho);  % in this case, please use
                                            % smaller lambda = 0.001
    E = solve_l1l2(X - XZ + U,lambda/rho);
   %  E = solve_l1(X - XZ + U,lambda, rho);
     
  
    %Update U
    dU = X - XZ - E;
    
    % If we don't want to U, then comment the line below
    U = U + dU;   
    
    diffZ = norm(Zmatk - Zmat,'fro');
    relChgZ = diffZ/normfX;
    relChgE = norm(E - Ek,'fro')/normfX;
    relChg = max(relChgZ,relChgE);
    recErr = norm(dU,'fro')/normfX;
    convergenced = recErr <tol1 && relChg < tol2;
    if DEBUG
        if iter==1 || mod(iter,5)==0 || convergenced
%             disp(['iter ' num2str(iter) ',rank(Z)=' num2str(rank(Zmat)) ',relChg=' num2str(max(relChgZ,relChgE))...
%                 ',recErr=' num2str(recErr)]);
%            imagesc(abs(Zmat)+abs(Zmat'))
            drawnow
        end
    end
    if convergenced
%    if recErr <tol1 & mu*max(relChgZ,relChgE) < tol2 %this is the correct
%    stopping criteria. 
        break;
%    else
%        Y = Y + mu*dY;
%        
%        if mu*relChg < tol2
%            mu = min(max_mu, mu*rho);
%        end
    end
    rho = min(1.1*rho,1e5);
   
end

function [E] = solve_l1(W,lambda, mu)
% min lambda |x|_1 + 0.5 mu |x - w|^2_2
E = sign(W) .* max(abs(W) - lambda/mu, 0); 
end

function [E] = solve_l1l2(W,lambda)
n = size(W,2);
E = W;
for i=1:n
    E(:,i) = solve_l2(W(:,i),lambda);
end
end

function [x] = solve_l2(w,lambda)
% min lambda |x|_2 + |x-w|_2^2
nw = norm(w);
if nw>lambda
    x = (nw-lambda)*w/nw;
else
    x = zeros(length(w),1);
end
end
end