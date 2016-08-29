function [x, cost, info, options] =  FISTA_manfold(problem, x, options)

% The outputs x and cost are the best reached point on the manifold and its
% cost. The struct-array info contains information about the iterations:
%   iter : the iteration number (0 for the initial guess)
%   cost : cost value
%   time : elapsed time in seconds
%   gradnorm : Riemannian norm of the gradient
%   stepsize : norm of the last tangent vector retracted
%   beta : value of the beta parameter (see options.beta_type)
%   linesearch : information logged by options.linesearch
%   And possibly additional information logged by options.statsfun.
% For example, type [info.gradnorm] to obtain a vector of the successive
% gradient norms reached.
%
% The options structure is used to overwrite the default values. All
% options have a default value and are hence optional. To force an option
% value, pass an options structure with a field options.optionname, where
% optionname is one of the following and the default value is indicated
% between parentheses:
%
%   tolgradnorm (1e-6)
%       The algorithm terminates if the norm of the gradient drops below this.
%   maxiter (1000)
%       The algorithm terminates if maxiter iterations have been executed.
%   maxtime (Inf)
%       The algorithm terminates if maxtime seconds elapsed.
%   minstepsize (1e-10)
%       The algorithm terminates if the linesearch returns a displacement
%       vector (to be retracted) smaller in norm than this value.
%   beta_type ('H-S')
%       Conjugate gradient beta rule used to construct the new search
%       direction, based on a linear combination of the previous search
%       direction and the new (preconditioned) gradient. Possible values
%       for this parameter are:

% Verify that the problem description is sufficient for the solver.
if ~canGetCost(problem)
    warning('manopt:getCost', ...
        'No cost provided. The algorithm will likely abort.');
end
if ~canGetGradient(problem)
    warning('manopt:getGradient', ...
        'No gradient provided. The algorithm will likely abort.');
end

% Set local defaults here
localdefaults.minstepsize = 1e-10;
localdefaults.maxiter = 1000;
localdefaults.tolgradnorm = 1e-6;
localdefaults.storedepth = 2;
% Changed by NB : H-S has the "auto restart" property.
% See Hager-Zhang 2005/2006 survey about CG methods.
% Well, the auto restart comes from the 'max(0, ...)', not so much from the
% reason stated in Hager-Zhang I believe. P-R also has auto restart.
localdefaults.beta_type = 'H-S';
localdefaults.orth_value = Inf; % by BM as suggested in Nocedal and Wright


% Depending on whether the problem structure specifies a hint for
% line-search algorithms, choose a default line-search that works on
% its own (typical) or that uses the hint.
if ~canGetLinesearch(problem)
    localdefaults.linesearch = @linesearch_adaptive;
    localdefaults.linesearchfast = @linesearch_fast;
else
    localdefaults.linesearch = @linesearch_hint;
end

% Merge global and local defaults, then merge w/ user options, if any.
localdefaults = mergeOptions(getGlobalDefaults(), localdefaults);
if ~exist('options', 'var') || isempty(options)
    options = struct();
end
options = mergeOptions(localdefaults, options);

% for convenience
inner = problem.M.inner;
lincomb = problem.M.lincomb;

% Create a store database
storedb = struct();

timetic = tic();

% If no initial point x is given by the user, generate one at random.
if ~exist('x', 'var') || isempty(x)
    x = problem.M.rand();
end

y = x;
% Compute objective-related quantities for x
[cost grad storedb] = getCostGrad(problem, y, storedb);
gradnorm = problem.M.norm(y, grad);
[Pgrad storedb] = getPrecon(problem, y, grad, storedb);
gradPgrad = inner(y, grad, Pgrad);

% Iteration counter (at any point, iter is the number of fully executed
% iterations so far)
iter = 0;

% Save stats in a struct array info and preallocate,
% see http://people.csail.mit.edu/jskelly/blog/?x=entry:entry091030-033941
stats = savestats();
info(1) = stats;
info(min(10000, options.maxiter+1)).iter = [];

% Initial linesearch memory
lsmem = [];

%%----------------------------------------------------------------
% if options.verbosity >= 2
%     fprintf(' iter\t                cost val\t     grad. norm\n');
% end

% Compute a first descent direction (not normalized)
desc_dir = lincomb(y, -1, Pgrad);

t = 1;

% Start iterating until stopping criterion triggers
while true
    
    % % Display iteration information---------------------------------
    %     if options.verbosity >= 2
    %           fprintf('%5d\t%+.16e\t%.8e\n', iter, cost, gradnorm);
    %     end
    
    % Start timing this iteration
    timetic = tic();
    
    % Run standard stopping criterion checks
    [stop reason] = stoppingcriterion(problem, y, options, info, iter+1);
    
    % Run specific stopping criterion check
    if ~stop && abs(stats.stepsize) < options.minstepsize
        stop = true;
        reason = 'Last stepsize smaller than minimum allowed. See options.minstepsize.';
    end
    
    if stop
        if options.verbosity >= 1
            %       fprintf([reason '\n']);
        end
        break;
    end
    
    
    % The line search algorithms require the directional derivative of the
    % cost at the current point x along the search direction.
    df0 = -inner(y, grad, grad);
    
    % If we didn't get a descent direction: restart, i.e., switch to the
    % negative gradient. Equivalent to resetting the CG direction to a
    % steepest descent step, which discards the past information.
    if df0 >= 0
        
        % Or we switch to the negative gradient direction.
        if options.verbosity >= 3
            fprintf(['Conjugate gradient info: got an ascent direction '...
                '(df0 = %2e), reset to the (preconditioned) '...
                'steepest descent direction.\n'], df0);
        end
        % Reset to negative gradient: this discards the CG memory.
        desc_dir = lincomb(y, -1, Pgrad);
        df0 = -gradPgrad;
        
    end
    
    % Execute line search
    [stepsize newx storedb lsmem lsstats] = options.linesearchfast(...
        problem, y, desc_dir, cost, df0, options, storedb, lsmem);
    
    newx.S = diag(max(diag(newx.S)-0.1*stepsize,0));
    [newxcost newxgrad storedb] = getCostGrad(problem, newx, storedb);
    
    % Fast iterative
    if iter==0
        newy = newx;
        [newycost newygrad storedb] = getCostGrad(problem, newy, storedb);
        newt = (1+sqrt(1+4*t^2))/2;
        newygradnorm = problem.M.norm(newy, newygrad);
        [Pnewygrad storedb] = getPrecon(problem, newy, newygrad, storedb);
        newgradPnewgrad = inner(newy, newygrad, Pnewygrad);
        %        dir = scaleTxM(gc_new, -1);
    elseif iter ==1;
        newy = newx;
        [newycost newygrad storedb] = getCostGrad(problem, newy, storedb);
        newt = (1+sqrt(1+4*t^2))/2;
        newygradnorm = problem.M.norm(newy, newygrad);
        [Pnewygrad storedb] = getPrecon(problem, newy, newygrad, storedb);
        newgradPnewgrad = inner(newy, newygrad, Pnewygrad);
        %       dir = scaleTxM(gc_new, -1);
        newxmat = newx.U*newx.S*newx.V';
    elseif iter>1
        
        newt = (1+sqrt(1+4*t^2))/2;
        beta = (t-1)/newt;
        newxmat = newx.U*newx.S*newx.V';
        x_newx= problem. M.proj(newx,newxmat-xmat);
        newy = problem.M.retr(newx, x_newx,beta);
        [newycost newygrad storedb] = getCostGrad(problem, newy, storedb);
         newgradPnewgrad = inner(newy, newygrad, Pnewygrad);
       
        if newycost- newxcost >0
            newy = newx;
            newygrad = newxgrad;
            newygradnorm = newxgradnorm;
            [Pnewygrad storedb] = getPrecon(problem, newy, newygrad, storedb);
            newgradPnewgrad = inner(newy, newygrad, Pnewygrad);
        else
            newt = (1+sqrt(1+4*t^2))/2;
            tempt = (t-1)/newt;
            % xmat = x.U*x.S*x.V';
            newxmat = newx.U*newx.S*newx.V';
            x_newx= problem. M.proj(newx,newxmat-xmat);
            newy = problem.M.retr(newx, x_newx,tempt);
            % Compute the new cost-related quantities for x
            [newycost newygrad storedb] = getCostGrad(problem, newy, storedb);
            % newgradnorm = problem.M.norm(newy, newygrad);
            [Pnewygrad storedb] = getPrecon(problem, newy, newygrad, storedb);
            newgradPnewgrad = inner(newy, newygrad, Pnewygrad);
        end
    end
    
    if strcmpi(options.beta_type, 'steep') || ...
            strcmpi(options.beta_type, 'S-D')              % Gradient Descent
        
        beta = 0;
        desc_dir = lincomb(y, -1, Pnewygrad);
        
    else
        
        oldgrad = problem.M.transp(y, newy, grad);
        orth_grads = inner(newy, oldgrad, Pnewygrad)/newgradPnewgrad;
        
        % Powell's restart strategy (see page 12 of Hager and Zhang's
        % survey on conjugate gradient methods, for example)
        if abs(orth_grads) >= options.orth_value,
            beta = 0;
            desc_dir = lincomb(y, -1, Pnewygrad);
            
        else % Compute the CG modification
            
            desc_dir = problem.M.transp(y, newy, desc_dir);
            
            if strcmp(options.beta_type, 'F-R')  % Fletcher-Reeves
                beta = newgradPnewgrad / gradPgrad;
                
            elseif strcmp(options.beta_type, 'P-R')  % Polak-Ribiere+
                % vector grad(new) - transported grad(current)
                diff = lincomb(newy, 1, newygrad, -1, oldgrad);
                ip_diff = inner(newy, Pnewygrad, diff);
                beta = ip_diff/gradPgrad;
                beta = max(0, beta);
                
            elseif strcmp(options.beta_type, 'H-S')  % Hestenes-Stiefel+
                diff = lincomb(newy, 1, newygrad, -1, oldgrad);
                ip_diff = inner(newy, Pnewygrad, diff);
                beta = ip_diff / inner(newy, diff, desc_dir);
                beta = max(0, beta);
                
            elseif strcmp(options.beta_type, 'H-Z') % Hager-Zhang+
                diff = lincomb(newy, 1, newygrad, -1, oldgrad);
                Poldgrad = problem.M.transp(y, newy, Pgrad);
                Pdiff = lincomb(newy, 1, Pnewygrad, -1, Poldgrad);
                deno = inner(newy, diff, desc_dir);
                numo = inner(newy, diff, Pnewygrad);
                numo = numo - 2*inner(newy, diff, Pdiff)*...
                    inner(newy, desc_dir, newygrad)/deno;
                beta = numo/deno;
                
                % Robustness (see Hager-Zhang paper mentioned above)
                desc_dir_norm = problem.M.norm(newy, desc_dir);
                eta_HZ = -1/(desc_dir_norm * min(0.01, gradnorm));
                beta = max(beta,  eta_HZ);
                
            else
                error(['Unknown options.beta_type. ' ...
                    'Should be steep, S-D, F-R, P-R, H-S or H-Z.']);
            end
            desc_dir = lincomb(newy, -1, Pnewygrad, beta, desc_dir);
        end
        
    end
    
    % Make sure we don't use too much memory for the store database.
    storedb = purgeStoredb(storedb, options.storedepth);
    
    
    % Update iterate info
    x = newx;
    y =newy;
    t = newt;
    if iter>0
        xmat=newxmat;
    end
    cost = newycost;
    grad = newygrad;
    Pgrad = Pnewygrad;
    gradnorm = newygradnorm;
    gradPgrad = newgradPnewgrad;
    
    % iter is the number of iterations we have accomplished.
    iter = iter + 1;
    
    % Log statistics for freshly executed iteration
    stats = savestats();
    info(iter+1) = stats; %#ok<AGROW>
    
end


info = info(1:iter+1);

% if options.verbosity >= 1
%        fprintf('Total time is %f [s] (excludes statsfun)\n', info(end).time);
% end

% Routine in charge of collecting the current iteration stats
    function stats = savestats()
        stats.iter = iter;
        stats.cost = cost;
        stats.gradnorm = gradnorm;
        if iter == 0
            stats.stepsize = nan;
            stats.time = toc(timetic);
            stats.linesearch = [];
            stats.beta = 0;
        else
            stats.stepsize = stepsize;
            stats.time = info(iter).time + toc(timetic);
            stats.linesearch = lsstats;
            stats.beta = beta;
        end
        stats = applyStatsfun(problem, x, storedb, options, stats);
    end

end


