function [beta, d_beta, closest_models, explained_variance, d_beta_error] = getDbetas_error_bar(st, windows, deltaT, order, check_fit, quant)
% d_beta + error bars using FULL observed Hessian covariance and Euclidean gradient.
% - varphi estimated by myYuleWalker
% - Cov(phi) ≈ (-H)^{-1}, H from profiled Gaussian/CLS objective
% - d_beta is Euclidean distance to the beta-manifold (2024 definition)

    %#ok<*NASGU>
    if nargin < 6, quant = []; end

    % --- Fit (Yule-Walker) ---
    [varphi, varNoise, sigAll] = myYuleWalker(st, windows, deltaT, order);

    % varNoise from myYuleWalker is normalized (r0=1)
    explained_variance = max(0, min(1, 1 - max(varNoise,0)));

    beta = (2:2:(2*order)).';
    d_beta = zeros(size(beta));
    d_beta_error = zeros(size(beta));
    closest_models = zeros(numel(beta), order);

    % --- FULL Hessian covariance (profiled), Euclidean geometry ---
    cov_fac = cov_factor_full_hessian_euclid(st, windows, deltaT, order, varphi, 'window', 'ols');


    % --- Loop over beta ---
    for i = 1:numel(beta)
        [d_beta(i), closest_phi] = getFixedPointDistance(order, beta(i), varphi);

        if d_beta(i) < eps
            d_beta_error(i) = 0;  % on the manifold
        else
            g = (varphi(:) - closest_phi(:)) / d_beta(i);   % Euclidean unit normal
            v = apply_cov_fac(cov_fac, g);
            d_beta_error(i) = sqrt( max(g.' * v, 0) );
        end

        closest_models(i,:) = closest_phi(:).';
    end

    if check_fit
        subplot(2,3,6);
        errorbar(beta, d_beta, d_beta_error, 'k.', 'MarkerSize', 8, 'LineStyle', 'none');
        xlabel('\beta'); ylabel('d_\beta');
        set(gca,'YScale','log'); set(gcf,'Color','w'); box off;
    end
end

% ----------------------------------------------------------------------
function cov_fac = cov_factor_full_hessian_euclid(st, windows, deltaT, p, varphi, mean_mode, cov_mode, ridge)
% COV_FACTOR_FULL_HESSIAN_EUCLID
% Build a factorized covariance operator for AR(p) coefficients φ based on
% stacked-window AR regression. Intended for use with Euclidean δ-method:
%   SE[d_beta] ≈ sqrt( g' * Cov(φ) * g ),  g = (φ̂ - φ*_β)/||φ̂ - φ*_β||.
%
% INPUTS
%   st         : signal or spike-times container (vector | cell | struct)
%   windows    : [nW x 2] sample indices, 1-based inclusive [start end]
%   deltaT     : sampling period (sec); used when binning spike times
%   p          : AR order
%   varphi     : (optional) coefficient vector at which to evaluate the
%                observed Hessian; not used in 'ols'/'expected' modes
%   mean_mode  : 'global' (one mean across all rows) or 'window' (demean per window)
%   cov_mode   : 'ols'      -> Cov(φ) = σ̂² (X'X)^{-1}       [RECOMMENDED]
%                'expected' -> same as 'ols' (alias)
%                'observed_at_mle' -> use observed Hessian at φ_MLE
%                'observed_at_phi' -> observed Hessian at provided varphi
%   ridge      : small ridge added if needed (default: adaptive)
%
% OUTPUT
%   cov_fac : struct describing inverse(Hessian) without forming explicit inv:
%       .type  = 'chol' or 'direct'
%       .R     = upper Cholesky of M (when type=='chol'), where Cov ≈ inv(M)
%       .M     = SPD matrix (when chol fails; used with backslash)
%       .scale = scaling (usually 1.0)
%
% NOTES
% - This function *stacks windows* into one AR design X,y (without storing it),
%   accumulating G=X'X, C=X'y, and yy=y'y.
% - In 'ols'/'expected' modes, Cov(φ) = σ̂² (X'X)^{-1}, with σ̂² = RSS/T.
% - In observed modes, we form H = T*( G/A - 2*B B'/A^2 ), with
%   A = e'e, B = X'e, G = X'X, and e = y - Xφ.  Cov ≈ (-H)^{-1}.
%   (At MLE, B≈0 so observed ≈ expected.)
%
% -------------------------------------------------------------------------

    if nargin < 6 || isempty(mean_mode), mean_mode = 'window'; end
    if nargin < 7 || isempty(cov_mode),  cov_mode  = 'ols';    end
    if nargin < 8 || isempty(ridge),     ridge     = [];        end
    if isempty(varphi), varphi = zeros(p,1); end

    % ---------- 1) Accumulate sufficient statistics over all windows ----------
    [G, C, yy, T_like] = accumulate_stats_(st, windows, deltaT, p, mean_mode);

    % Guard: not enough rows or ill-conditioned design
    if T_like <= p || ~all(isfinite([yy, T_like])) || any(~isfinite(G(:))) || rcond_safe_(G) < 1e-12
        % Fall back to a tiny isotropic covariance
        d = max(p,1);
        I = eye(d);
        scale = 1 / max(T_like - p, 1);
        cov_fac = chol_or_direct_(scale*I);
        return
    end

    % ---------- 2) Choose covariance mode ----------
    cov_mode = lower(string(cov_mode));
    switch cov_mode
        case {"ols","expected"}
            % MLE / expected Fisher: Cov = σ̂² (X'X)^{-1}
            phi_mle = G \ C;
            % RSS via normal equations (one pass)
            A = yy - 2*phi_mle.'*C + phi_mle.'*G*phi_mle;     % = ||y - Xφ_mle||^2
            sigma2 = max(A / max(T_like,1), eps);
            % We want a factorization for inv(Cov) = (1/σ̂²) * G
            M = (1/max(sigma2,eps)) * G;

        case "observed_at_mle"
            % Observed Hessian at MLE (B≈0 -> essentially expected Fisher)
            phi_mle = G \ C;
            % Compute A and (tiny) B at MLE:
            A = yy - 2*phi_mle.'*C + phi_mle.'*G*phi_mle;
            B = C - G*phi_mle;                                  % ≈ 0
            M = T_like * ( (G / max(A,eps)) - (2*(B*B.') / max(A,eps)^2) );

        case "observed_at_phi"
            % Observed Hessian at supplied varphi (can be ill-conditioned if far from MLE)
            % Need A and B at varphi
            [A, B] = residual_moments_at_phi_(st, windows, deltaT, p, mean_mode, varphi);
            M = T_like * ( (G / max(A,eps)) - (2*(B*B.') / max(A,eps)^2) );

        otherwise
            error('Unknown cov_mode: %s', cov_mode);
    end

    % Symmetrize and add minimal ridge if needed
    M = (M + M.') * 0.5;
    if isempty(ridge)
        % Adaptive ridge proportional to trace if needed
        ridge = 0;
        if ~isfinite(rcond_safe_(M)) || rcond_safe_(M) < 1e-10
            ridge = max(trace(M), 1) * 1e-8 / max(p,1);
        end
    end
    if ridge > 0, M = M + ridge * eye(p); end

    % ---------- 3) Return factorization for Cov ≈ inv(M) ----------
    cov_fac = chol_or_direct_(M);
    cov_fac.scale = 1.0;    % kept for API symmetry

end

% ======================================================================
% =                           SUBFUNCTIONS                             =
% ======================================================================
function v = apply_cov_fac(fac, g)
% APPLY_COV_FAC  Compute v = (scale * inv(M)) * g using a stored factorization.
% - fac.type == 'chol': fac.R is upper-triangular with M = R'*R
% - fac.type == 'direct': fac.M is the SPD matrix M
% g can be p×1 or p×k (multiple columns handled at once).

    if ~isfield(fac,'scale') || isempty(fac.scale), fac.scale = 1.0; end

    switch lower(string(fac.type))
        case "chol"
            % inv(M)*g = R \ (R' \ g)
            v = fac.scale * (fac.R \ (fac.R' \ g));
        case "direct"
            % inv(M)*g = M \ g
            v = fac.scale * (fac.M \ g);
        otherwise
            error('apply_cov_fac:badfac','Unknown factor type: %s', string(fac.type));
    end
end

function [G, C, yy, T_like] = accumulate_stats_(st, windows, deltaT, p, mean_mode)
% Accumulate G=X'X, C=X'y, yy=y'y, and total row count T_like across windows.
    xs = bin_segments_fast_(st, windows, deltaT);
    nW = numel(xs);

    % Global mean if requested
    switch lower(mean_mode)
        case 'global'
            mu = mean( vertcat(xs{:}) , 'omitnan' );
            if isempty(mu) || ~isfinite(mu), mu = 0; end
        otherwise % 'window'
            mu = [];
    end

    G = zeros(p,p);
    C = zeros(p,1);
    yy = 0;
    T_like = 0;

    for i = 1:nW
        x = xs{i};
        if isempty(x), continue; end
        x = x(:);

        T = numel(x);
        if T <= p, continue; end

        if isempty(mu), xm = x - mean(x);
        else,           xm = x - mu;
        end

        K = T - p;                       % #rows contributed by this window
        % Build lagged regressor Xi (K x p) and target y (K x 1)
        Xi = zeros(K, p);
        for j = 1:p
            Xi(:, j) = xm((p - j + 1):(T - j));
        end
        y  = xm(p+1:T);

        % Update sufficient statistics
        G = G + Xi.' * Xi;               % X'X
        C = C + Xi.' * y;                % X'y
        yy = yy + y.' * y;               % y'y
        T_like = T_like + K;
    end
end

function [A, B] = residual_moments_at_phi_(st, windows, deltaT, p, mean_mode, phi)
% Compute A = e'e and B = X'e at a given φ by streaming over windows.
    xs = bin_segments_fast_(st, windows, deltaT);
    nW = numel(xs);

    switch lower(mean_mode)
        case 'global'
            mu = mean( vertcat(xs{:}) , 'omitnan' );
            if isempty(mu) || ~isfinite(mu), mu = 0; end
        otherwise
            mu = [];
    end

    A = 0;
    B = zeros(p,1);
    for i = 1:nW
        x = xs{i};
        if isempty(x), continue; end
        x = x(:);

        T = numel(x);
        if T <= p, continue; end

        if isempty(mu), xm = x - mean(x);
        else,           xm = x - mu;
        end

        K = T - p;
        Xi = zeros(K, p);
        for j = 1:p
            Xi(:, j) = xm((p - j + 1):(T - j));
        end
        y  = xm(p+1:T);

        e  = y - Xi*phi(:);
        A  = A + e.'*e;
        B  = B + Xi.'*e;
    end
end

function xs = bin_segments_fast_(st, windows, deltaT)
% Return a cell array of binned values for each [start end] interval.
% Uses user-defined getFR if available; otherwise bins spike times.
    nW = size(windows,1);
    xs = cell(nW,1);

    if exist('getFR','file') == 2
        for i = 1:nW
            s = windows(i,1); e = windows(i,2);
            if e <= s, xs{i} = zeros(0,1); continue; end
            xi = getFR([s e], st, deltaT);
            xs{i} = xi(:);
        end
        return
    end

    spikes = pooledSpikes_(st);
    if isempty(spikes)
        for i = 1:nW, xs{i} = zeros(0,1); end
        return
    end

    t0 = min(windows(:,1)); t1 = max(windows(:,2));
    if t1 <= t0
        for i = 1:nW, xs{i} = zeros(0,1); end
        return
    end

    edges = t0:deltaT:t1;
    if edges(end) < t1, edges = [edges, t1]; end
    if numel(edges) < 2
        for i = 1:nW, xs{i} = zeros(0,1); end
        return
    end

    counts = histcounts(spikes, edges).';
    numBins = numel(counts);
    for i = 1:nW
        s = windows(i,1); e = windows(i,2);
        if e <= s, xs{i} = zeros(0,1); continue; end
        i_start = max(1, floor((s - t0)/deltaT) + 1);
        i_end   = min(numBins, ceil((e - t0)/deltaT));
        if i_end < i_start
            xs{i} = zeros(0,1);
        else
            xs{i} = counts(i_start:i_end);
        end
    end
end

function v = pooledSpikes_(st)
% Flatten spike-time containers: struct/cell/vector -> sorted column
    if isstruct(st)
        c = struct2cell(st);
        if isempty(c), v = zeros(0,1); return; end
        v = vertcat(c{:});
    elseif iscell(st)
        if isempty(st), v = zeros(0,1); return; end
        v = vertcat(st{:});
    else
        v = st;
    end
    v = v(:);
    v = v(isfinite(v));
    if isempty(v), return; end
    v = sort(v, 'ascend');
end

function r = rcond_safe_(A)
    r = rcond((A+A')*0.5);
    if ~isfinite(r), r = 0; end
end

function fac = chol_or_direct_(M)
% Represent inv(M) with a factor (prefer Cholesky).
    M = (M + M.') * 0.5;
    [R,pflag] = chol(M);
    if pflag == 0
        fac.type  = 'chol';
        fac.R     = R;       % M = R'R
        fac.M     = [];
        fac.scale = 1.0;
    else
        fac.type  = 'direct';
        fac.M     = M;
        fac.R     = [];
        fac.scale = 1.0;
    end
end


