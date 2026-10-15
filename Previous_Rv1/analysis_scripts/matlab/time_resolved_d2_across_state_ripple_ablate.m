%% D2 time-resolved (state-pure) analysis
% Requirements:
% - getDbetas(st', [t0 t1], deltaT, order, check_fit, quant)
% - *.SleepState.states.mat, *.cell_metrics.cellinfo.mat, *.MergePoints.events.mat (optional), *.session.mat (optional)
% - Optional: restrict to a brain region (e.g., CA1)

clear; clc;

%% ---------------- User parameters ----------------
window_size = 30;     % seconds, length of each sliding window
step_size   = 2;      % seconds hop between adjacent windows
guard_sec   = 5;      % seconds shaved off each end of scored intervals
deltaT      = 0.04;   % bin size used inside getDbetas (s)
order       = 20;     % AR order
quant       = 0.10;   % avalanche threshold quantile used by getDbetas (pass-through)
brain_region= 'CA1';  % set [] to use all cells

% Save options
do_save_mat = true;
base_dir=pwd
save_dir = fullfile(base_dir, 'complex_system');
if ~exist(save_dir, 'dir'), mkdir(save_dir); end

%% ---------------- Load session files ----------------
basepath = pwd;
[~, basename] = fileparts(basepath);

% Sleep scoring
load(fullfile(basepath, [basename,'.SleepState.states.mat']), 'SleepState');

% Cell metrics (for spike times by region)
load(fullfile(basepath, [basename,'.cell_metrics.cellinfo.mat']), 'cell_metrics');
load(fullfile(basepath, [basename,'.ripples.events.']), 'ripples');
% (Optional) session/merge info; not strictly required for this script
if exist(fullfile(basepath, [basename,'.MergePoints.events.mat']), 'file')
    load(fullfile(basepath, [basename,'.MergePoints.events.mat']), 'MergePoints');
end
if exist(fullfile(basepath, [basename,'.session.mat']), 'file')
    load(fullfile(basepath, [basename,'.session.mat']), 'session');
end
%% ---------------- Build pooled spike-time list ----------------
if ~isempty(brain_region)
    idx = contains(cell_metrics.brainRegion, brain_region, 'IgnoreCase', true);
%     idx = contains(cell_metrics.brainRegion, 'CA1', 'IgnoreCase', true) | ...
%           contains(cell_metrics.brainRegion, 'CA2', 'IgnoreCase', true);
else
    idx = true(numel(cell_metrics.spikes.times), 1);
end

ca_spike_times = cell_metrics.spikes.times(idx);
st = cell2mat(ca_spike_times(:));     % pooled spike times (s)
st = sort(st(:));                     % ensure ascending column

fprintf('Using %d cells; total spikes: %d\n', nnz(idx), numel(st));

%% ---------------- Collect state intervals ----------------
nrem = SleepState.ints.NREMstate;   % Nx2
rem  = SleepState.ints.REMstate;    % Mx2
wake = SleepState.ints.WAKEstate;   % Kx2

% Erode each interval by guard_sec, and drop short/invalid results
nrem = erode_and_filter(nrem, guard_sec, window_size);
rem  = erode_and_filter(rem,  guard_sec, window_size);
wake = erode_and_filter(wake, guard_sec, window_size);

% Build a combined table of (start,end,type) and sort
T = table;
T = addIntervals(T, nrem, 'NREM');
T = addIntervals(T, rem,  'REM');
T = addIntervals(T, wake, 'WAKE');
T = sortrows(T, 'start');

%% ================== Time-resolved D2 with Ripple Ablation ==================
% States to analyze
labels = {'NREM','REM','WAKE'};

% Pre-allocate structure array (faster than growing it)
results = repmat(struct('state','','center',[],'d2',[],'d2_err',[]), numel(labels), 1);

rng(0);  % reproducibility

% Minimum total ripple-free duration (in seconds) required
MIN_FREE_SEC = 1.0;   % tune if needed (e.g., 0.5, 1.0, 2.0)

%% --- Check / start Parallel Pool ---
if isempty(gcp('nocreate'))
    parpool; % starts workers if not already running
end

%% --- Prepare Ripple Data ---
ripple_iv = [];
if exist('ripples','var') && isfield(ripples, 'timestamps') && ~isempty(ripples.timestamps)
    ripple_iv = sortrows(ripples.timestamps, 1);   % sort by start time
end
nRip = size(ripple_iv,1);

%% ================== Main State Loop ==================
for s = 1:numel(labels)
    lab  = labels{s};
    rows = strcmp(T.type, lab);
    ivs  = [T.start(rows), T.end(rows)];   % [start, end] intervals for this state

    % Sliding windows within this state
    [wins, centers] = windows_in_intervals(ivs, window_size, step_size);
    nW = size(wins,1);

    % Pre-allocate output vectors for this state
    d2_vals     = nan(nW,1);
    d2_err_vals = nan(nW,1);

    isNREM = strcmp(lab,'NREM');

    % Use Parallel Loop
    % Diagnostic plotting (check_fit) is disabled in the parallel loop.
    parfor w = 1:nW
        curr_win = wins(w,:);      % [start, end] of this 30 s window

        % local copies for parfor
        local_d2      = NaN;
        local_d2_err  = NaN;

        try
            win_use = curr_win;  % default: whole window

            % ----- Ripple subtraction for NREM -----
            if isNREM && nRip > 0
                % Quick coarse check: does this window even overlap ripple range?
                if curr_win(2) > ripple_iv(1,1) && curr_win(1) < ripple_iv(end,2)
                    % Subtract ripples from this window -> ripple-free subwindows
                    win_use = fast_subtract_intervals(curr_win, ripple_iv);
                end

                % If nothing left after ablation, mark as NaN and skip
                if isempty(win_use)
                    d2_vals(w)     = local_d2;
                    d2_err_vals(w) = local_d2_err;
                    continue;
                end
            end

            % ----- Minimum duration check (avoid fitting AR on tiny snippets) -----
            if size(win_use,1) > 1
                total_len = sum(win_use(:,2) - win_use(:,1));
            else
                total_len = win_use(2) - win_use(1);
            end

            if total_len < MIN_FREE_SEC
                % Not enough ripple-free data in this window for stable AR fit
                d2_vals(w)     = local_d2;
                d2_err_vals(w) = local_d2_err;
                continue;
            end

            % ----- Compute D-betas (no diagnostics in parfor) -----
            % 'false' disables plotting/check_fit inside getDbetas
            [~, d_beta, ~, d_beta_err] = getDbetas(st', win_use, deltaT, order, false, quant);

            local_d2     = d_beta(1);       % use β = 2 as D²
            local_d2_err = d_beta_err(1);   % corresponding error

            d2_vals(w)     = local_d2;
            d2_err_vals(w) = local_d2_err;

        catch
            % On failure, leave NaN for this window.
            d2_vals(w)     = local_d2;
            d2_err_vals(w) = local_d2_err;
        end
    end

    % Store results for this state
    results(s).state    = lab;
    results(s).center   = centers;
    results(s).d2       = d2_vals;
    results(s).d2_err   = d2_err_vals;

    fprintf('Finished state: %s (%d windows)\n', lab, nW);
end

%% ---------------- Stitch into a single time series ----------------
% We keep them separate by state for stats, but we can also make a global timeline.
all_centers = [];
all_d2      = [];
all_state   = [];

for s = 1:numel(results)
    all_centers = [all_centers; results(s).center(:)];
    all_d2      = [all_d2;      results(s).d2(:)];
    all_state   = [all_state;   repmat(string(results(s).state), numel(results(s).center), 1)];
end

[all_centers, ord] = sort(all_centers);
all_d2       = all_d2(ord);
all_state    = all_state(ord);


%% ---------------- Plot: time-resolved D² with state colors ----------------
f = figure('Color','w'); hold on

% Colors per state
state_keys = {'NREM','REM','WAKE'};
cols = lines(numel(state_keys));
C = containers.Map(state_keys, num2cell(cols,2));

% We'll *append* legend handles only for states that appear
leg_handles = gobjects(0);
leg_names   = {};

for s = 1:numel(labels)
    lbl = labels{s};
    % fallback color if label isn't in the map
    if isKey(C, lbl), col = C(lbl); else, col = [0.4 0.4 0.4]; end

    mask = (all_state == lbl);

    % skip entirely if this state has no samples
    if ~any(mask)
        continue
    end

    [runs_start, runs_end] = contiguous_runs(mask);
    took_handle = false;

    for k = 1:numel(runs_start)
        idx = runs_start(k):runs_end(k);
        h = plot(all_centers(idx), all_d2(idx), '-', 'Color', col, 'LineWidth', 1.5);
        % Faint error bars over the colored line segments:
        hold on;



        if ~took_handle
            % keep only the FIRST segment of this state for the legend
            set(h,'DisplayName', lbl);
            leg_handles(end+1,1) = h; %#ok<AGROW>
            leg_names{end+1,1}   = lbl; %#ok<AGROW>
            took_handle = true;
        else
            % hide other segments of the same state from legend
            set(get(get(h,'Annotation'),'LegendInformation'), 'IconDisplayStyle','off');
        end
    end
end

xlabel('Time (s)');
ylabel('D^2 (distance to \beta=2 hyperplane)');

title(sprintf('State-pure time-resolved D^2 (win=%gs, hop=%gs, guard=%gs, AR(p=%d)) [%s]', ...
    window_size, step_size, guard_sec, order, brain_region));

% Only call legend if we actually have at least one handle
if ~isempty(leg_handles)
    legend(leg_handles, leg_names, 'Location','best');
else
    legend('off');
end

grid on; box off;

% Save figure
br = brain_region;
if isempty(br), br = 'ALL'; end
br_safe = regexprep(char(br), '[^\w-]+', '_');
fname_fig = sprintf('D2_time_resolved_%s.fig', br_safe);
savefig(f, fullfile(save_dir, fname_fig));


%% ---------------- Save outputs ----------------
n_cells = numel(ca_spike_times);      % <— add this
%---------------- Save outputs ----------------
if do_save_mat
    out = struct();
    out.params = struct('window_size',window_size,'step_size',step_size,'guard_sec',guard_sec, ...
                        'deltaT',deltaT,'order',order,'quant',quant,'brain_region',brain_region, ...
                        'n_cells', n_cells,'total_spikes',numel(st));   % <— saved in params
    out.results = results;
    out.timeline = table(all_centers, all_d2, all_state, ...
        'VariableNames', {'center_s','d2','state'});

    br = brain_region;
    if isempty(br), br = 'ALL'; end                 % fallback
    br_safe = regexprep(char(br), '[^\w-]+', '_');  % filename-safe

    save(fullfile(save_dir, sprintf('%s_%s_D2_time_resolved_rippleablate.mat', ...
         basename, br_safe)), '-struct', 'out');
end
fprintf('Done. Windows computed: %d\n', numel(all_d2));

% ---------------- Helper functions ----------------
function T = addIntervals(T, iv, label)
    if isempty(iv), return; end
    n = size(iv,1);
    T = [T; table(iv(:,1), iv(:,2), repmat({label},n,1), ...
        'VariableNames', {'start','end','type'})];
end

function iv2 = erode_and_filter(iv, guard, min_len)
    % Erode each interval by 'guard' seconds on both ends;
    % keep only intervals with remaining duration >= min_len
    if isempty(iv), iv2 = iv; return; end
    iv2 = [iv(:,1)+guard, iv(:,2)-guard];
    dur = iv2(:,2) - iv2(:,1);
    keep = dur >= min_len;
    iv2 = iv2(keep,:);
end

function [wins, centers] = windows_in_intervals(iv, wlen, hop)
    % Generate sliding windows fully inside any of the intervals in iv
    wins = zeros(0,2);
    for i = 1:size(iv,1)
        t0 = iv(i,1); t1 = iv(i,2);
        if (t1 - t0) < wlen, continue; end
        starts = t0 : hop : (t1 - wlen);
        if isempty(starts), continue; end
        ends   = starts + wlen;
        wins   = [wins; [starts(:), ends(:)]]; %#ok<AGROW>
    end
    centers = mean(wins,2);
end

function [rs, re] = contiguous_runs(mask)
    % Return start/end indices of contiguous true runs in a logical vector
    mask = mask(:)';
    d = diff([false, mask, false]);
    rs = find(d==1);
    re = find(d==-1) - 1;
end
function win_out = fast_subtract_intervals(win, ripples)
% FAST_SUBTRACT_INTERVALS
%   win     : 1×2 [start, end] window
%   ripples : R×2 [start, end], sorted by start time
% Returns:
%   win_out : M×2 ripple-free subwindows within 'win'

    t0 = win(1);
    t1 = win(2);

    % Find ripples that overlap this window:
    % (start < t1) AND (end > t0)
    idx = ripples(:,1) < t1 & ripples(:,2) > t0;

    if ~any(idx)
        % No overlapping ripples -> entire window is ripple-free
        win_out = win;
        return;
    end

    relevant_rips = ripples(idx, :);

    % Subtract ripples (linear scan on small subset)
    win_out = zeros(size(relevant_rips,1)+1, 2); % upper bound on number of segments
    count = 0;

    curr_start = t0;

    for i = 1:size(relevant_rips, 1)
        r0 = max(relevant_rips(i,1), t0);  % ripple start inside window
        r1 = min(relevant_rips(i,2), t1);  % ripple end inside window

        % Interval BEFORE this ripple, if any
        if r0 > curr_start
            count = count + 1;
            win_out(count, :) = [curr_start, r0];
        end

        % Move pointer to AFTER this ripple
        curr_start = max(curr_start, r1);

        if curr_start >= t1
            break;
        end
    end

    % Tail AFTER the last overlapping ripple
    if curr_start < t1
        count = count + 1;
        win_out(count, :) = [curr_start, t1];
    end

    % Trim pre-allocation
    win_out = win_out(1:count, :);
end
