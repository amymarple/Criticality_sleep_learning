%% D2 time-resolved (state-pure) analysis
% Requirements:
% - getDbetas(st', [t0 t1], deltaT, order, check_fit, quant)
% - *.SleepState.states.mat, *.cell_metrics.cellinfo.mat, *.MergePoints.events.mat (optional), *.session.mat (optional)
% - Optional: restrict to a brain region (e.g., CA1)

clearvars -except i base17 folders17 base18 folders18; clc;

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
br = brain_region;
if isempty(br), br = 'ALL'; end
br_safe = regexprep(char(br), '[^\w-]+', '_');
outFile = fullfile(save_dir, sprintf('%s_%s_D2_time_resolved_error_bar.mat', ...
    basename, br_safe));
% 
% if exist(outFile, 'file')
%     fprintf('File already exists: %s\nStopping script.\n', outFile);
%     return;   % stops the script
% end


% Sleep scoring
load(fullfile(basepath, [basename,'.SleepState.states.mat']), 'SleepState');

% Cell metrics (for spike times by region)
load(fullfile(basepath, [basename,'.cell_metrics.cellinfo.mat']), 'cell_metrics');

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

%% ---------------- Slide windows within each state ----------------
% We generate windows that are guaranteed to be fully inside a single state interval.
labels = {'NREM','REM','WAKE'};% ---------------- Slide windows within each state ----------------
results = struct('state',{},'center',{},'d2',{},'d2_err',{});


rng(0);  % reproducibility for any later resampling

check_every = 500;   % visualize the fit occasionally
wcount = 0;
num_bootstraps = 500;
useParfor    = true;
for s = 1:numel(labels)
    lab  = labels{s};
    rows = strcmp(T.type, lab);
    ivs  = [T.start(rows), T.end(rows)];   % intervals for this state

    [wins, centers] = windows_in_intervals(ivs, window_size, step_size);
    nW = size(wins,1);

    d2_vals = nan(nW,1);
    d2_err_vals = nan(nW,1);   
    for w = 1:nW
        curr_win = wins(w,:);
        wcount = wcount + 1;
        %check_fit = (mod(wcount, check_every)==0);
        check_fit=false;
        try
            % Compute D-betas; we only keep the first (β=2) distance as D²
            [~, d_beta, ~, ~, d_beta_error] = getDbetas_error_bar(st', curr_win, deltaT, order, check_fit);
            d2_vals(w) = d_beta(1);
            d2_err_vals(w) = d_beta_error(1);   % store the corresponding error
            % Optionally save a snapshot of fit diagnostics
            if check_fit
                figfile = fullfile(save_dir, sprintf('fitcheck_%06d_%s.png', wcount, lab));
                saveas(gcf, figfile);
                close(gcf);
            end
    
        catch ME
            % If getDbetas or saveas fails, skip this iteration
            warning('Window %d failed: %s', w, ME.message);
            continue;
        end
    end

    % store
    results(end+1).state  = lab;        %#ok<SAGROW>
    results(end).center   = centers;
    results(end).d2       = d2_vals;
    results(end).d2_err   = d2_err_vals;
end

%% ---------------- Stitch into a single time series ----------------
% We keep them separate by state for stats, but we can also make a global timeline.
all_centers = [];
all_d2      = [];
all_d2_error= [];          
all_state   = [];

for s = 1:numel(results)
    all_centers = [all_centers; results(s).center(:)];
    all_d2      = [all_d2;      results(s).d2(:)];
    all_d2_error= [all_d2_error;results(s).d2_err(:)];
    all_state   = [all_state;   repmat(string(results(s).state), numel(results(s).center), 1)];
end

[all_centers, ord] = sort(all_centers);
all_d2       = all_d2(ord);
all_d2_error = all_d2_error(ord);    % stays aligned with the sort order
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

    mask_state = (all_state == lbl);
    if ~any(mask_state), continue; end

    [runs_start, runs_end] = contiguous_runs(mask_state);
    took_handle = false;

    for k = 1:numel(runs_start)
        idx = runs_start(k):runs_end(k);

        % data for this contiguous run
        x = all_centers(idx);
        y = all_d2(idx);
        e = all_d2_error(idx);   % symmetric ± error

        % keep finite (and positive y for a log scale)
        use_log = strcmpi(get(gca,'YScale'),'log');
        ok = isfinite(x) & isfinite(y) & isfinite(e);
        if use_log, ok = ok & (y > 0); end
        x = x(ok); y = y(ok); e = e(ok);
        if numel(x) < 2, continue; end

        % log-safe: cap error so lower bound stays positive on log-y
        if use_log
            e = min(e, 0.99*y);
        end

        % bounds
        tiny  = 1e-9;                   % guard for log scale
        lower = y - e;
        upper = y + e;
        if use_log
            lower = max(lower, tiny);
        end

        % sort by time so the polygon doesn't self-intersect
        [xs, I] = sort(x);
        ys      = y(I);
        lower   = lower(I);
        upper   = upper(I);

        % shaded band behind the line (same color, faint)
        hBand = fill([xs; flipud(xs)], [lower; flipud(upper)], col, ...
                     'FaceAlpha', 0.50, 'EdgeColor', 'none', 'HandleVisibility','off');
        uistack(hBand,'bottom');

        % mean line on top
        h = plot(xs, ys, '-', 'Color', col, 'LineWidth', 1.5);

        if ~took_handle
            set(h,'DisplayName', lbl);
            leg_handles(end+1,1) = h; %#ok<AGROW>
            leg_names{end+1,1}   = lbl; %#ok<AGROW>
            took_handle = true;
        else
            set(get(get(h,'Annotation'),'LegendInformation'), 'IconDisplayStyle','off');
        end
    end
end

xlabel('Time (s)');
ylabel('D^2 (distance to \beta=2 hyperplane)');
% legend(leg_handles, leg_names, 'Location','best');  % uncomment if needed
box off

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
                        'n_cells', n_cells,'total_spikes',numel(st));
    out.results = results;
    out.timeline = table(all_centers, all_d2, all_d2_error, all_state, ...
        'VariableNames', {'center_s','d2','d2_err','state'});

    br = brain_region; if isempty(br), br = 'ALL'; end
    br_safe = regexprep(char(br), '[^\w-]+', '_');

 %   save(fullfile(save_dir, sprintf('%s_%s_D2_time_resolved_error_bar.mat', ...
   %      basename, br_safe)), '-struct', 'out');
     save(fullfile(save_dir, sprintf('%s_%s_D2_time_resolved_error_bar_forplot.mat', ...
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
