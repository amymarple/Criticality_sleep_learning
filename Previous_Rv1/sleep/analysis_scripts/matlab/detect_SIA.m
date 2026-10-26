function [SIA_periods, zscored_signal] = detect_SIA(LFP, srate, spike_times, ripple_timestamps, spike_thresh, ripple_thresh, NREM_epochs)
% DETECT_SIA Detects Small Irregular Activity (SIA) periods from CA1 LFP data
% Input:
%   LFP              - Two-column matrix [timestamps, LFP values]
%   srate            - Sampling rate of the LFP signal (Hz)
%   spike_times      - Cell array of spike timestamps for each unit (1xN cell)
%   ripple_timestamps - Timestamps of detected ripples (start and end)
%   spike_thresh     - Threshold for fraction of active neurons within 25 ms
%   ripple_thresh    - Ripple rate threshold
%   NREM_epochs      - Nx2 matrix with start and end timestamps of NREM epochs
% Output:
%   SIA_periods      - Detected SIA periods (start and end timestamps)
%   zscored_signal   - Z-scored, smoothed LFP signal

% Extract LFP timestamps and values
timestamps = LFP(:, 1);
lfp_values = LFP(:, 2);


% Step 2: Square the filtered LFP signal
squared_LFP = lfp_values  .^ 2;

% Step 3: Smooth the squared signal using a 300-ms Gaussian window
window_size = round(0.3 * srate); % 300 ms window
gaussian_window = gausswin(window_size);
gaussian_window = gaussian_window / sum(gaussian_window); % Normalize
smoothed_LFP = conv(squared_LFP, gaussian_window, 'same');

% Step 4: Take the square root of the smoothed signal
sqrt_LFP = sqrt(smoothed_LFP);

% Step 5: Z-score the resulting signal
zscored_signal = (sqrt_LFP - mean(sqrt_LFP)) / std(sqrt_LFP);

% Global 20th percentile threshold
global_threshold = prctile(zscored_signal, 20);

% Precompute spike times as a matrix for efficiency
spike_matrix = cellfun(@(x) x(:), spike_times, 'UniformOutput', false);
spike_matrix = vertcat(spike_matrix{:});

% Precompute active fraction and ripple rate within each NREM epoch
active_fractions = [];
ripple_rates = [];
time_bins = [];
for epoch = 1:size(NREM_epochs, 1)
    epoch_bins = NREM_epochs(epoch, 1):0.025:NREM_epochs(epoch, 2);
    % Precompute firing rate matrix
    firing_matrix = zeros(length(spike_times), length(epoch_bins) - 1);
    for unit_idx = 1:length(spike_times)
        [counts, ~] = histcounts(spike_times{unit_idx}, epoch_bins);
        firing_matrix(unit_idx, :) = counts;
    end
    % Compute active fraction for each bin
    frac_active = sum(firing_matrix > 0, 1) / length(spike_times);
    active_fractions = [active_fractions; frac_active'];
    % Compute ripple rate
    ripple_counts = histcounts(ripple_timestamps(:, 1), epoch_bins);
    ripple_rate = ripple_counts / 0.025;
    ripple_rates = [ripple_rates; ripple_rate'];
    % Store time bin centers
    time_bins = [time_bins; epoch_bins(1:end-1)' + 0.0125];
end

% Step 6: Identify SIA periods using global threshold
below_threshold = zscored_signal < global_threshold;
starts = find(diff([0; below_threshold]) == 1);
ends = find(diff([below_threshold; 0]) == -1);
intervals = [timestamps(starts), timestamps(ends)];

% Remove intervals shorter than 250 ms
intervals = intervals(diff(intervals, 1, 2) >0, :);

% Validate each SIA interval using precomputed fractions and rates
SIA_periods = [];
for j = 1:size(intervals, 1)
    % Restrict to relevant precomputed values
    bin_indices = find(time_bins >= intervals(j, 1) & time_bins <= intervals(j, 2));
    avg_active_fraction = mean(active_fractions(bin_indices));
    avg_ripple_rate = mean(ripple_rates(bin_indices));
    % Check criteria for the entire interval
    if avg_active_fraction < spike_thresh && avg_ripple_rate < ripple_thresh
        SIA_periods = [SIA_periods; intervals(j, :)];
    end
end

% Step 7: Plot the distribution and threshold
figure;
histogram(zscored_signal, 100, 'BinLimits', [-3, 3]);
hold on;
xline(global_threshold, 'r', '20th Percentile');
title('Z-scored LFP Distribution with SIA Threshold');
xlabel('Z-scored LFP');
ylabel('Count');
hold off;

% Step 8: Plot the duration of SIA periods
SIA_durations = diff(SIA_periods, 1, 2); % Calculate durations
figure;
histogram(SIA_durations, 50);
title('Distribution of SIA Period Durations');
xlabel('Duration (s)');
ylabel('Count');

end
