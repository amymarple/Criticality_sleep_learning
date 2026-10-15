
clear all
basepath=pwd
[~,basename] = fileparts(basepath);
savepath = 'D:\Documents\GitHub\R_Project\PPP\codebook\Criticality_analysis\ddc\data';
brain_region = 'CA1';
animal='mouse'
% Extract animal name and day from basepath
[path_parts] = split(basepath, filesep); % Split the basepath using file separator
animal_name = path_parts{4}; % Assuming 'PPP7' is the third part of the path
day = basename         % Assuming 'day7' is the fourth part of the path
basepath = char(basepath); % Convert string to char array
basename = char(basename); % Convert basename to char array
load(fullfile(basepath, [basename, '.cell_metrics.cellinfo.mat']), 'cell_metrics');
load(fullfile(basepath, [basename, '.animal.behavior.mat']), 'behavior');

% Construct the file name
load(fullfile(basepath, [basename, '.ripples.events.mat']), 'ripples');
load(fullfile(basepath, [basename, '.SleepState.states.mat']), 'SleepState');
load(fullfile(basepath, [basename, '.session.mat']), 'session');
start=[];
stop=[];
for ep = 1:length(session.epochs)
    if contains(session.epochs{ep}.environment,'Sleep','IgnoreCase', true)
        start = [start;session.epochs{ep}.startTime];
        stop = [stop;session.epochs{ep}.stopTime];
    end
end
% Find entries that contain the brain region keyword (case-insensitive)
idx = contains(cell_metrics.brainRegion, brain_region, 'IgnoreCase', true) & ...
      strcmp(cell_metrics.putativeCellType, 'Pyramidal Cell');;%&cell_metrics.shankID<5;

% Extract spike times (cell array of spike times)
spike_times = cell_metrics.spikes.times(idx);  
% Get the number of cells in ca1_spike_times
num_cells = numel(spike_times);
% Display the result
disp(['Number of cells in ca1_spike_times: ', num2str(num_cells)]);
% Concatenate all spike times into a single 1D array
st = cell2mat(spike_times(:));  

sleep_epoch=[start stop];
NREMLFP=GetAyaLFP(65);
NREM=SleepState.ints.NREMstate
fltered_NREM=NREM(((NREM(:,2)-NREM(:,1))>120),:);
NREM_LFP=Restrict(NREMLFP,fltered_NREM(:,:));
ripple_thresh=0.02;
spike_thresh=0.01;
srate=1250
[SIA_periods, zscored_signal] = detect_SIA(NREM_LFP, srate, spike_times, ripples.timestamps, spike_thresh, ripple_thresh, fltered_NREM);
% Save SIA periods to a MAT file
save('SIA_event.mat', 'SIA_periods');
swslope=[SleepState.detectorinfo.detectionparms.SleepScoreMetrics.t_clus  SleepState.detectorinfo.detectionparms.SleepScoreMetrics.broadbandSlowWave]; 
save('swslope.mat', 'swslope');
%%
plot_SIA_raster(NREM_LFP, srate, spike_times, SIA_periods)
function plot_SIA_raster(LFP, srate, spike_times, SIA_periods)
% PLOT_SIA_RASTER Plot raster and LFP with SIA annotation
% Input:
%   LFP          - Two-column matrix [timestamps, LFP values]
%   srate        - Sampling rate of the LFP signal (Hz)
%   spike_times  - Cell array of spike timestamps for each unit (1xN cell)
%   SIA_periods  - Detected SIA periods (start and end timestamps)

% Randomly select an SIA period
rand_idx = randi(size(SIA_periods, 1));
SIA_center = mean(SIA_periods(rand_idx, :));
epoch_start = SIA_center - 5;
epoch_end = SIA_center + 5;

% Extract LFP and downsample for plotting
lfp_idx = LFP(:, 1) >= epoch_start & LFP(:, 1) <= epoch_end;
lfp_time = LFP(lfp_idx, 1);
lfp_values = LFP(lfp_idx, 2);
downsample_factor = round(srate / 100); % Downsample to 100 Hz
lfp_time_ds = downsample(lfp_time, downsample_factor);
lfp_values_ds = downsample(lfp_values, downsample_factor);

% Plot LFP trace
figure;
subplot(2, 1, 1);
plot(lfp_time_ds, lfp_values_ds, 'k');
hold on;

% Highlight the selected SIA period
area([SIA_periods(rand_idx, 1), SIA_periods(rand_idx, 2)], [max(lfp_values), max(lfp_values)], 'FaceColor', [1, 0.7, 0.7], 'EdgeColor', 'none');
xlabel('Time (s)');
ylabel('LFP');
title('LFP Trace around SIA Period');
hold off;

% Plot raster plot
subplot(2, 1, 2);
hold on;
for unit = 1:length(spike_times)
    spikes = spike_times{unit};
    spikes_in_window = spikes(spikes >= epoch_start & spikes <= epoch_end);
    for s = 1:length(spikes_in_window)
        plot([spikes_in_window(s), spikes_in_window(s)], [unit - 0.4, unit + 0.4], 'k');
    end
end
xlabel('Time (s)');
ylabel('Unit');
title('Spike Raster Plot around SIA Period');
hold off;

end


