% Find indices of CA1 neurons
function coarse_grain_ddc(basepath, window_size, step_size,deltaT)
[~,basename] = fileparts(basepath);
savepath = 'D:\Documents\GitHub\R_Project\PPP\codebook\Criticality_analysis\ddc\data\coarse_grain_bin_explore';
brain_region = 'CA1';
animal='mouse'
% Extract animal name and day from basepath
[path_parts] = split(basepath, filesep); % Split the basepath using file separator
animal_name = path_parts{4}; % Assuming 'PPP7' is the third part of the path
day = basename         % Assuming 'day7' is the fourth part of the path
% Construct the file name
load(fullfile(basepath, [basename, '.SleepState.states.mat']), 'SleepState');
load(fullfile(basepath, [basename, '.cell_metrics.cellinfo.mat']), 'cell_metrics');

try
    load(fullfile(basepath, [basename, '.ripples.events.mat']), 'ripples');
catch
    ripples.peaks=[];
    ripples.frequency=[];
end


% Find entries that contain the brain region keyword (case-insensitive)
idx = contains(cell_metrics.brainRegion, brain_region, 'IgnoreCase', true);%&cell_metrics.shankID<5;

% Extract spike times (cell array of spike times)
ca1_spike_times = cell_metrics.spikes.times(idx);  
% Get the number of cells in ca1_spike_times
num_cells = numel(ca1_spike_times);

% Display the result
disp(['Number of cells in ca1_spike_times: ', num2str(num_cells)]);
% Concatenate all spike times into a single 1D array
st = cell2mat(ca1_spike_times(:));  

% Sort spike times in ascending order
st = sort(st); 
NREM=SleepState.ints.NREMstate;
WAKE = double(SleepState.ints.WAKEstate);
NREM = NREM((NREM(:,2) - NREM(:,1)) > 120, :);
WAKE= WAKE((WAKE(:,2) - WAKE(:,1)) > 120, :);
wake_theta= Restrict(SleepState.ints.THETA,WAKE);
REM= SleepState.ints.REMstate;
ripples_inNREM=Restrict(ripples.peaks,NREM);
% Create the full save path
save_path = fullfile(basepath, 'complex_system');
if ~exist(save_path, 'dir')
    mkdir(save_path); % Create the directory if it doesn't exist
end
ddc_state=struct();
   
%%
% Convert NREM sleep intervals to double
NREM = double(SleepState.ints.NREMstate);


% Parameters for getDbetas
order = 20;
check_fit = false;
quant = 0.1;

% Initialize results
ex_variance=[];
d2 = [];        % Store d_beta(1) values
centers = [];   % Store center of each window
i=1
% Loop over each NREM epoch
for epoch = 1:size(NREM, 1)
    epoch_start = NREM(epoch, 1);
    epoch_end = NREM(epoch, 2);
 
    % Generate sliding windows
    windows = [];
    for start_time = epoch_start:step_size:(epoch_end - window_size)
        end_time = start_time + window_size;
        windows = [windows; start_time, end_time];  % Store window intervals
    end

    % Compute d_beta(1) for each window
    for w = 1:size(windows, 1)
        curr_window = windows(w, :);  
        center = mean(curr_window);  % Compute center timestamp
          if mod(w, 500) == 0&window_size>4
                check_fit = true;
          else
                check_fit = false;
          end
        % Compute branching deviation for this window
        % Assuming 'st' is your spike times and 'curr_window' is [start, stop]
        spike_times = st(st >= curr_window(1) & st < curr_window(2));
        
        % Bin into spike rate
        edges = curr_window(1):deltaT:curr_window(2);
        binned_counts = histcounts(spike_times, edges);
        
        % Check if the window is usable
        if length(binned_counts) < 10 || all(binned_counts == 0)
            warning('Skipping window: too few time bins or no spikes');
            continue;  % or set output to NaN/empty
        end
        [beta, d_beta, closest_models,explain_variance] = getDbetas(st', curr_window, deltaT, order, check_fit, quant);
        if mod(w, 500) == 0
            saveas(gcf, fullfile(save_path, ['Figure_' num2str(i) '_' num2str(window_size) '.png'])); % Save figure as PNG
            close(gcf); % Close the figure to save memory
        end
        % Store the first d_beta value
        d2 = [d2; d_beta(1)];
        ex_variance=[ex_variance; explain_variance];
        centers = [centers; center];
        i=i+1
    end
end

%% aggregates ripple_rate
swr_times=ripples_inNREM;
% Keep only valid entries
D2_times = centers;
D2_values = d2;
% Define window size (e.g., ±30s around each D2 center time)
rp_window_size =window_size/2;

% Initialize ripple rate storage
ripple_rates = zeros(size(D2_times));

% Aggregate SWR counts within window around each D2 time
for i = 1:length(D2_times)
    t_start = D2_times(i) - rp_window_size;
    t_end = D2_times(i) + rp_window_size;
    
    % Count SWRs within the window
    swr_count = sum(swr_times >= t_start & swr_times <= t_end);
    
    % Compute ripple rate (SWR count per second)
    ripple_rates(i) = swr_count / (2 * rp_window_size);
end
% Divide D2_values into sextiles
sextile_edges = quantile(D2_values, [0 1/6 2/6 3/6 4/6 5/6 1]); % Define sextiles
sextile_labels = 1:6; % Sextile indices
sextile_ripple_rate = zeros(1, 6); % To store mean ripple rate for each sextile
sextile_ripple_sem = zeros(1, 6);  % To store SEM for each sextile

% Compute mean and SEM ripple rate for each sextile
for i = 1:6
    in_sextile = D2_values >= sextile_edges(i) & D2_values < sextile_edges(i+1);
    sextile_ripple_rate(i) = mean(ripple_rates(in_sextile), 'omitnan'); % Mean ripple rate
    sextile_ripple_sem(i) = std(ripple_rates(in_sextile), 'omitnan') / sqrt(sum(in_sextile)); % SEM
end

% Plot sextile data with error bars
figure;
errorbar(sextile_labels, sextile_ripple_rate, sextile_ripple_sem, '-o', ...
         'LineWidth', 2, 'MarkerSize', 6, 'CapSize', 8);
xticks(sextile_labels); % Use sextile indices as x-axis ticks
xticklabels(compose('Sextile %d', sextile_labels)); % Label sextiles
xlabel('Sextiles of D_2 Value');
ylabel('Mean Ripple Rate (events/sec)');
title(sprintf('Ripple Rate by Sextiles of D_2 (SEM)\n%s - %s - Window %dsec delta_T %d0ms ', animal_name, brain_region, window_size,deltaT*100));
grid on;
% Construct the file name with the additional details
filename = sprintf('%s_%s_%s_win%ddelT%d_ripplerate_d2rate.fig', animal_name, day, brain_region, window_size,deltaT*100);
% Full path to save the figure
full_save_path = fullfile(savepath, filename);
saveas(gcf, full_save_path); % Save as MATLAB figure
disp(['Figure saved as: ' full_save_path]);
%%
save_animal_data(savepath, animal_name, day, brain_region, D2_values, ripple_rates,ex_variance,animal,centers,window_size,deltaT);
end
function save_animal_data(savepath, animal_name, day, brain_region, D2_values, ripple_rates,ex_variance,animal,centers,window_size,deltaT)
    % Function to save or update a MAT file with animal data.
    %
    % Inputs:
    %   savepath      - Directory where the MAT file will be saved
    %   animal_name   - Name of the animal (e.g., 'PPP7')
    %   day           - Experimental day (e.g., 'day7')
    %   brain_region  - Brain region (e.g., 'CA1')
    %   D2_values     - D2 values array
    %   barrage_rates - Barrage rates array
    %   ripple_rates  - Ripple rates array (optional)
    %
    % Example:
    %   save_animal_data('D:\path', 'PPP7', 'day7', 'CA1', D2_values, barrage_rates, ripple_rates);

    % Construct the file name and dynamic field name
    mat_filename = fullfile(savepath, ['20250324timebin_grained', animal, '.mat']);
    field_name = sprintf('%s_%s_%s_win%dsdeltaT%d', animal_name, day, brain_region, window_size*10,deltaT*100);
    % Create a struct with the data
    animal_data.D2_values = D2_values;
    animal_data.centers = centers;
    animal_data.explain_variance =ex_variance;
    animal_data.ripple_rates = ripple_rates;
    animal_data.window_size=window_size
    % Check if the MAT file already exists
    if isfile(mat_filename)
        % Load existing file
        loaded_data = load(mat_filename);

        % Check if the field already exists
        if isfield(loaded_data, field_name)
            disp(['Field "' field_name '" already exists. Replacing it with new data.']);
        else
            disp(['Adding new field "' field_name '" to the MAT file.']);
        end

        % Update the field in the loaded data
        loaded_data.(field_name) = animal_data;

        % Save the updated structure back to the file
        save(mat_filename, '-struct', 'loaded_data');
    else
        % Create a new file with the field
        disp(['MAT file does not exist. Creating new file with field "' field_name '".']);
        new_data.(field_name) = animal_data;
        save(mat_filename, '-struct', 'new_data');
    end

    % Display success message
    disp(['Data saved successfully in: ' mat_filename]);
end

