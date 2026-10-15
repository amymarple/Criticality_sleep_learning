% Control: second half of the pre-sleep period, used as baseline for comparison
clear all
basepath=pwd;
[~,basename] = fileparts(basepath);
savepath = 'D:\Documents\GitHub\R_Project\PPP\codebook\Criticality_analysis\ddc\data';
brain_region = 'CA1';
animal='mouse'
% Extract animal name and day from basepath
[path_parts] = split(basepath, filesep); % Split the basepath using file separator
animal_name = path_parts{4}; % Assuming 'PPP7' is the third part of the path
day = basename         % Assuming 'day7' is the fourth part of the path
% Construct the file name
load(fullfile(basepath, [basename, '.SleepState.states.mat']), 'SleepState');
load(fullfile(basepath, [basename, '.cell_metrics.cellinfo.mat']), 'cell_metrics');
load(fullfile(basepath, [basename, '.MergePoints.events.mat']), 'MergePoints');
load(fullfile(basepath, [basename, '.pulses.events']), 'pulses');
try
    load(fullfile(basepath, [basename, '.ripples.events.mat']), 'ripples');
catch
    ripples.peaks=[];
    ripples.frequency=[];
end
NREM=SleepState.ints.NREMstate;
% sleep_index=7;led_index=1;
% preled=Restrict(NREM,MergePoints.timestamps(led_index,:));
% pre=Restrict(NREM,MergePoints.timestamps(sleep_index,:));
pre=Restrict(NREM,[0,pulses.timestamps(1,1)])
preled=Restrict(NREM,[pulses.timestamps(1,1),MergePoints.timestamps(1,2)])

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

pre_total_duration = sum(pre(:,2) - pre(:,1));
preled_total_duration = sum(preled(:,2) - preled(:,1));

% Define target total duration as the minimum of the two.
target_duration = min(pre_total_duration, preled_total_duration);

% Function to trim intervals to achieve the target total duration.
trimIntervals = @(intervals, target) trimIntervalsFun(intervals, target);

trimmed_pre = trimIntervals(pre, target_duration);
trimmed_preled = trimIntervals(preled, target_duration);
[first_half, second_half] = splitPresleepIntervals(pre);
d2=struct();
%%
deltaT = 0.04;
order = 20;
check_fit = true;
quant = 0.1;
for whichState = [1, 2,3]

    if whichState == 1
        windows = trimmed_pre;
    elseif whichState == 2
        windows = trimmed_preled;
    elseif whichState == 3
        windows = second_half;
    end

    try
    [beta, d_beta, closest_models] = getDbetas(st', windows, deltaT, order, check_fit, quant);
    catch
        d_beta = [];
    end

    if whichState == 1
        disp(['sleep d2 = ', num2str(d_beta(1))])
        d2.pre=d_beta(1);
%         figure_filename = sprintf('Figure_wake_d2_%s_%s_%s.png', animal_name, day, brain_region)
%         saveas(gcf, fullfile(save_path, figure_filename));
%         close(gcf); % Close the figure to save memory
    elseif whichState == 2
        disp(['sleepled d2 = ', num2str(d_beta(1))])
        d2.preled=d_beta(1);
%         figure_filename = sprintf('Figure_nrem_d2_%s_%s_%s.png', animal_name, day, brain_region)
%         saveas(gcf, fullfile(save_path, figure_filename));
%         close(gcf); % Close the figure to save memory
%         close(gcf); % Close the figure to save memory
    elseif whichState == 3
        disp(['control d2 = ', num2str(d_beta(1))])
        d2.control=d_beta(1);
%         figure_filename = sprintf('Figure_nrem_d2_%s_%s_%s.png', animal_name, day, brain_region)
%         saveas(gcf, fullfile(save_path, figure_filename));
%         close(gcf); % Close the figure to save memory
    end
end

%%
save_animal_data(savepath, animal_name, day, brain_region, animal,d2)
function save_animal_data(savepath, animal_name, day, brain_region, animal,d2)
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
    mat_filename = fullfile(savepath, ['CA1_uLED_sleep', animal, '.mat']);
    field_name = sprintf('%s_%s_%s', animal_name, day, brain_region);

    % Create a struct with the data
    animal_data.d2 = d2;

    % Add ripple_rates if provided
    if exist('ripple_rates', 'var') && ~isempty(ripple_rates)
        animal_data.ripple_rates = ripple_rates;
    end

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