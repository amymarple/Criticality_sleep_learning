%% low high rippple
cd W:\data\PPP\PPP20\day4
time_resolved_d2_across_state
cd W:\data\PPP\PPP20\day5
time_resolved_d2_across_state
cd W:\data\PPP\PPP21\day5
time_resolved_d2_across_state
cd W:\data\PPP\PPP21\day4
time_resolved_d2_across_state
cd W:\data\PPP\PPP8\day8
time_resolved_d2_across_state
cd W:\data\PPP\PPP4\day11
time_resolved_d2_across_state
cd W:\data\PPP\PPP7\day7
time_resolved_d2_across_state
cd W:\data\PPP\PPP7\day12
time_resolved_d2_across_state
cd W:\data\PPP\PPP7\day10
time_resolved_d2_across_state
cd W:\data\PPP\PPP23\day7
time_resolved_d2_across_state
cd W:\data\PPP\PPP23\day8
time_resolved_d2_across_state
%%
cd W:\data\PPP\PPP20\day7
time_resolved_d2_across_state
cd W:\data\PPP\PPP20\day8
time_resolved_d2_across_state
cd W:\data\PPP\PPP21\day7
time_resolved_d2_across_state
cd W:\data\PPP\PPP21\day8
time_resolved_d2_across_state
cd W:\data\PPP\PPP23\day8
time_resolved_d2_across_state
cd W:\data\PPP\PPP23\day9
time_resolved_d2_across_state


cd W:\data\PPP\PPP8\day12
time_resolved_d2_across_state
cd W:\data\PPP\PPP8\day13
time_resolved_d2_across_state

cd W:\data\PPP\PPP7\day14
time_resolved_d2_across_state
%%  barr rate vs d2
cd X:\data\Barrage\NN8\day7 
time_resolved_d2_across_state
cd X:\data\Barrage\NN3\day15
time_resolved_d2_across_state

cd X:\data\Barrage\NN2\day24
time_resolved_d2_across_state

cd X:\data\Barrage\NN2\day10
time_resolved_d2_across_state

cd W:\data\PPP\CA2\PPP7\day12
time_resolved_d2_across_state

cd X:\data\Barrage\NN2\day10_3
time_resolved_d2_across_state

cd X:\data\Barrage\NN3\day12
time_resolved_d2_across_state

cd X:\data\Barrage\NN3\day16
time_resolved_d2_across_state

cd X:\data\Barrage\NN3\day25
time_resolved_d2_across_state

cd X:\data\Barrage\NN8\day6
time_resolved_d2_across_state

cd X:\data\Barrage\NN16\day13
time_resolved_d2_across_state
%% Barr ablate
cd X:\data\Barrage\NN8\day7 
time_resolved_d2_across_state_barrage_ablate
cd X:\data\Barrage\NN3\day15
time_resolved_d2_across_state_barrage_ablate

cd X:\data\Barrage\NN2\day24
time_resolved_d2_across_state_barrage_ablate

cd X:\data\Barrage\NN2\day10
time_resolved_d2_across_state_barrage_ablate

cd W:\data\PPP\CA2\PPP7\day12
time_resolved_d2_across_state_barrage_ablate

cd X:\data\Barrage\NN2\day10_3
time_resolved_d2_across_state_barrage_ablate

cd X:\data\Barrage\NN3\day12
time_resolved_d2_across_state_barrage_ablate

cd X:\data\Barrage\NN3\day16
time_resolved_d2_across_state_barrage_ablate

cd X:\data\Barrage\NN3\day25
time_resolved_d2_across_state_barrage_ablate

cd X:\data\Barrage\NN8\day6
time_resolved_d2_across_state_barrage_ablate

cd X:\data\Barrage\NN16\day13
time_resolved_d2_across_state_barrage_ablate
%% ripple generation
addpath(genpath('D:/Documents/GitHub/Sleep_criticaltiy/notebooks/helper/time_resolved_d2_across_state_EMG.mat'));
savepath;   % optional: persist for future sessions
cd X:\data\Barrage\NN8\day15
time_resolved_d2_across_state_EMG
cd X:\data\Barrage\NN8\day20
time_resolved_d2_across_state_EMG
cd X:\data\Barrage\NN8\day24
time_resolved_d2_across_state_EMG
cd X:\data\Barrage\NN8\day25
time_resolved_d2_across_state_EMG

cd X:\data\Barrage\NN9\day21
time_resolved_d2_across_state_EMG
cd X:\data\Barrage\NN10\day17
time_resolved_d2_across_state_EMG
cd X:\data\Barrage\NN10\day20
time_resolved_d2_across_state_EMG
cd X:\data\Barrage\NN12\day15
time_resolved_d2_across_state_EMG
cd X:\data\Barrage\NN12\day21
time_resolved_d2_across_state_EMG

%%
% Load the original .mat file
basepath=pwd;
[parent, ~] = fileparts(basepath);     % strip 'complex_system'
[~, basename] = fileparts(parent);     % -> 'day20'

fname = sprintf('%s_CA1_D2_time_resolved_STATELESS.mat', basename);
data  = load(fname);   % or just load(fname) if cd'ed there)

% Extract the table
timeline = data.timeline;

% Save as top-level variables for Python
center_s = timeline.center_s;
d2 = timeline.d2;

% Save them directly (no struct)
save(sprintf('%s_CA1_D2_center_d2.mat', basename), 'center_s', 'd2', '-v7');
%%  barr manipulation
% cd X:\data\Barrage\SNCG06\day17
% time_resolved_d2_across_state
% cd X:\data\Barrage\SNCG06\day21
% time_resolved_d2_across_state
% cd X:\data\Barrage\SNCG06\day21b
% time_resolved_d2_across_state
% cd X:\data\Barrage\SNCG06\day22
% time_resolved_d2_across_state
% cd X:\data\Barrage\SNCG06\day23
% time_resolved_d2_across_state
% cd X:\data\Barrage\SNCG06\day27
% time_resolved_d2_across_state
cd X:\data\Barrage\SNCG08\day10
time_resolved_d2_across_state_error_bar
cd X:\data\Barrage\SNCG08\day14
time_resolved_d2_across_state_error_bar
cd X:\data\Barrage\SNCG08\day16
time_resolved_d2_across_state_error_bar
cd X:\data\Barrage\SNCG08\day17
time_resolved_d2_across_state_error_bar
cd X:\data\Barrage\SNCG09\day10
time_resolved_d2_across_state_error_bar
cd X:\data\Barrage\SNCG09\day12
time_resolved_d2_across_state_error_bar
cd X:\data\Barrage\SNCG09\day13
time_resolved_d2_across_state_error_bar

cd X:\data\Barrage\SNCG06\day17
time_resolved_d2_across_state_error_bar
cd X:\data\Barrage\SNCG06\day21
time_resolved_d2_across_state_error_bar
cd X:\data\Barrage\SNCG06\day21b
time_resolved_d2_across_state_error_bar
cd X:\data\Barrage\SNCG06\day22
time_resolved_d2_across_state_error_bar
cd X:\data\Barrage\SNCG06\day23
time_resolved_d2_across_state_error_bar
cd X:\data\Barrage\SNCG06\day27
time_resolved_d2_across_state_error_bar
%% uLED 
cd X:\data\Barrage\LAK1\day5
time_resolved_d2_across_state_error_bar
cd X:\data\Barrage\NN13\day31
time_resolved_d2_across_state_error_bar
cd X:\data\Barrage\NN13\day23
time_resolved_d2_across_state_error_bar
cd X:\data\Barrage\NN13\day29
time_resolved_d2_across_state_error_bar
cd X:\data\Barrage\NN3\day12
time_resolved_d2_across_state_error_bar
cd X:\data\Barrage\NN2\day22
time_resolved_d2_across_state_error_bar

%%  ripple generation 
cd X:\data\Barrage\NN8\day15
time_resolved_d2_across_state_error_bar
cd X:\data\Barrage\NN8\day20
time_resolved_d2_across_state_error_bar
cd X:\data\Barrage\NN8\day24
time_resolved_d2_across_state_error_bar
cd X:\data\Barrage\NN8\day25
time_resolved_d2_across_state_error_bar

cd X:\data\Barrage\NN9\day21
time_resolved_d2_across_state_error_bar
cd X:\data\Barrage\NN10\day17
time_resolved_d2_across_state_error_bar
cd X:\data\Barrage\NN10\day20
time_resolved_d2_across_state_error_bar
cd X:\data\Barrage\NN12\day15
time_resolved_d2_across_state_error_bar
cd X:\data\Barrage\NN12\day21
time_resolved_d2_across_state_error_bar
%% ripple generation
cd X:\data\Barrage\NN8\day15
time_resolved_d2_across_state_error_bar
cd X:\data\Barrage\NN8\day20
time_resolved_d2_across_state_error_bar
cd X:\data\Barrage\NN8\day24
time_resolved_d2_across_state_error_bar
cd X:\data\Barrage\NN8\day25
time_resolved_d2_across_state_error_bar

cd X:\data\Barrage\NN9\day21
time_resolved_d2_across_state_error_bar
cd X:\data\Barrage\NN10\day17
time_resolved_d2_across_state_error_bar
cd X:\data\Barrage\NN10\day20
time_resolved_d2_across_state_error_bar
cd X:\data\Barrage\NN12\day15
time_resolved_d2_across_state_error_bar
cd X:\data\Barrage\NN12\day21
time_resolved_d2_across_state_error_bar

%% error bar 
%% low high rippple
cd W:\data\PPP\PPP20\day4
time_resolved_d2_across_state
cd W:\data\PPP\PPP20\day5
time_resolved_d2_across_state
cd W:\data\PPP\PPP21\day5
time_resolved_d2_across_state
cd W:\data\PPP\PPP21\day4
time_resolved_d2_across_state
cd W:\data\PPP\PPP8\day8
time_resolved_d2_across_state
cd W:\data\PPP\PPP4\day11
time_resolved_d2_across_state
cd W:\data\PPP\PPP7\day7
time_resolved_d2_across_state
cd W:\data\PPP\PPP7\day12
time_resolved_d2_across_state
cd W:\data\PPP\PPP7\day10
time_resolved_d2_across_state
cd W:\data\PPP\PPP23\day7
time_resolved_d2_across_state
cd W:\data\PPP\PPP23\day8
time_resolved_d2_across_state
%%
cd W:\data\PPP\PPP20\day4
time_resolved_d2_across_state_error_bar
cd W:\data\PPP\PPP20\day5
time_resolved_d2_across_state_error_bar
cd W:\data\PPP\PPP21\day5
time_resolved_d2_across_state_error_bar
cd W:\data\PPP\PPP21\day4
time_resolved_d2_across_state_error_bar
cd W:\data\PPP\PPP8\day8
time_resolved_d2_across_state_error_bar
cd W:\data\PPP\PPP4\day11
time_resolved_d2_across_state_error_bar
cd W:\data\PPP\PPP7\day7
time_resolved_d2_across_state_error_bar
cd W:\data\PPP\PPP7\day8
time_resolved_d2_across_state_error_bar
cd W:\data\PPP\PPP7\day12
time_resolved_d2_across_state_error_bar
cd W:\data\PPP\PPP7\day10
time_resolved_d2_across_state_error_bar
cd W:\data\PPP\PPP23\day7
time_resolved_d2_across_state_error_bar
cd W:\data\PPP\PPP23\day8
time_resolved_d2_across_state_error_bar
cd X:\data\Barrage\NN8\day7 
time_resolved_d2_across_state_error_bar
cd X:\data\Barrage\NN3\day15
time_resolved_d2_across_state_error_bar

cd X:\data\Barrage\NN2\day24
time_resolved_d2_across_state_error_bar

cd X:\data\Barrage\NN2\day10
time_resolved_d2_across_state_error_bar

cd W:\data\PPP\CA2\PPP7\day12
time_resolved_d2_across_state_error_bar

cd X:\data\Barrage\NN2\day10_3
time_resolved_d2_across_state_error_bar

cd X:\data\Barrage\NN3\day12
time_resolved_d2_across_state_error_bar

cd X:\data\Barrage\NN3\day16
time_resolved_d2_across_state_error_bar

cd X:\data\Barrage\NN3\day25
time_resolved_d2_across_state_error_bar

cd X:\data\Barrage\NN8\day6
time_resolved_d2_across_state_error_bar

cd X:\data\Barrage\NN16\day13
time_resolved_d2_across_state_error_bar
%% generalization
cd W:\data\PPP\PPP20\day7
time_resolved_d2_across_state_error_bar
cd W:\data\PPP\PPP20\day8
time_resolved_d2_across_state_error_bar
cd W:\data\PPP\PPP20\day9
time_resolved_d2_across_state_error_bar
cd W:\data\PPP\PPP20\day13
time_resolved_d2_across_state_error_bar
cd W:\data\PPP\PPP20\day14
time_resolved_d2_across_state_error_bar
cd W:\data\PPP\PPP21\day7
time_resolved_d2_across_state_error_bar
cd W:\data\PPP\PPP21\day8
time_resolved_d2_across_state_error_bar
cd W:\data\PPP\PPP21\day9
time_resolved_d2_across_state_error_bar
cd W:\data\PPP\PPP21\day13
time_resolved_d2_across_state_error_bar
cd W:\data\PPP\PPP21\day14
time_resolved_d2_across_state_error_bar
cd W:\data\PPP\PPP23\day8
time_resolved_d2_across_state_error_bar
cd W:\data\PPP\PPP23\day9
time_resolved_d2_across_state_error_bar
cd W:\data\PPP\PPP23\day10
time_resolved_d2_across_state_error_bar
cd W:\data\PPP\PPP23\day11
time_resolved_d2_across_state_error_bar
cd W:\data\PPP\PPP23\day12
time_resolved_d2_across_state_error_bar
cd W:\data\PPP\PPP8\day12
time_resolved_d2_across_state_error_bar
cd W:\data\PPP\PPP8\day13
time_resolved_d2_across_state_error_bar
cd W:\data\PPP\PPP8\day15
time_resolved_d2_across_state_error_bar
cd W:\data\PPP\PPP20\day3
time_resolved_d2_across_state_error_bar
cd W:\data\PPP\PPP21\day3
time_resolved_d2_across_state_error_bar
cd W:\data\PPP\PPP23\day4
time_resolved_d2_across_state_error_bar
cd W:\data\PPP\PPP8\day6
time_resolved_d2_across_state_error_bar
cd W:\data\PPP\PPP4\day8
time_resolved_d2_across_state_error_bar
cd W:\data\PPP\PPP7\day14
time_resolved_d2_across_state_error_bar

%%
cd W:\data\PPP\PPP20\day10
time_resolved_d2_across_state_error_bar
cd W:\data\PPP\PPP20\day15
time_resolved_d2_across_state_error_bar
cd W:\data\PPP\PPP21\day10
time_resolved_d2_across_state_error_bar
cd W:\data\PPP\PPP21\day15
time_resolved_d2_across_state_error_bar
%% loop
cd W:\data\PPP\PPP20\day4
time_resolved_d2_across_state_ripple_ablate

cd W:\data\PPP\PPP20\day5
time_resolved_d2_across_state_ripple_ablate

cd W:\data\PPP\PPP21\day5
time_resolved_d2_across_state_ripple_ablate

cd W:\data\PPP\PPP21\day4
time_resolved_d2_across_state_ripple_ablate

cd W:\data\PPP\PPP8\day8
time_resolved_d2_across_state_ripple_ablate

cd W:\data\PPP\PPP4\day11
time_resolved_d2_across_state_ripple_ablate

cd W:\data\PPP\PPP7\day7
time_resolved_d2_across_state_ripple_ablate

cd W:\data\PPP\PPP7\day8
time_resolved_d2_across_state_ripple_ablate

cd W:\data\PPP\PPP7\day12
time_resolved_d2_across_state_ripple_ablate

cd W:\data\PPP\PPP7\day10
time_resolved_d2_across_state_ripple_ablate

cd W:\data\PPP\PPP23\day7
time_resolved_d2_across_state_ripple_ablate

cd W:\data\PPP\PPP23\day8
time_resolved_d2_across_state_ripple_ablate

cd W:\data\PPP\PPP20\day3
time_resolved_d2_across_state_ripple_ablate
cd W:\data\PPP\PPP21\day3
time_resolved_d2_across_state_ripple_ablate

%% cck disruption
cd X:\data\Barrage\SNCG01\day16
time_resolved_d2_across_state_error_bar

cd X:\data\Barrage\SNCG02\day16
time_resolved_d2_across_state_error_bar

cd X:\data\Barrage\SNCG02\day17
time_resolved_d2_across_state_error_bar

cd X:\data\Barrage\SNCG01\day17
time_resolved_d2_across_state_error_bar

cd X:\data\Barrage\SNCG02\day19
time_resolved_d2_across_state_error_bar

cd X:\data\Barrage\SNCG01\day19
time_resolved_d2_across_state_error_bar

cd X:\data\Barrage\SNCG01\day22
time_resolved_d2_across_state_error_bar

cd X:\data\Barrage\SNCG04\day8
time_resolved_d2_across_state_error_bar

%% =====================  rat HP17  ==========================
base17 = 'V:\data\hpc_ctx_project\HP17';

folders17 = {
    'hp17_day2_20250413'
    'hp17_day23_20250507'
    'hp17_day26_20250510'
    'hp17_day32_20250516'
    'hp17_day34_20250518'
    'hp17_day36_20250520'
    'hp17_day40_20250526'
    'hp17_day43_20250529'
    'hp17_day49_20250604'
};

for i = 1:length(folders17)
    session_path = fullfile(base17, folders17{i});
    fprintf('Processing: %s\n', session_path);

    cd(session_path);
    try
        time_resolved_d2_across_state_error_bar;
        fprintf(' ✓ Finished %s\n', folders17{i});
    catch ME
        fprintf(' ✗ Error in %s: %s\n', folders17{i}, ME.message);
    end
end


% =====================  HP18  ==========================
base18 = 'V:\data\hpc_ctx_project\HP18';

folders18 = {
    'hp18_day13_20250417'
    'hp18_day16_20250420'
    'hp18_day17_20250421'
    'hp18_day36_20250510'
    'hp18_day40_20250514'
    'hp18_day48_20250522'
    'hp18_day49_20250523'
    'hp18_day51_20250525'
    'hp18_day52_20250526'
    'hp18_day55_20250529'
};

for i = 1:length(folders18)
    session_path = fullfile(base18, folders18{i});
    fprintf('Processing: %s\n', session_path);

    cd(session_path);
    try
        time_resolved_d2_across_state_error_bar;
        fprintf(' ✓ Finished %s\n', folders18{i});
    catch ME
        fprintf(' ✗ Error in %s: %s\n', folders18{i}, ME.message);
    end
end

%% presleep midsleep postsleep
cd W:\data\PPP\PPP4\day8
time_resolved_d2_across_state_error_bar
cd W:\data\PPP\PPP4\day11
time_resolved_d2_across_state_error_bar
cd W:\data\PPP\PPP8\day7
time_resolved_d2_across_state_error_bar
cd W:\data\PPP\PPP21\day3
time_resolved_d2_across_state_error_bar
cd W:\data\PPP\PPP20\day3
time_resolved_d2_across_state_error_bar
%% ablate ripples
cd X:\data\Barrage\LAK1\day5
time_resolved_d2_across_state_ripple_ablate
cd X:\data\Barrage\NN13\day23
time_resolved_d2_across_state_ripple_ablate
cd X:\data\Barrage\NN13\day31
time_resolved_d2_across_state_ripple_ablate
cd X:\data\Barrage\NN13\day29
time_resolved_d2_across_state_ripple_ablate
cd X:\data\Barrage\NN3\day12
time_resolved_d2_across_state_ripple_ablate
cd X:\data\Barrage\NN2\day22
time_resolved_d2_across_state_ripple_ablate
%% state transation 
cd W:\data\PPP\PPP20\day4
time_resolved_d2_state_transition

cd W:\data\PPP\PPP20\day5
time_resolved_d2_state_transition

cd W:\data\PPP\PPP21\day5
time_resolved_d2_state_transition

cd W:\data\PPP\PPP21\day4
time_resolved_d2_state_transition

cd W:\data\PPP\PPP8\day8
time_resolved_d2_state_transition

cd W:\data\PPP\PPP4\day11
time_resolved_d2_state_transition

cd W:\data\PPP\PPP7\day7
time_resolved_d2_state_transition

cd W:\data\PPP\PPP7\day8
time_resolved_d2_state_transition

cd W:\data\PPP\PPP7\day12
time_resolved_d2_state_transition

cd W:\data\PPP\PPP7\day10
time_resolved_d2_state_transition

cd W:\data\PPP\PPP23\day7
time_resolved_d2_state_transition

cd W:\data\PPP\PPP23\day8
time_resolved_d2_state_transition

cd W:\data\PPP\PPP20\day3
time_resolved_d2_state_transition

cd W:\data\PPP\PPP21\day3
time_resolved_d2_state_transition
%% ablate ripple state transition
cd W:\data\PPP\PPP20\day4
time_resolved_d2_across_state_ripple_ablate_state_transition

cd W:\data\PPP\PPP20\day5
time_resolved_d2_across_state_ripple_ablate_state_transition

cd W:\data\PPP\PPP21\day5
time_resolved_d2_across_state_ripple_ablate_state_transition

cd W:\data\PPP\PPP21\day4
time_resolved_d2_across_state_ripple_ablate_state_transition

cd W:\data\PPP\PPP8\day8
time_resolved_d2_across_state_ripple_ablate_state_transition

cd W:\data\PPP\PPP4\day11
time_resolved_d2_across_state_ripple_ablate_state_transition

cd W:\data\PPP\PPP7\day7
time_resolved_d2_across_state_ripple_ablate_state_transition

cd W:\data\PPP\PPP7\day8
time_resolved_d2_across_state_ripple_ablate_state_transition

cd W:\data\PPP\PPP7\day12
time_resolved_d2_across_state_ripple_ablate_state_transition

cd W:\data\PPP\PPP7\day10
time_resolved_d2_across_state_ripple_ablate_state_transition

cd W:\data\PPP\PPP23\day7
time_resolved_d2_across_state_ripple_ablate_state_transition

cd W:\data\PPP\PPP23\day8
time_resolved_d2_across_state_ripple_ablate_state_transition

cd W:\data\PPP\PPP20\day3
time_resolved_d2_across_state_ripple_ablate_state_transition

cd W:\data\PPP\PPP21\day3
time_resolved_d2_across_state_ripple_ablate_state_transition

%%
cd Z:\Data\AYAold\AB4\day07
time_resolved_d2_across_state_error_bar

cd Z:\Data\AYAold\AB4\day08
time_resolved_d2_across_state_error_bar

cd Z:\Data\AYAold\AB4\day09
time_resolved_d2_across_state_error_bar

cd Z:\Data\AYAold\AB4\day11
time_resolved_d2_across_state_error_bar

cd Z:\Data\AYAold\AYA6\day17
time_resolved_d2_across_state_error_bar

cd Z:\Data\AYAold\AYA6\day19
time_resolved_d2_across_state_error_bar

cd Z:\Data\AYAold\AYA6\day20
time_resolved_d2_across_state_error_bar

cd Z:\Data\AYAold\AYA7\day24
time_resolved_d2_across_state_error_bar

cd Z:\Data\AYAold\AYA7\day25
time_resolved_d2_across_state_error_bar

cd Z:\Data\AYAold\AYA9\day15
time_resolved_d2_across_state_error_bar

cd Z:\Data\AYAold\AYA9\day16
time_resolved_d2_across_state_error_bar

cd Z:\Data\AYAold\AYA9\day17
time_resolved_d2_across_state_error_bar

cd Z:\Data\AYAold\AYA9\day20
time_resolved_d2_across_state_error_bar

cd Z:\Data\AYAold\AYA10\day12
time_resolved_d2_across_state_error_bar

cd Z:\Data\AYAold\AYA10\day25
time_resolved_d2_across_state_error_bar

cd Z:\Data\AYAold\AYA10\day27
time_resolved_d2_across_state_error_bar

cd Z:\Data\AYAold\AYA10\day31
time_resolved_d2_across_state_error_bar

cd Z:\Data\AYAold\AYA10\day32
time_resolved_d2_across_state_error_bar

cd Z:\Data\AYAold\AYA10\day34
time_resolved_d2_across_state_error_bar

%% generalizatioN

cd U:\data\HMC\HMC3\day15
time_resolved_d2_across_state_error_bar
cd U:\data\HMC\HMC3\day16
time_resolved_d2_across_state_error_bar

cd S:\data\latentseq\RM04\rm04_day20_20250728
time_resolved_d2_across_state_error_bar
cd S:\data\latentseq\RM04\rm04_day16_20250723
time_resolved_d2_across_state_error_bar

cd S:\data\latentseq\RM04\rm04_day16_20250723
time_resolved_d2_across_state_error_bar

cd U:\data\HMC\HMC2\day8
time_resolved_d2_across_state_error_bar
cd U:\data\HMC\HMC2\day9
time_resolved_d2_across_state_error_bar
cd S:\data\latentseq\RM04\rm04_day12_20250710
time_resolved_d2_across_state_error_bar
cd S:\data\latentseq\RM04\rm04_day14_20250712
time_resolved_d2_across_state_error_bar


cd S:\data\latentseq\RM10\rm10_day4_20250808
time_resolved_d2_across_state_error_bar
cd S:\data\latentseq\RM10\rm10_day5_20250809
time_resolved_d2_across_state_error_bar
cd S:\data\latentseq\RM10\rm10_day6_20250810
time_resolved_d2_across_state_error_bar
cd S:\data\latentseq\RM10\rm10_day10_20250814
time_resolved_d2_across_state_error_bar
cd S:\data\latentseq\RM10\rm10_day11_20250815
time_resolved_d2_across_state_error_bar
cd S:\data\latentseq\RM10\rm10_day12_20250817
time_resolved_d2_across_state_error_bar

%%
cd('Z:\Data\GrosmarkAD\Achilles\Achilles_10252013')
time_resolved_d2_across_state_error_bar

cd('Z:\Data\GrosmarkAD\Achilles\Achilles_11012013')
time_resolved_d2_across_state_error_bar

cd('Z:\Data\GrosmarkAD\Buddy\Buddy_06272013')
time_resolved_d2_across_state_error_bar

cd('Z:\Data\GrosmarkAD\Cicero\Cicero_09012014')
time_resolved_d2_across_state_error_bar

cd('Z:\Data\GrosmarkAD\Cicero\Cicero_09102014')
time_resolved_d2_across_state_error_bar

cd('Z:\Data\GrosmarkAD\Cicero\Cicero_09172014')
time_resolved_d2_across_state_error_bar

cd('Z:\Data\GrosmarkAD\Gatsby\Gatsby_08022013')
time_resolved_d2_across_state_error_bar

cd('Z:\Data\GrosmarkAD\Gatsby\Gatsby_08282013')
time_resolved_d2_across_state_error_bar

cd('U:\data\HMC\HMC1\day8')
time_resolved_d2_across_state_error_bar

cd('Z:\Data\Kenji\ec014.459_480')
time_resolved_d2_across_state_error_bar

%%
cd V:\data\hpc_ctx_project\HP15\hp15_day31_20250130
time_resolved_d2_across_state_error_bar
cd V:\data\hpc_ctx_project\HP18\hp18_day52_20250526
time_resolved_d2_across_state_error_bar
cd V:\data\hpc_ctx_project\HP18\hp18_day55_20250529
time_resolved_d2_across_state_error_bar
cd V:\data\hpc_ctx_project\HP15\hp15_day35_20250205
time_resolved_d2_across_state_error_bar
cd S:\data\latentseq\RM10\rm10_day27_20250908
time_resolved_d2_across_state_error_bar
cd S:\data\latentseq\RM10\rm10_day28_29250909
time_resolved_d2_across_state_error_bar
%%
cd U:\data\HMC\HMC1\day8
time_resolved_d2_across_state_error_bar

cd U:\data\HMC\HMC1\day9
time_resolved_d2_across_state_error_bar

cd U:\data\HMC\HMC1\day10
time_resolved_d2_across_state_error_bar

cd U:\data\HMC\HMC1\day11
time_resolved_d2_across_state_error_bar

cd U:\data\HMC\HMC1\day12
time_resolved_d2_across_state_error_bar

cd U:\data\HMC\HMC1\day13
time_resolved_d2_across_state_error_bar

cd W:\data\PPP\PPP8\day14
time_resolved_d2_across_state_error_bar

