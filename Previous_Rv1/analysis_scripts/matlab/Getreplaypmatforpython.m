% Settings
basepath = 'W:\data\PPP\PPP20\day4';  % or your folder
epoch = 5;   % k is your epoch number

left_all = {};    % initialize empty cell array
right_all = {};

% Loop through all i in this epoch
nEvents = length(replaytrajactory{1, epoch}.pMat);  % how many events

for i = 1:nEvents
    if isempty(replaytrajactory{1, epoch}.pMat{1,i})
        continue;  % skip empty ones
    end

    left_traj = replaytrajactory{1, epoch}.pMat{1,i}{1,1}.pMat;
    right_traj = replaytrajactory{1, epoch}.pMat{1,i}{1,2}.pMat;

    left_all{i} = left_traj;
    right_all{i} = right_traj;
end

% Save into a MAT file
save(fullfile(basepath, sprintf('epoch%d_replay_maps.mat', epoch)), 'left_all', 'right_all');


