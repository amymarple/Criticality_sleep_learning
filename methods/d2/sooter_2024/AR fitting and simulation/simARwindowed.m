function [sigList] = simARwindowed(varphi, varNoise, window_lengths)

window_lengths = window_lengths(window_lengths>0);
sigList = cell(length(window_lengths), 1);
for i = 1:length(window_lengths)
    sigList{i} = simAR(varphi, varNoise, window_lengths(i));
end

end