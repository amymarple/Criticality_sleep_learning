function [sig] = getFR(windows, spks, deltaT)
sig = [];
for i = 1:size(windows, 1)
    currSpks = spks(windows(i, 1) <= spks & spks < windows(i, 2));
    bins = windows(i, 1):deltaT:windows(i, 2);
    currSig = histc(currSpks, bins);
    sig = [sig currSig(1:(end-1))];
end
end