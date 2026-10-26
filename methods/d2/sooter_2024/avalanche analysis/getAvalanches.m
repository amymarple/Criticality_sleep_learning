function [sizes, durations] = getAvalanches(timeSeries, quant, above_thresh)
%getAvalanches takes a time series of data and returns a list of avalanche
%sizes

if ~(above_thresh == 0 | above_thresh == 1)
    disp('Warning. Invalid value of above_thresh')
end
thresh = quantile(timeSeries, quant);

below = find(timeSeries <= thresh);

sizes = [];
durations = [];
count = 1;

for i = 1:(size(below, 2) - 1)
    lower = below(i);
    upper = below(i + 1);
    if (upper - lower) > 1
        sizes(count) = sum(timeSeries((lower + 1):(upper - 1)) - above_thresh*thresh);
        durations(count) = upper - lower - 1;
        count = count + 1;
    end
end
end