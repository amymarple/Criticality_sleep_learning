function [sizes, durations, sizesAR, durationsAR, lengths, F, FAR, freq, pxx_resamp, pxxAR_resamp, varphi, varNoise] = getARstats(st, windows, deltaT, quant, order, spec_props_also, pw_time)
%AR model fitting
[varphi, varNoise] = myYuleWalker(st, windows, deltaT, order);

%avalanche distributions
sigList = simARwindowed(varphi, varNoise, round((windows(:, 2)-windows(:, 1))/deltaT));
sizesAR = [];
durationsAR = [];
for i = 1:length(sigList)
    [sz, dur] = getAvalanches(sigList{i}', quant, 1);
    sizesAR = [sizesAR sz];
    durationsAR = [durationsAR dur];
end
avsExp = getAvShapes(windows, quant, deltaT, st, false, false, true, false);
sizes = avsExp.size;
durations = avsExp.duration;
durationsAR = durationsAR(sizesAR>=min(sizes));
sizesAR = sizesAR(sizesAR>=min(sizes));

%spectral properties
if spec_props_also
    pw_width = round(pw_time/deltaT);
    lengths = round(logspace(log10(4), log10(pw_width), 10*round(log10(pw_width/4))));
    F = myDFA(st, windows, deltaT, lengths);
    FAR = zeros(size(F));
    numBinsTot = zeros(size(F));
    for i = 1:length(sigList)
        [Fcurr, numBins] = DFA(sigList{i}, lengths);
        numBinsTot = numBinsTot+numBins;
        extra = Fcurr.*numBins;
        extra(isnan(extra)) = 0;
        FAR = FAR + extra;
    end
    FAR = FAR./numBinsTot;
    [pxx, freq, numUsed] = myPSD(st, windows, deltaT, pw_width);
    [pxxAR, freqAR, numUsed] = myPSD_sigList(sigList, deltaT, pw_width);
    freq_bins = logspace(log10(freq(2)), log10(freq(end)), round(10*log10(freq(end)/freq(2))));
    pxx_resamp = zeros(1, length(freq_bins)-1);
    pxxAR_resamp = zeros(1, length(freq_bins)-1);
    for i = 1:(length(freq_bins)-1)
        pxx_resamp(i) = mean(pxx(freq_bins(i)<=freq & freq<freq_bins(i+1)));
        pxxAR_resamp(i) = mean(pxxAR(freq_bins(i)<=freqAR & freqAR<freq_bins(i+1)));
    end
    freq = (freq_bins(2:end)+freq_bins(1:(end-1)))/2;
else
    lengths = [];
    F = [];
    FAR = [];
    freq = [];
    pxx_resamp = [];
    pxxAR_resamp = [];
end
end