function [F] = myDFA(st, windows, deltaT, lengths)
    totF = zeros(length(lengths), 1);
    numBins = zeros(length(lengths), 1);
    sigAll = getFR(windows, st, deltaT);
    for i = 1:size(windows, 1)
        sig = getFR([windows(i, :)], st, deltaT);
        sig = (sig-mean(sigAll))/std(sigAll);
        [currF, numBinsCurr] = DFA(sig, lengths);
        extra = currF.*numBinsCurr;
        extra(isnan(extra)) = 0;
        totF = totF + extra;
        numBins = numBins + numBinsCurr;
    end
    F = totF./numBins;
    
end