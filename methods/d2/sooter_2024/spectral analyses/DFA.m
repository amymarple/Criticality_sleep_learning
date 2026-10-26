function [F, numBins] = DFA(sig, lengths)
sig = sig - mean(sig);
if size(sig, 2) > size(sig, 1)
    sig = sig';
end
sig = cumsum(sig);
F = zeros(length(lengths), 1);
numBins = zeros(length(lengths), 1);
for i = 1:length(lengths)
    currLength = lengths(i);
    start = 1:round(currLength/2):(length(sig)-currLength);%bins have 50% overlap
    currF = zeros(length(start), 1);
    for j = 1:length(start)
        currSig = sig(start(j):(start(j) + currLength-1));
        linFitParam = polyfit(1:length(currSig), currSig, 1);
        linFit = (1:length(currSig))'*linFitParam(1) + linFitParam(2);
        currSigDetr = currSig - linFit;
        currF(j) = std(currSigDetr);
    end
    F(i) = mean(currF);
    numBins(i) = length(start);
end