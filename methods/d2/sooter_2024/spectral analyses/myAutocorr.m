function [lags, acf] = myAutocorr(st, windows, deltaT, maxLag)
    acf = zeros(1, maxLag);
    counts = zeros(1, maxLag);
    sigAll = getFR(windows, st, deltaT);
    for i = 1:size(windows, 1)
        sig = getFR([windows(i, :)], st, deltaT);
        sig = (sig-mean(sigAll))/std(sigAll);
        for j = 1:min([length(acf) length(sig)-2])
            acf(j) = acf(j) + sum(sig(1:(length(sig)-j)).*sig((1+j):end));
            counts(j) = counts(j) + length(sig)-j;
        end
    end
    lags = (1:maxLag)*deltaT;
    acf = acf./counts;
    acf = (acf-0)/1; %since signal has been normalized
end