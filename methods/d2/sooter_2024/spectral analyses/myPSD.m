function [pxx, freq, numUsed] = myPSD(st, windows, deltaT, pw_width)
%pw_width in number of samples

count = 0;
numUsed = 0;
sigAll = getFR(windows, st, deltaT);

for i = 1:size(windows, 1)
    sig = getFR([windows(i,:)], st, deltaT);
    
    sig = (sig-mean(sigAll))/std(sigAll);
    if length(sig) > pw_width
        [pxxCurr, freqCurr] = pwelch(sig, pw_width, [], [], 1/deltaT);
        numWindows = floor(length(sig)/(pw_width/2));
        if numUsed == 0
            freq = freqCurr;
            pxx = pxxCurr*numWindows;
        else
            pxx = pxx + pxxCurr*numWindows;
        end
        count = count + numWindows;
        numUsed = numUsed + 1;
    end
end
pxx = pxx/count;
end